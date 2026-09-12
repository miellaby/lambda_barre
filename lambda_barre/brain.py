"""The brain of λ̄: orchestrates the world model and the policy.

This wires the "IA à 2 modèles" (``models.py``) into the live loop. During
*éveil* (wake) the policy drives the actuator consignes and every transition
(state, action, next-state) is journaled into an experience buffer. During
*sommeil* (sleep) the brain trains offline:

  1. the world model learns to predict the signal slots of the next token
     across a trajectory of SEQ_STEPS consecutive transitions (teacher forcing,
     salve-within-type attention mask);
  2. the policy is trained by reinforcement learning on *imagined*
     trajectories: it samples an action, the (frozen) world model rolls the
     state forward, and the policy is pushed (REINFORCE) toward actions whose
     imagined trajectories have low predicted cost.

That is the model-based RL loop of `Lambda barre.md`: the policy learns from
the world model's anticipation of consequences, not from direct reward.

A salve is a list of 16 dense tokens (13 sensory + 3 motor), each a 25-float
vector (see ``tokenize``). All tensors live on CPU. The buffer stores plain
Python lists of tokens (list[list[float]]).
"""
from __future__ import annotations

import os
import random
import time
from collections import deque

import torch

from . import models as M
from .models import build_salve_mask
from .tokenize import (STATE_TOKENS, ACTION_TOKENS, SALVE_TOKENS, DENSE_DIM,
                       N_SIGNAL, _N_SIGNALS, N_POLICY_STATE, SIG_OFFSET,
                       _COST_IDX, _REWARD_IDX, _COST_KEYS, _COST_WEIGHTS,
                       _BY_KEY, _LAYOUT, _prefix,
                       state_tokens, action_tokens)


# Fixed layout of a training sequence: a trajectory of SEQ_STEPS transitions.
# [s0, a0, s1, a1, ..., a_{N-1}, sN] = (state + action) * N + state tokens.
SEQ_STEPS = 10
_SEQ_LEN = SEQ_STEPS * SALVE_TOKENS + STATE_TOKENS          # 173

# Precomputed cost metadata (from the dense layout) for the vectorized
# trajectory-cost: 5 unsigned cost signals (token 11) + 1 signed confort
# signal (token 12). Denormalize: unsigned -> t*scale, signed -> (2t-1)*scale.
_COST_SCALES = torch.tensor([_BY_KEY[k].scale for k in _COST_KEYS],
                            dtype=torch.float32)              # [5]
_COST_SIGNED = torch.tensor([_BY_KEY[k].signed for k in _COST_KEYS],
                            dtype=torch.bool)                  # [5]
_COST_W = torch.tensor([_COST_WEIGHTS[k] for k in _COST_KEYS],
                      dtype=torch.float32)                    # [5]
_CONFORT_SCALE = _BY_KEY["confort"].scale
_CONFORT_W = _COST_WEIGHTS["confort"]

# Flat signal-grid indices for the 47 policy scalars: tokens 0..10, first
# _N_SIGNALS[i] slots each. Used to gather them in one op from a [B, 13, 16]
# signal grid.
_POL_SCALAR_IDX = []
for _i in range(_COST_IDX):               # tokens 0..10
    for _k in range(_N_SIGNALS[_i]):
        _POL_SCALAR_IDX.append(_i * N_SIGNAL + _k)
assert len(_POL_SCALAR_IDX) == N_POLICY_STATE  # 47

# Action encoding metadata: 5 consignes → (token-within-action-block, slot,
# scale, signed). The action block is tokens 13..15 (3 tokens).
_ACT_KEYS = ("limb_l_theta", "limb_l_d", "limb_r_theta", "limb_r_d",
             "tail_theta")
_ACT_LAYOUT = []   # (tok_in_block, slot)
for _i in range(STATE_TOKENS, SALVE_TOKENS):    # 13, 14, 15
    for _k, _key in enumerate(_LAYOUT[_i].signals):
        _ACT_LAYOUT.append((_i - STATE_TOKENS, _k, _key))
_ACT_TOK = torch.tensor([t for t, _, _ in _ACT_LAYOUT], dtype=torch.long)
_ACT_SLOT = torch.tensor([s for _, s, _ in _ACT_LAYOUT], dtype=torch.long)
_ACT_SCALES = torch.tensor([_BY_KEY[k].scale for _, _, k in _ACT_LAYOUT],
                           dtype=torch.float32)
_ACT_SIGNED = torch.tensor([_BY_KEY[k].signed for _, _, k in _ACT_LAYOUT],
                           dtype=torch.bool)
# Template: 3 action tokens with structural prefix, zeroed signals.
_ACT_TEMPLATE = torch.tensor(
    [_prefix(_LAYOUT[i]) + [0.0] * N_SIGNAL
     for i in range(STATE_TOKENS, SALVE_TOKENS)], dtype=torch.float32)

# Per-target-position valid-slot mask: position p predicts token p+1, whose
# token index within a salve is (p+1) % 16 — only its first ``_N_SIGNALS[tidx]``
# signal slots are real (the rest are zero-padded and excluded from the loss).
_target_valid = torch.zeros(_SEQ_LEN - 1, N_SIGNAL, dtype=torch.bool)
for _p in range(_SEQ_LEN - 1):
    _tidx = (_p + 1) % SALVE_TOKENS
    _target_valid[_p, :_N_SIGNALS[_tidx]] = True

# Salve-within-type attention mask for the fixed training sequence.
_GROUP_MASK = build_salve_mask(_SEQ_LEN, torch.device("cpu"))


class ExperienceBuffer:
    """Concurrent sequence recorder with randomized start offsets.

    Multiple recorders run in parallel. Each captures ``seq_len`` consecutive
    salves. A new recorder starts after a random delay (uniform in
    [seq_len//3, seq_len]) to decorrelate the learning cadence from the
    physics simulation cadence — preventing resonance between what is learned
    and the simulation's rhythm.

    Completed sequences are stored in a ring buffer. Each sequence is a list
    of salves in chronological order
    """

    def __init__(self, seq_len: int = 10, capacity: int = 400, seed: int = 0):
        self.seq_len = seq_len
        self._completed: deque = deque(maxlen=capacity)
        self._active: list[list] = []
        self._rng = random.Random(seed)
        self._min_gap = seq_len//3
        self._max_gap = seq_len
        self._countdown = 2   # start a new recorder after 2 ticks

    def __len__(self) -> int:
        return len(self._completed)

    def push(self, salve) -> None:
        # append to all active recorders
        for rec in self._active:
            rec.append(salve)
        # seal completed recorders
        still_active = []
        for rec in self._active:
            if len(rec) >= self.seq_len:
                self._completed.append(rec)
            else:
                still_active.append(rec)
        self._active = still_active
        # maybe start a new recorder
        self._countdown -= 1
        if self._countdown <= 0:
            self._active.append([salve])
            self._countdown = self._rng.randint(self._min_gap, self._max_gap)

    def sample(self, batch: int):
        """Sample ``batch`` completed sequences. Returns a list of sequences,
        each a list of salves. Returns [] if empty."""
        n = len(self._completed)
        if n == 0:
            return []
        idx = [self._rng.randrange(n) for _ in range(min(batch, n))]
        return [list(self._completed[i]) for i in idx]


class Brain:
    """Holds both networks, drives the live loop, and runs sleep training.

    Lifecycle in the main loop:

        brain.act(salve_t)            -> (theta_l, d_l, theta_r, d_r, tail_t)
        ... physics steps, sensors ...
        brain.record(salve_t, salve_next)
        brain.sleep()                 # on demand: offline training of both models
    """

    def __init__(self, lr_wm: float = 3e-4, lr_pol: float = 1e-3, act_std: float = 0.05,
                 explore_std: float = 0.2, device: str = "cpu", seed: int = 0):
        torch.manual_seed(seed)
        self.device = torch.device(device)
        d_model = 64
        self.world = M.WorldModel(d_model=d_model, nhead=4, layers=3,
                                  dim_ff=384).to(self.device)
        self.policy = M.Policy(n_state=N_POLICY_STATE + d_model).to(self.device)
        # Normalizes the world model's intermediate latent before feeding it
        # to the policy, so scalars [0,1] and latent are on the same scale.
        self.latent_norm = torch.nn.LayerNorm(d_model).to(self.device)
        self.opt_wm = torch.optim.Adam(self.world.parameters(), lr=lr_wm)
        self.opt_pol = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.latent_norm.parameters()),
            lr=lr_pol)
        self.buffer = ExperienceBuffer(seq_len=SEQ_STEPS, seed=seed)
        self.group_mask = _GROUP_MASK.to(self.device)
        self.target_valid = _target_valid.to(self.device)
        self.act_std = act_std
        self.explore_std = explore_std
        # Precomputed cost tensors, moved to device for the vectorized
        # trajectory-cost in train_policy.
        self._cost_scales = _COST_SCALES.to(self.device)
        self._cost_signed = _COST_SIGNED.to(self.device)
        self._cost_w = _COST_W.to(self.device)
        # Policy-scalar and action-encode indices/templates for the vectorized
        # train_policy path.
        self._pol_idx = torch.tensor(_POL_SCALAR_IDX, dtype=torch.long,
                                      device=self.device)
        self._act_tok = _ACT_TOK.to(self.device)
        self._act_slot = _ACT_SLOT.to(self.device)
        self._act_scales = _ACT_SCALES.to(self.device)
        self._act_signed = _ACT_SIGNED.to(self.device)
        self._act_tmpl = _ACT_TEMPLATE.to(self.device)
        # Rolling history of recent (state_tokens, action_tokens) for the
        # wake-time world model context. The policy reads the world model's
        # intermediate latent, which needs the last SEQ_STEPS-1 transitions.
        self._wake_history: list[tuple[list, list]] = []
        self._cached_latent = None   # [1, d_model] — refreshed at 1 Hz by wake_tick
        self._cached_scalars = None  # [1, 47] — scalar state at last wake_tick
        # REINFORCE baseline (running mean of imagined returns)
        self._baseline = 0.0
        self._baseline_alpha = 0.1
        # last sleep stats, for the HUD
        self.last_wm_loss = float("nan")
        self.last_pol_loss = float("nan")
        self.last_return = float("nan")
        self.mode = "wake"
        # inference timing (exponential moving average, ms)
        self._wm_time = 0.0
        self._pol_time = 0.0
        self._time_alpha = 0.1

    # --- live loop -----------------------------------------------------------
    @torch.no_grad()
    def wake_tick(self, salve) -> None:
        """World model tick: run a forward pass on the rolling history +
        current salve to produce a fresh latent for the policy. Also journals
        the transition into the experience buffer."""
        self.mode = "wake"
        cur_state = state_tokens(salve)
        ctx = self._build_context(cur_state)
        self.world.eval()
        t0 = time.perf_counter()
        self.world(ctx)
        self._cached_latent = self.latent_norm(self.world.last_latent())  # [1, d_model]
        self._wm_time = (1 - self._time_alpha) * self._wm_time \
            + self._time_alpha * (time.perf_counter() - t0) * 1000
        self._cached_scalars = self._policy_scalars_batch(
            torch.tensor(cur_state, dtype=torch.float32,
                         device=self.device).unsqueeze(0))   # [1, 47]
        # update history with this transition's state + action
        a_toks = action_tokens(salve)
        self._wake_history.append((cur_state, a_toks))
        if len(self._wake_history) > SEQ_STEPS - 1:
            self._wake_history.pop(0)

    @torch.no_grad()
    def act(self, salve, std=None) -> tuple[float, float, float, float, float]:
        """Return 5 actuator consignes. Reuses the cached latent from the
        last wake_tick(). If no latent yet (first tick), runs a fallback forward."""
        if self._cached_latent is None:
            self.wake_tick(salve)   # has to run wm at least one
        t0 = time.perf_counter()
        pol_in = torch.cat([self._cached_scalars, self._cached_latent], dim=1)
        action, _, _ = self.policy.sample(pol_in, self.act_std if std is None else std)
        self._pol_time = (1 - self._time_alpha) * self._pol_time \
            + self._time_alpha * (time.perf_counter() - t0) * 1000
        a = action[0].tolist()
        return a[0], a[1], a[2], a[3], a[4]

    def record(self, salve) -> None:
        """Journal one transition into the experience buffer."""
        self.buffer.push(salve)

    def clear_history(self) -> None:
        """Clear the wake context history, cached latent, and active recorders."""
        self._wake_history.clear()
        self._cached_latent = None
        self._cached_scalars = None
        self.buffer._active.clear()
        self._countdown = 0

    def _build_context(self, cur_state: list):
        """Build a [1, L, 25] context tensor from the wake history + current
        state. The layout is [s0, a0, s1, a1, ..., s_{k-1}, a_{k-1}, s_k]."""
        n = len(self._wake_history)
        ctx_len = n * SALVE_TOKENS + STATE_TOKENS
        ctx = torch.zeros(1, ctx_len, DENSE_DIM, device=self.device)
        pos = 0
        for s_toks, a_toks in self._wake_history:
            ctx[0, pos:pos + STATE_TOKENS] = torch.tensor(
                s_toks, dtype=torch.float32, device=self.device)
            pos += STATE_TOKENS
            ctx[0, pos:pos + ACTION_TOKENS] = torch.tensor(
                a_toks, dtype=torch.float32, device=self.device)
            pos += ACTION_TOKENS
        ctx[0, pos:pos + STATE_TOKENS] = torch.tensor(
            cur_state, dtype=torch.float32, device=self.device)
        return ctx

    # --- sleep: world model training ----------------------------------------
    def _batch_tensors(self, sequences):
        """Build a [B, L, 25] tensor for a batch of multi-step transition
        sequences.

        Each sequence is a list of SEQ_STEPS salves of 16 Sensor tokens and 3 Actuator tokens:
        [s0, a0, s1, a1, ..., a_{N-1}, sN]"""
        B = len(sequences)

        vals = torch.zeros(B, _SEQ_LEN, DENSE_DIM, device=self.device)
        for b, seq in enumerate(sequences):
            pos = 0
            for i, salve in enumerate(seq):
                assert len(salve) == SALVE_TOKENS
                vals[b, pos:pos + STATE_TOKENS] = torch.tensor(
                    salve[:STATE_TOKENS], dtype=torch.float32, device=self.device)
                pos += STATE_TOKENS
                # last action tokens are not predicted
                if i < len(seq) - 1:
                    vals[b, pos:pos + ACTION_TOKENS] = torch.tensor(
                        salve[STATE_TOKENS:SALVE_TOKENS], dtype=torch.float32, device=self.device)
                    pos += ACTION_TOKENS
        return vals

    def train_world(self, epochs: int = 4, batch: int = 32):
        """Train the world model. Generator: yields (label, value) after each
        epoch."""
        if len(self.buffer) < 1:
            return
        self.world.train()
        losses = []
        for ep in range(epochs):
            sequences = self.buffer.sample(batch)
            if not sequences:
                break
            vals = self._batch_tensors(sequences)             # [B, L, 25]
            pred = self.world(vals, attn_mask=self.group_mask)  # [B, L, 16]
            # position p predicts token p+1's signal slots
            pred_signals = pred[:, :-1, :]                   # [B, L-1, 16]
            tgt_signals = vals[:, 1:, SIG_OFFSET:]           # [B, L-1, 16]
            m = self.target_valid                              # [L-1, 16]
            m_b = m.unsqueeze(0).expand_as(pred_signals)        # [B, L-1, 16]
            sq = (pred_signals - tgt_signals) ** 2
            loss = (sq * m_b).sum() / m_b.sum().clamp(min=1)
            self.opt_wm.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world.parameters(), 1.0)
            self.opt_wm.step()
            losses.append(loss.item())
            yield "wm", loss.item()
        self.last_wm_loss = sum(losses) / len(losses) if losses else float("nan")

    def _salve_cost_batch(self, gen: torch.Tensor) -> torch.Tensor:
        """Vectorized ``salve_cost`` over a batch of predicted states.

        gen: [B, 13, 25] (or [B, 16, 25]) float tensors. Returns [B] cost —
        the same weighted innate cost as ``tokenize.salve_cost`` but computed
        in one tensor op, no per-element Python loop or ``.tolist()`` sync."""
        n = len(_COST_KEYS)
        c = gen[:, _COST_IDX, SIG_OFFSET:SIG_OFFSET + n]        # [B, 5]
        denorm = torch.where(self._cost_signed,
                             (2 * c - 1) * self._cost_scales,
                             c * self._cost_scales)             # [B, 5]
        confort = gen[:, _REWARD_IDX, SIG_OFFSET]               # [B]
        confort_phys = (2 * confort - 1) * _CONFORT_SCALE       # [B]
        return (denorm * self._cost_w).sum(1) + _CONFORT_W * confort_phys

    def _policy_scalars_batch(self, state_tokens: torch.Tensor) -> torch.Tensor:
        """Vectorized ``policy_scalars``: extract the 47 non-reward sensory
        scalars from a batch of state tokens. state_tokens: [B, 13, 25] →
        [B, 47], one gather op, no per-element loop or ``.tolist()``."""
        B = state_tokens.shape[0]
        sigs = state_tokens[:, :, SIG_OFFSET:SIG_OFFSET + N_SIGNAL]   # [B,13,16]
        flat = sigs.reshape(B, STATE_TOKENS * N_SIGNAL)                # [B,208]
        return flat.index_select(1, self._pol_idx)                    # [B,47]

    def _encode_action_batch(self, action: torch.Tensor) -> torch.Tensor:
        """Vectorized ``encode_action``: [B, 5] raw consignes → [B, 3, 25]
        dense action tokens. Normalizes (signed → (t+1)/2, unsigned → t),
        scatters into the structural-prefixed template — no ``.tolist()``."""
        B = action.shape[0]
        s = self._act_scales                              # [5]
        if action.dim() == 1:
            action = action.unsqueeze(0)
        clamped = action.clamp(
            torch.where(self._act_signed, -s, torch.zeros_like(s)),
            s)                                           # [B, 5]
        t = clamped / s                                   # [B, 5]
        norm = torch.where(self._act_signed, (t + 1) / 2, t)  # [B, 5]
        toks = self._act_tmpl.unsqueeze(0).expand(B, -1, -1).clone()  # [B,3,25]
        toks[:, self._act_tok, SIG_OFFSET + self._act_slot] = norm
        return toks                                      # [B, 3, 25]

    # --- sleep: policy training via real context + imagined rollout ----------
    def train_policy(self, steps: int = 32, batch: int = 16,
                     gamma: float = 0.9):
        """REINFORCE with one gradient step per sequence. Generator: yields
        (label, value) after each step."""
        if len(self.buffer) < 1:
            return
        self.world.eval()
        self.policy.train()
        n_ctx = 4  # real transitions used as context (s0..s3, a0..a3, s4)
        n_imagine = SEQ_STEPS - n_ctx  # 6 imagined steps (s5..s10)
        losses = []
        for _ in range(steps):
            sequences = self.buffer.sample(batch)
            if not sequences:
                break
            B = len(sequences)
            device = self.device

            # build real context: [s0, a0, s1, ..., a3, s4]
            ctx_len = n_ctx * SALVE_TOKENS + STATE_TOKENS
            ctx = torch.zeros(B, ctx_len, DENSE_DIM, device=device)
            for b, seq in enumerate(sequences):
                pos = 0
                for i in range(n_ctx):
                    salve = seq[i]
                    assert len(salve) == SALVE_TOKENS
                    s = salve[:STATE_TOKENS]
                    a = salve[STATE_TOKENS:SALVE_TOKENS]
                    ctx[b, pos:pos + STATE_TOKENS] = torch.tensor(
                        s, dtype=torch.float32, device=device)
                    pos += STATE_TOKENS
                    ctx[b, pos:pos + ACTION_TOKENS] = torch.tensor(
                        a, dtype=torch.float32, device=device)
                    pos += ACTION_TOKENS
                # s4 (the state the policy acts on)
                s4 = seq[n_ctx][:STATE_TOKENS]
                ctx[b, pos:pos + STATE_TOKENS] = torch.tensor(
                    s4, dtype=torch.float32, device=device)

            # 1. world model forward on real context → intermediate latent
            ctx_mask = build_salve_mask(ctx_len, device)
            with torch.no_grad():
                self.world(ctx, attn_mask=ctx_mask)
                latent = self.world.last_latent().detach()  # [B, d_model]
                latent = self.latent_norm(latent)

            # 2. policy infers a4 from scalars + latent (keeps gradient)
            cur_scalars = self._policy_scalars_batch(
                ctx[:, -STATE_TOKENS:])                        # [B, 47]
            pol_in = torch.cat([cur_scalars, latent], dim=1)    # [B, 47+d_model]
            action, raw, raw_mean = self.policy.sample(pol_in, self.explore_std)
            logp = self.policy.log_prob(raw, raw_mean, self.explore_std)

            # 3. append a4 to context
            act_toks = self._encode_action_batch(action)      # [B, 3, 25]
            ctx = torch.cat([ctx, act_toks], dim=1)

            # 4. WM imagines s5..s10, accumulate cost
            total_cost = torch.zeros(B, device=device)
            for k in range(n_imagine):
                with torch.no_grad():
                    gen = self.world.predict_next(ctx)         # [B, 13, 25]
                step_cost = self._salve_cost_batch(gen)        # [B]
                total_cost = total_cost + (gamma ** k) * step_cost
                # append imagined state for next prediction
                ctx = torch.cat([ctx, gen], dim=1)

            # 5. REINFORCE
            returns = -total_cost
            adv = returns - self._baseline
            self._baseline = (1 - self._baseline_alpha) * self._baseline \
                + self._baseline_alpha * float(returns.mean().item())
            loss = -(logp * adv).mean()
            self.opt_pol.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt_pol.step()
            losses.append(loss.item())
            yield "pol", loss.item()
        self.last_pol_loss = sum(losses) / len(losses) if losses else float("nan")
        self.last_return = float(returns.mean().item()) if losses else float("nan")

    # --- sleep entry point ---------------------------------------------------
    def sleep(self, wm_epochs: int = 4, wm_batch: int = 32,
              pol_steps: int = 32, pol_batch: int = 16):
        """One full sleep cycle. Generator: yields (phase, value) after each
        training step so the caller can keep the UI responsive."""
        self.mode = "sleep"
        for label, value in self.train_world(wm_epochs, wm_batch):
            yield label, value
        for label, value in self.train_policy(pol_steps, pol_batch):
            yield label, value
        self.mode = "wake"
        yield "done", {
            "wm_loss": self.last_wm_loss,
            "pol_loss": self.last_pol_loss,
            "return": self.last_return,
            "buffer": len(self.buffer),
        }

    # --- persistence ---------------------------------------------------------
    def state_dict(self):
        return {
            "world": self.world.state_dict(),
            "policy": self.policy.state_dict(),
        }

    def load_state_dict(self, sd):
        if "world" in sd:
            self.world.load_state_dict(sd["world"])
        if "policy" in sd:
            self.policy.load_state_dict(sd["policy"])

    def save_checkpoint(self, wm_path: str, pol_path: str) -> None:
        """Save world model and policy to separate files."""
        torch.save({
            "world": self.world.state_dict(),
            "opt_wm": self.opt_wm.state_dict(),
        }, wm_path)
        torch.save({
            "policy": self.policy.state_dict(),
            "latent_norm": self.latent_norm.state_dict(),
            "opt_pol": self.opt_pol.state_dict(),
        }, pol_path)

    def load_checkpoint(self, wm_path: str, pol_path: str) -> None:
        """Load world model and policy from separate files. Each is loaded
        independently — a mismatch in one does not affect the other."""
        if os.path.exists(wm_path):
            sd = torch.load(wm_path, map_location=self.device, weights_only=True)
            if "world" in sd:
                self.world.load_state_dict(sd["world"])
            if "opt_wm" in sd:
                self.opt_wm.load_state_dict(sd["opt_wm"])
        if os.path.exists(pol_path):
            sd = torch.load(pol_path, map_location=self.device, weights_only=True)
            if "policy" in sd:
                self.policy.load_state_dict(sd["policy"])
            if "latent_norm" in sd:
                self.latent_norm.load_state_dict(sd["latent_norm"])
            if "opt_pol" in sd:
                self.opt_pol.load_state_dict(sd["opt_pol"])
