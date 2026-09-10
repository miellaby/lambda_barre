"""The brain of λ̄: orchestrates the world model and the policy.

This wires the "IA à 2 modèles" (``models.py``) into the live loop. During
*éveil* (wake) the policy drives the actuator consignes and every transition
(state, action, next-state) is journaled into an experience buffer. During
*sommeil* (sleep) the brain trains offline:

  1. the world model learns to predict the next state's token values from
     logged (state, action, next-state) sequences (teacher forcing);
  2. the policy is trained by reinforcement learning on *imagined*
     trajectories: it samples actions, the (frozen) world model rolls them
     forward, and the policy is pushed (REINFORCE) toward actions whose
     imagined trajectories have low predicted cost.

That is the model-based RL loop of `Lambda barre.md`: the policy learns from
the world model's anticipation of consequences, not from direct reward.

All tensors live on CPU. The buffer stores plain Python lists of quantized
token values (ids are structural and fixed by the salve layout).
"""
from __future__ import annotations

import random
from collections import deque

import torch
import torch.nn.functional as F

from . import models as M
from .tokenize import (STATE_IDS, ACTION_IDS, STATE_LEN, ACTION_LEN,
                       state_values, salve_cost, encode_action,
                       state_token_values, action_token_values,
                       scalars_from_state_vals)


# Fixed id layout of a training sequence: [state_t, action_t, state_{t+1}].
_SEQ_IDS = STATE_IDS + ACTION_IDS + STATE_IDS          # 128 ids
_SEQ_LEN = len(_SEQ_IDS)
# positions i (0.._SEQ_LEN-2) whose predicted target (token i+1) is a scalar
# value rather than a separator — the world model is only scored on these.
_TARGET_SCALAR_MASK = torch.tensor(
    [_SEQ_IDS[i + 1] not in (1, 2, 3, 4, 5, 6, 7) for i in range(_SEQ_LEN - 1)],
    dtype=torch.bool)


class ExperienceBuffer:
    """Ring buffer of (state_vals, action_vals, next_state_vals) as lists of
    quantized token values (ints in 0-255). State parts are 61 tokens (separators
    included, value 0), action parts are 6 tokens (the ACTION group: 1 separator
    + 5 scalars)."""

    def __init__(self, capacity: int = 4000):
        self._buf: deque = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, state_vals, action_vals, next_state_vals) -> None:
        self._buf.append((list(state_vals), list(action_vals),
                         list(next_state_vals)))

    def sample(self, batch: int):
        n = len(self._buf)
        idx = [random.randrange(n) for _ in range(min(batch, n))]
        out = [self._buf[i] for i in idx]
        return out


class Brain:
    """Holds both networks, drives the live loop, and runs sleep training.

    Lifecycle in the main loop:

        brain.act(salve_t)            -> (theta_l, d_l, theta_r, d_r, tail_t)
        ... physics steps, sensors ...
        brain.record(salve_t, salve_next)
        brain.sleep()                 # on demand: offline training of both models
    """

    def __init__(self, lr_wm: float = 3e-4, lr_pol: float = 1e-3,
                 explore_std: float = 0.4, device: str = "cpu", seed: int = 0):
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.world = M.WorldModel(d_model=96, nhead=4, layers=2, dim_ff=384).to(self.device)
        self.policy = M.Policy().to(self.device)
        self.opt_wm = torch.optim.Adam(self.world.parameters(), lr=lr_wm)
        self.opt_pol = torch.optim.Adam(self.policy.parameters(), lr=lr_pol)
        self.buffer = ExperienceBuffer()
        self.seq_ids = torch.tensor([_SEQ_IDS], dtype=torch.long,
                                    device=self.device)
        self.target_mask = _TARGET_SCALAR_MASK.to(self.device)
        self.explore_std = explore_std
        # REINFORCE baseline (running mean of imagined returns)
        self._baseline = 0.0
        self._baseline_alpha = 0.1
        # last sleep stats, for the HUD
        self.last_wm_loss = float("nan")
        self.last_pol_loss = float("nan")
        self.last_return = float("nan")
        self.mode = "wake"

    # --- live loop -----------------------------------------------------------
    @torch.no_grad()
    def act(self, salve) -> tuple[float, float, float, float, float]:
        """Return 5 actuator consignes for the current state salve."""
        self.mode = "wake"
        sv = torch.tensor([state_values(salve)], dtype=torch.float32,
                          device=self.device) / 255.0
        action, _, _ = self.policy.sample(sv, self.explore_std)
        a = action[0].tolist()
        return a[0], a[1], a[2], a[3], a[4]

    def record(self, salve, next_salve) -> None:
        """Journal one transition. The action is read out of ``salve``'s ACTION
        group; the states are the state parts of both salves (61 tokens each,
        separators included)."""
        self.buffer.push(state_token_values(salve), action_token_values(salve),
                         state_token_values(next_salve))

    # --- sleep: world model training ----------------------------------------
    def _batch_tensors(self, samples):
        """Build (ids[B,128], vals[B,128], target_vals[B,127], mask[127]) for
        a batch of transitions."""
        B = len(samples)
        vals = torch.zeros(B, _SEQ_LEN, dtype=torch.long, device=self.device)
        for b, (s, a, ns) in enumerate(samples):
            vals[b, :STATE_LEN] = torch.tensor(s, dtype=torch.long)
            vals[b, STATE_LEN:STATE_LEN + ACTION_LEN] = torch.tensor(a, dtype=torch.long)
            vals[b, STATE_LEN + ACTION_LEN:] = torch.tensor(ns, dtype=torch.long)
        ids = self.seq_ids.expand(B, -1)
        target_vals = vals[:, 1:]            # what each position must predict
        return ids, vals, target_vals

    def train_world(self, epochs: int = 4, batch: int = 32) -> float:
        """Train the world model on the buffer. Returns mean loss."""
        if len(self.buffer) < 8:
            return float("nan")
        self.world.train()
        losses = []
        for _ in range(epochs):
            samples = self.buffer.sample(batch)
            if not samples:
                break
            ids, vals, target_vals = self._batch_tensors(samples)
            logits = self.world(ids, vals)          # [B, L, 256]
            pred = logits[:, :-1, :]                 # predict positions 1..L-1
            tgt = target_vals.clamp(0, 255)
            # CE only on scalar-target positions
            m = self.target_mask
            loss = F.cross_entropy(pred[:, m].transpose(1, 2), tgt[:, m])
            self.opt_wm.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world.parameters(), 1.0)
            self.opt_wm.step()
            losses.append(loss.item())
        return sum(losses) / len(losses) if losses else float("nan")

    # --- sleep: policy training via imagined trajectories --------------------
    def train_policy(self, steps: int = 32, horizon: int = 4, batch: int = 16,
                     gamma: float = 0.9) -> float:
        """REINFORCE on imagined trajectories. Returns mean policy loss.

        For each sampled starting state, the policy samples an action; the
        world model imagines the next state and its cost; we repeat for
        ``horizon`` steps, discounting cost. The policy is pushed toward
        actions whose imagined return (negative cumulative cost) is high."""
        if len(self.buffer) < 8:
            return float("nan")
        self.world.eval()
        self.policy.train()
        losses = []
        for _ in range(steps):
            samples = self.buffer.sample(batch)
            if not samples:
                break
            starts = [s for (s, a, ns) in samples]   # 61-token state-val lists
            state_vals = torch.tensor(
                [scalars_from_state_vals(s) for s in starts],
                dtype=torch.float32, device=self.device) / 255.0
            # sample first action from the policy (keeps grad)
            action, raw, raw_mean = self.policy.sample(state_vals, self.explore_std)
            logp = self.policy.log_prob(raw, raw_mean, self.explore_std)
            # roll imagined trajectory
            with torch.no_grad():
                returns = self._roll_return(action, starts, horizon, gamma)
            # REINFORCE: maximize E[logp * (return - baseline)]
            adv = returns - self._baseline
            self._baseline = (1 - self._baseline_alpha) * self._baseline \
                + self._baseline_alpha * float(returns.mean().item())
            loss = -(logp * adv).mean()
            self.opt_pol.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt_pol.step()
            losses.append(loss.item())
        self.last_return = float(returns.mean().item()) if losses else float("nan")
        return sum(losses) / len(losses) if losses else float("nan")

    @torch.no_grad()
    def _roll_return(self, first_action, start_states, horizon, gamma):
        """Imagine ``horizon`` steps from each start state, beginning with the
        policy's sampled first action, then greedy policy actions. Returns the
        discounted cumulative cost (a cost → low is good) as a [B] tensor."""
        B = first_action.shape[0]
        device = first_action.device
        total = torch.zeros(B, device=device)
        action = first_action
        # current state as token-value lists
        cur = [list(start_states[b]) for b in range(B)]
        for k in range(horizon):
            # tokenize the action (consignes) into ACTION group values
            act_vals = []
            for b in range(B):
                a = action[b].tolist()
                toks = encode_action(a[0], a[1], a[2], a[3], a[4])
                act_vals.append([v for (_id, v) in toks])  # 6 vals (sep + 5)
            # build context tensors [B, 67]
            ids = torch.zeros(B, STATE_LEN + ACTION_LEN, dtype=torch.long,
                              device=device)
            vals = torch.zeros(B, STATE_LEN + ACTION_LEN, dtype=torch.long,
                               device=device)
            for b in range(B):
                ids[b, :STATE_LEN] = torch.tensor(STATE_IDS, device=device)
                ids[b, STATE_LEN:] = torch.tensor(ACTION_IDS, device=device)
                vals[b, :STATE_LEN] = torch.tensor(cur[b], dtype=torch.long, device=device)
                vals[b, STATE_LEN:] = torch.tensor(act_vals[b], dtype=torch.long, device=device)
            gen = self.world.predict_next(ids, vals)
            # cost + next state
            step_cost = torch.zeros(B, device=device)
            next_states = []
            for b in range(B):
                ns = gen[b].tolist()
                stoks = list(zip(STATE_IDS, ns))
                step_cost[b] = salve_cost(stoks)
                next_states.append(ns)
            total += (gamma ** k) * step_cost
            # next action: greedy policy on the new state (55 scalar view)
            sv = torch.tensor([scalars_from_state_vals(ns) for ns in next_states],
                              dtype=torch.float32, device=self.device) / 255.0
            action = self.policy(sv)
            cur = next_states
        # return = -cumulative cost (we maximize return ⇔ minimize cost)
        return -total

    # --- sleep entry point ---------------------------------------------------
    def sleep(self, wm_epochs: int = 4, wm_batch: int = 32,
              pol_steps: int = 32, pol_batch: int = 16, horizon: int = 4):
        """One full sleep cycle: train the world model, then the policy.

        Returns a dict of stats for the HUD."""
        self.mode = "sleep"
        self.last_wm_loss = self.train_world(wm_epochs, wm_batch)
        self.last_pol_loss = self.train_policy(pol_steps, horizon, pol_batch)
        self.mode = "wake"
        return {
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
