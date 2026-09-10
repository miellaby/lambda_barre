"""The two models of λ̄'s brain: a world model and a policy.

This is the "IA à 2 modèles" from `Lambda barre.md` § "Architecture du système
de décision apprenant":

  World model  — an autoregressive transformer over the multimodal token stream
                 that learns P(s(t+1) | s(t), a(t)). Trained offline during sleep
                 on logged (state, action, next-state) sequences.

  Policy      — a small network π(a|s) that produces the 5 actuator consignes
                 and is trained by reinforcement learning to minimize the cost
                 *predicted* by the world model, using imagined trajectories.

Tokens come from ``tokenize`` as (id, value) pairs, both in 0-255. The world
model embeds each token as ``id_embed(id) + value_embed(value) + pos_embed(pos)``
and predicts the *value* of the next token (256 classes) under a causal mask;
the token id is structural (fixed salve layout) so only the value is a free
prediction — which is exactly the multi-modal distribution over quantized
values the design doc calls for.

The policy reads the 55 sensory scalar values of the state (normalized to
[0, 1]) and outputs a Gaussian over the 5 actuator consignes in their physical
ranges, so sampled actions are always valid consignes.
"""
from __future__ import annotations

import math
import os


def _configure_torch() -> None:
    """Pick a portable CPU torch configuration.

    On some hosts (e.g. containers with an unusable /proc/cpuinfo) the oneDNN
    (MKL-DNN) fused kernels raise SIGILL. This project is a single-machine,
    CPU-only desktop pet, so we default to the portable "math" attention
    backend with oneDNN disabled. Set ``LAMBDA_TORCH_MKLDNN=1`` to restore the
    optimized kernels on machines known to support them.
    """
    if os.environ.get("LAMBDA_TORCH_MKLDNN"):
        return
    import torch
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


_configure_torch()

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenize import STATE_IDS, STATE_LEN


# --- actuator consigne ranges (mirror body.py) -------------------------------
LIMB_MIN = 10.0
LIMB_MAX = 32.0
THETA_RANGE = math.pi / 2.0   # limb / tail theta consignes in [-pi/2, +pi/2]


# =============================================================================
# World model
# =============================================================================
class WorldModel(nn.Module):
    """Autoregressive next-value transformer over the token stream.

    A "transition" sequence is ``[state_t (61), action_t (6), state_{t+1} (61)]``
    = 128 tokens. ``forward`` returns value logits at every position predicting
    the *next* token's value (shifted by one). Training masks separator positions
    (their value is always 0 and carries no signal).

    ``roll`` autoregressively generates the 61 next-state tokens given a context
    of ``[state_t, action_t]`` (67 tokens), sampling one value at a time. Since
    the id layout of the next state is fixed and known, only values are sampled.
    """

    VOCAB = 256

    def __init__(self, d_model: int = 128, nhead: int = 4, layers: int = 3,
                 dim_ff: int = 512, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.id_embed = nn.Embedding(self.VOCAB, d_model)
        self.val_embed = nn.Embedding(self.VOCAB, d_model)
        # +1 so a 0 position is distinct from any token id/value embedding of 0
        self.pos_embed = nn.Embedding(256, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, layers)
        self.head = nn.Linear(d_model, self.VOCAB)
        self.state_ids = torch.tensor(STATE_IDS, dtype=torch.long)

    @staticmethod
    def _causal_mask(L: int, device) -> torch.Tensor:
        # True = position i may attend to position j
        return torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))

    def forward(self, ids: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
        """ids, vals: [B, L] long tensors. Returns value logits [B, L, 256]
        predicting the value of the token at position i+1 (causal)."""
        B, L = ids.shape
        device = ids.device
        pos = torch.arange(L, device=device)
        x = (self.id_embed(ids) + self.val_embed(vals)
             + self.pos_embed(pos)) * math.sqrt(self.d_model)
        mask = self._causal_mask(L, device)
        x = self.transformer(x, mask=mask)
        return self.head(x)

    @torch.no_grad()
    def roll(self, ctx_ids: torch.Tensor, ctx_vals: torch.Tensor,
             n_gen: int = STATE_LEN, temperature: float = 0.0) -> torch.Tensor:
        """Generate ``n_gen`` next-state token values given context tokens.

        ctx_ids/ctx_vals: [B, Ctx] (the [state_t, action_t] context, 67 tokens).
        Returns generated values [B, n_gen]. The generated ids are the fixed
        state layout. temperature=0 → greedy (argmax); >0 → sampling.
        """
        device = ctx_ids.device
        B = ctx_ids.shape[0]
        gen_ids = self.state_ids.to(device).unsqueeze(0).expand(B, -1)[:, :n_gen]
        ids = ctx_ids.clone()
        vals = ctx_vals.clone()
        out_vals = []
        for t in range(n_gen):
            logits = self.forward(ids, vals)  # [B, L, 256]
            last = logits[:, -1, :]            # predict next token's value
            if temperature <= 0:
                nxt = last.argmax(dim=-1)
            else:
                probs = F.softmax(last / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1).squeeze(-1)
            out_vals.append(nxt)
            nid = gen_ids[:, t].unsqueeze(1)
            nval = nxt.unsqueeze(1)
            ids = torch.cat([ids, nid], dim=1)
            vals = torch.cat([vals, nval], dim=1)
        return torch.stack(out_vals, dim=1)  # [B, n_gen]

    @torch.no_grad()
    def predict_next(self, ctx_ids: torch.Tensor, ctx_vals: torch.Tensor
                     ) -> torch.Tensor:
        """One-shot (parallel) prediction of the next state's 61 token values
        from the [state_t, action_t] context (67 tokens).

        Unlike ``roll`` (autoregressive), this predicts every next-state token
        in a single forward pass, conditioning each only on (state, action)
        plus a zeroed placeholder for the rest of the next state. It is far
        cheaper and is what the policy uses to *imagine* trajectories during
        sleep; the world model itself is still trained with the causal
        autoregressive objective. Returns predicted values [B, STATE_LEN]."""
        device = ctx_ids.device
        B = ctx_ids.shape[0]
        ns_ids = self.state_ids.to(device).unsqueeze(0).expand(B, -1)
        ns_vals = torch.zeros(B, STATE_LEN, dtype=torch.long, device=device)
        ids = torch.cat([ctx_ids, ns_ids], dim=1)        # [B, 128]
        vals = torch.cat([ctx_vals, ns_vals], dim=1)     # [B, 128]
        logits = self.forward(ids, vals)                # [B, 128, 256]
        # token at position p is predicted by logits[:, p-1]; the next-state
        # tokens occupy positions 67..127, predicted by logits[:, 66:127].
        return logits[:, 66:127, :].argmax(dim=-1)       # [B, 61]


# =============================================================================
# Policy
# =============================================================================
class Policy(nn.Module):
    """π(a|s): a Gaussian over the 5 actuator consignes, conditioned on the
    55 normalized sensory scalar values of the state.

    Output means are squashed into valid consigne ranges:
      limb theta / tail theta -> tanh * THETA_RANGE  in [-pi/2, pi/2]
      limb d                   -> LIMB_MIN + sigmoid*(LIMB_MAX-LIMB_MIN)

    Exploration uses a fixed (decaying) std supplied by the caller. For imagined
    rollouts and REINFORCE, the same Gaussian defines log_prob.
    """

    N_STATE = 55   # 55 sensory scalar values (state part, separators excluded)
    N_ACT = 5

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.N_STATE, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, self.N_ACT)

    def forward(self, state_vec: torch.Tensor) -> torch.Tensor:
        """state_vec: [B, 55] float in [0, 1]. Returns action consignes [B, 5]
        in physical ranges (the policy mean / greedy action)."""
        h = self.net(state_vec)
        raw = self.mean_head(h)
        return self._squash(raw)

    @staticmethod
    def _squash(raw: torch.Tensor) -> torch.Tensor:
        # raw: [B, 5] -> consignes [B, 5]
        theta_l = torch.tanh(raw[:, 0]) * THETA_RANGE
        d_l = LIMB_MIN + torch.sigmoid(raw[:, 1]) * (LIMB_MAX - LIMB_MIN)
        theta_r = torch.tanh(raw[:, 2]) * THETA_RANGE
        d_r = LIMB_MIN + torch.sigmoid(raw[:, 3]) * (LIMB_MAX - LIMB_MIN)
        tail = torch.tanh(raw[:, 4]) * THETA_RANGE
        return torch.stack([theta_l, d_l, theta_r, d_r, tail], dim=1)

    def sample(self, state_vec: torch.Tensor, std: float):
        """Sample an action and return (action [B,5], pre-squash raw [B,5]).

        Sampling is done in the *pre-squash* (logit) space with a fixed isotropic
        Gaussian, then squashed. log_prob is computed against the same raw
        distribution (a standard simplification: we optimize in raw space)."""
        h = self.net(state_vec)
        raw_mean = self.mean_head(h)             # [B, 5]
        dist = torch.distributions.Normal(raw_mean, std)
        raw = dist.rsample()
        action = self._squash(raw)
        return action, raw, raw_mean

    def log_prob(self, raw: torch.Tensor, raw_mean: torch.Tensor, std: float) -> torch.Tensor:
        """log p(raw | mean) under the isotropic Gaussian, summed over the 5
        dims. Returns [B]."""
        dist = torch.distributions.Normal(raw_mean, std)
        return dist.log_prob(raw).sum(dim=-1)
