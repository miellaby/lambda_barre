"""The two models of λ̄'s brain: a world model and a policy.

This is the "IA à 2 modèles" from `Lambda barre.md` § "Architecture du système
de décision apprenant":

  World model  — an autoregressive transformer over the dense multimodal token
                 stream that learns P(s(t+1) | s(t), a(t)). Trained offline
                 during sleep on logged (state, action, next-state) sequences.

  Policy      — a small network π(a|s) that produces the 5 actuator consignes
                 and is trained by reinforcement learning to minimize the cost
                 *predicted* by the world model, using imagined trajectories.

Tokens come from ``tokenize`` as dense 25-float vectors (9-float structural
prefix + 16 normalized signal slots). There are no learned embeddings: the
world model projects the 25-dim token to ``d_model`` with a linear layer, adds
a non-learned sinusoidal positional encoding keyed by *salve* index (all 16
tokens of a salve share the same position — this encodes time, not intra-salve
order), and regresses the 16 signal slots of the next token (MSE). The token
id/structure is fixed by the layout, so only the 16 signal values are free
predictions.

The policy reads the 47 non-reward sensory scalars of the state (normalized
to [0, 1]) plus the world model's intermediate latent (d_model) and outputs a
Gaussian over the 5 actuator consignes in their physical ranges, so sampled
actions are always valid consignes.
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

from .tokenize import (DENSE_DIM, N_SIGNAL, STATE_TOKENS, SALVE_TOKENS,
                       TYPE_DIM, MOD_DIM, CANAL_DIM, N_POLICY_STATE,
                       _LAYOUT, _prefix)


# --- actuator consigne ranges (mirror body.py) -------------------------------
LIMB_MIN = 10.0
LIMB_MAX = 32.0
THETA_RANGE = math.pi / 2.0   # limb / tail theta consignes in [-pi/2, +pi/2]


def build_salve_mask(L: int, device, salve_tokens: int = SALVE_TOKENS,
                     state_tokens: int = STATE_TOKENS) -> torch.Tensor:
    """Build the [L, L] float attention mask for a dense salve sequence.

    A query token may attend to a key token if the key is in an earlier salve,
    or in the same salve but of a different type (sensory vs motor) and not in
    the future, or is itself. This is the "masque par salve dans le type":
    when predicting a sensory token of salve i, the other sensory tokens of
    salve i are masked (no peeking at the state being predicted), while the
    full history remains visible. Self-attention is always allowed so no row
    is fully blocked. Returns a float mask (0.0 = allowed, -inf = blocked).
    """
    pos = torch.arange(L, device=device)
    salve_id = pos // salve_tokens                 # [L] salve index per pos
    is_sensory = (pos % salve_tokens) < state_tokens  # [L] bool
    si_q = salve_id.unsqueeze(1)                     # [L, 1] query salve
    si_k = salve_id.unsqueeze(0)                     # [1, L] key salve
    earlier = si_k < si_q                            # [L, L] key salve < query
    same = si_k == si_q
    diff_type = is_sensory.unsqueeze(0) != is_sensory.unsqueeze(1)
    k_idx = pos.unsqueeze(0)                         # [1, L]
    q_idx = pos.unsqueeze(1)                         # [L, 1]
    k_le_q = k_idx <= q_idx                          # [L, L] key not future
    same_diff_causal = same & diff_type & k_le_q
    self_eye = torch.eye(L, dtype=torch.bool, device=device)
    visible = earlier | same_diff_causal | self_eye
    mask = torch.zeros(L, L, device=device)
    mask[~visible] = float("-inf")
    return mask


# =============================================================================
# World model
# =============================================================================
class WorldModel(nn.Module):
    """Autoregressive next-signal transformer over the dense token stream.

    A training sequence is a trajectory of N transitions:
    ``[s0 (13), a0 (3), s1 (13), ..., a_{N-1} (3), sN (13)]`` (e.g. 173 tokens
    for N=10), each a 25-float vector. ``forward`` returns the predicted 16
    signal slots at every position (regression) under the salve-within-type
    mask; the target is the next token's signal section (shifted by one).

    ``predict_next`` produces the 13 next-state tokens in one parallel pass
    given a ``[state_t, action_t]`` context (16 tokens): it appends 13
    zero-signal state placeholders and reads the head outputs there.
    """

    def __init__(self, d_model: int = 64, nhead: int = 4, layers: int = 3,
                 dim_ff: int = 384, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(DENSE_DIM, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, layers)
        self.head = nn.Linear(d_model, N_SIGNAL)
        # Fixed next-state template: 13 state tokens with their structural
        # prefix and zeroed signal slots.
        tmpl = [_prefix(_LAYOUT[i]) + [0.0] * N_SIGNAL
                for i in range(STATE_TOKENS)]
        self.register_buffer("state_template",
                             torch.tensor(tmpl, dtype=torch.float32))
        # Hook on the n-1 transformer layer to capture the intermediate
        # latent representation — the policy reads this instead of raw scalars.
        self._latent = None
        self.transformer.layers[-2].register_forward_hook(self._capture_latent)

    def _capture_latent(self, module, input, output):
        self._latent = output  # [B, L, d_model]

    def last_latent(self) -> torch.Tensor:
        """Return the last token's intermediate-layer embedding from the most
        recent forward pass. This is the policy's observation: a compressed
        representation of the current state conditioned on the full history."""
        return self._latent[:, -1, :]  # [B, d_model]

    def _sinusoidal_pe(self, positions: torch.Tensor) -> torch.Tensor:
        """Non-learned sinusoidal positional encoding by salve index.

        ``positions`` is [L] long (salve indices); returns [L, d_model]. All
        tokens of the same salve share the same position — this encodes the
        salve's time step, not the intra-salve order (irrelevant since the
        structural prefix already identifies each channel)."""
        L = positions.shape[0]
        d = self.d_model
        device = positions.device
        div = torch.exp(torch.arange(0, d, 2, device=device,
                                     dtype=torch.float32)
                        * (-math.log(10000.0) / d))      # [d/2]
        pos_f = positions.float().unsqueeze(1)            # [L, 1]
        pe = torch.zeros(L, d, device=device)
        pe[:, 0::2] = torch.sin(pos_f * div)
        pe[:, 1::2] = torch.cos(pos_f * div)
        return pe

    def forward(self, x: torch.Tensor,
                attn_mask: torch.Tensor = None) -> torch.Tensor:
        """x: [B, L, 25] float tokens. Returns predicted signal slots
        [B, L, 16] (each position predicts the next token's 16 signals)."""
        B, L, _ = x.shape
        device = x.device
        h = self.in_proj(x)                              # [B, L, d_model]
        salve_pos = torch.arange(L, device=device) // SALVE_TOKENS
        h = h + self._sinusoidal_pe(salve_pos).unsqueeze(0)
        if attn_mask is None:
            attn_mask = build_salve_mask(L, device)
        else:
            attn_mask = attn_mask.to(device)
        h = self.transformer(h, mask=attn_mask)
        return self.head(h)                              # [B, L, 16]

    @torch.no_grad()
    def predict_next(self, ctx: torch.Tensor) -> torch.Tensor:
        """One-shot (parallel) prediction of the next state's 13 tokens from
        the ``[state_t, action_t]`` context.

        Appends 13 zero-signal state placeholders, runs one forward pass under
        the salve mask (each predicted sensory token attends to the context
        only, not to its siblings), and reconstructs the full 25-dim tokens by
        combining the known structural prefix with the clamped [0,1] predicted
        signals. Returns [B, 13, 25]."""
        B = ctx.shape[0]
        device = ctx.device
        ctx_len = ctx.shape[1]
        placeholder = self.state_template.to(device).unsqueeze(0).expand(
            B, -1, -1)                                    # [B, 13, 25]
        x = torch.cat([ctx, placeholder], dim=1)         # [B, ctx+13, 25]
        # Causal mask: the placeholder state tokens carry zero signals (nothing
        # to leak), and the imagined-rollout context may grow by 13-token
        # states (non-16-aligned), so the regular salve mask would mislabel
        # positions. Plain causality is robust to any context length.
        L = x.shape[1]
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
        mask = torch.zeros(L, L, device=device)
        mask[~causal] = float("-inf")
        out = self.forward(x, attn_mask=mask)            # [B, L, 16]
        pred = out[:, ctx_len:, :].clamp(0.0, 1.0)        # [B, 13, 16]
        full = self.state_template.to(device).unsqueeze(0).expand(
            B, -1, -1).clone()                            # [B, 13, 25]
        full[:, :, TYPE_DIM + MOD_DIM + CANAL_DIM:] = pred
        return full                                       # [B, 13, 25]


# =============================================================================
# Policy
# =============================================================================
class Policy(nn.Module):
    """π(a|s): a Gaussian over the 5 actuator consignes, conditioned on the
    intermediate latent of the world model (d_model dimensions) concatenated
    with the 47 non-reward sensory scalars (normalized to [0,1]).

    Output means are squashed into valid consigne ranges:
      limb theta / tail theta -> tanh * THETA_RANGE  in [-pi/2, pi/2]
      limb d                   -> LIMB_MIN + sigmoid*(LIMB_MAX-LIMB_MIN)

    Exploration uses a fixed (decaying) std supplied by the caller. For imagined
    rollouts and REINFORCE, the same Gaussian defines log_prob.
    """

    N_ACT = 5

    def __init__(self, n_state: int = N_POLICY_STATE + 96, hidden: int = 128):
        super().__init__()
        self.N_STATE = n_state
        self.net = nn.Sequential(
            nn.Linear(n_state, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, self.N_ACT)

    def forward(self, state_vec: torch.Tensor) -> torch.Tensor:
        """state_vec: [B, 143] float. Returns action consignes [B, 5]
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
