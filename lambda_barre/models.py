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


def configure_hardware(device_preference: str | None = None) -> tuple[torch.device, str]:
    """Configure PyTorch execution backend and return (torch.device, description).

    Modes:
      - 'cuda': Force CUDA GPU if available. Enables FlashSDP and MemEfficient SDP.
      - 'auto': Use CUDA if available, otherwise safe CPU.
      - 'cpu': Safe, portable CPU (default). Disables MKL-DNN to prevent SIGILL
               crashes on VMs or hosts with incomplete cpuinfo flags.
      - 'cpu-fast': Enable MKL-DNN on CPU if the host CPU is known to support it.

    Controlled via argument, or LAMBDA_DEVICE env var ('cuda', 'cpu', 'auto'),
    or LAMBDA_ACCEL=1 ('auto').
    """
    import os
    import torch
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=200)

    env_dev = os.environ.get("LAMBDA_DEVICE")
    env_accel = os.environ.get("LAMBDA_ACCEL")
    if device_preference is None:
        if env_dev:
            device_preference = env_dev.lower().strip()
        elif env_accel and env_accel not in ("0", "false", "no"):
            device_preference = "auto"
        else:
            device_preference = "cpu"

    pref = device_preference.lower().strip()

    if pref in ("cuda", "auto"):
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cudnn.benchmark = True
            dev_name = torch.cuda.get_device_name(0)
            return torch.device("cuda"), f"CUDA ({dev_name})"
        elif pref == "cuda":
            import warnings
            warnings.warn("CUDA was requested but torch.cuda.is_available() is False. Falling back to safe CPU.")

    if pref == "cpu-fast" or os.environ.get("LAMBDA_TORCH_MKLDNN"):
        torch.set_num_threads(min(os.cpu_count() or 4, 8))
        torch.backends.mkldnn.enabled = True
        return torch.device("cpu"), "CPU (MKL-DNN accelerated)"

    # Default: Safe, crash-free CPU (no MKL-DNN, math attention)
    torch.set_num_threads(4)
    torch.backends.mkldnn.enabled = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    return torch.device("cpu"), "CPU (Safe/Portable)"


# Safe initialization
_ACTIVE_DEVICE, _ACTIVE_DESC = configure_hardware()

import torch
import torch.nn as nn

from .tokenize import (_COST_IDX, _REWARD_IDX, DENSE_DIM, N_SIGNAL, STATE_TOKENS, SALVE_TOKENS,
                       ACTION_TOKENS, TYPE_DIM, MOD_DIM, CANAL_DIM, N_POLICY_STATE,
                       _LAYOUT, _prefix, _N_SIGNALS)


# --- actuator consigne ranges (mirror body.py) -------------------------------
LIMB_MIN = 10.0
LIMB_MAX = 32.0
THETA_RANGE = math.pi   # limb / tail theta consignes in [-pi, +pi]


def build_salve_mask(
    L: int,
    device
) -> torch.Tensor:
    """Classical Transformer mask
    """
    return torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=device),
            diagonal=1,
        )    

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
        # Fixed action template: 3 action tokens with their structural
        # prefix and zeroed signal slots.
        tmpl_act = [_prefix(_LAYOUT[i]) + [0.0] * N_SIGNAL
                    for i in range(STATE_TOKENS, SALVE_TOKENS)]
        self.register_buffer("action_template",
                             torch.tensor(tmpl_act, dtype=torch.float32))

        # Valid-signal masks to zero out padding slots in predicted tokens:
        # Each token has _N_SIGNALS[i] valid signal dimensions out of 16.
        # Slots >= _N_SIGNALS[i] must strictly remain 0.0 to prevent unpenalized
        # logits from leaking into in_proj(x) during multi-step imagination.
        s_mask = torch.zeros(STATE_TOKENS, N_SIGNAL, dtype=torch.float32)
        for i in range(STATE_TOKENS):
            s_mask[i, :_N_SIGNALS[i]] = 1.0
        self.register_buffer("state_valid_mask", s_mask)

        a_mask = torch.zeros(ACTION_TOKENS, N_SIGNAL, dtype=torch.float32)
        for i in range(ACTION_TOKENS):
            a_mask[i, :_N_SIGNALS[STATE_TOKENS + i]] = 1.0
        self.register_buffer("action_valid_mask", a_mask)

        # Hook on the middle transformer layer to capture the intermediate
        # latent representation (layer index 1 for 3 layers) — the policy reads this.
        self._latent = None
        mid_layer = layers // 2
        self.transformer.layers[mid_layer].register_forward_hook(self._capture_latent)

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
        # self.head produit les logits bruts, la sigmoid les transforme en [0,1]
        return self.head(h)                   # [B, L, 16]

    @torch.no_grad()
    def predict_next_state(self, ctx: torch.Tensor) -> torch.Tensor:
        """One-shot (parallel) prediction of the next state's 13 tokens from
        the ``[state_t, action_t]`` context.

        Appends 13 zero-signal state placeholders, runs one forward pass under
        the causal mask, and reconstructs the full 25-dim tokens by
        combining the known structural prefix with the clamped [0,1] predicted
        signals. Returns [B, 13, 25]."""
        B = ctx.shape[0]
        device = ctx.device
        ctx_len = ctx.shape[1]
        placeholder = self.state_template.to(device).unsqueeze(0).expand(
            B, -1, -1)                                    # [B, 13, 25]
        x = torch.cat([ctx, placeholder], dim=1)         # [B, ctx+13, 25]
        L = x.shape[1]
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
        mask = torch.zeros(L, L, device=device)
        mask[~causal] = float("-inf")
        out = self.forward(x, attn_mask=mask)            # [B, L, 16]
        pred = out[:, ctx_len - 1 : ctx_len - 1 + STATE_TOKENS, :]
        pred = pred.clamp(0.0, 1.0) * self.state_valid_mask.unsqueeze(0)
        full = self.state_template.to(device).unsqueeze(0).expand(
            B, -1, -1).clone()                            # [B, 13, 25]
        full[:, :, TYPE_DIM + MOD_DIM + CANAL_DIM:] = pred
        return full                                       # [B, 13, 25]

    @torch.no_grad()
    def predict_next(self, ctx: torch.Tensor) -> torch.Tensor:
        """Alias for ``predict_next_state`` for backwards compatibility."""
        return self.predict_next_state(ctx)

    @torch.no_grad()
    def predict_next_action(self, ctx: torch.Tensor) -> torch.Tensor:
        """Autoregressive prediction of the 3 action tokens from a context
        ending in state tokens (e.g. length = N * 16 + 13).

        For each action token i in 0..2:
          - appends placeholder i
          - runs forward pass under causal mask
          - reads prediction at position L-2 (predicting position L-1)
          - clamps to [0, 1], zeroes unused signal slots, and constructs 25-dim token
        Returns [B, 3, 25]."""
        B = ctx.shape[0]
        device = ctx.device
        cur = ctx
        pred_tokens = []
        for i in range(ACTION_TOKENS):
            ph = self.action_template[i:i + 1].to(device).unsqueeze(0).expand(
                B, -1, -1)                                # [B, 1, 25]
            x = torch.cat([cur, ph], dim=1)              # [B, cur_len+1, 25]
            L = x.shape[1]
            causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
            mask = torch.zeros(L, L, device=device)
            mask[~causal] = float("-inf")
            out = self.forward(x, attn_mask=mask)        # [B, L, 16]
            pred = out[:, -2, :].clamp(0.0, 1.0) * self.action_valid_mask[i]  # [B, 16]
            tok = ph.clone()                             # [B, 1, 25]
            tok[:, 0, TYPE_DIM + MOD_DIM + CANAL_DIM:] = pred
            pred_tokens.append(tok)
            cur = torch.cat([cur, tok], dim=1)
        return torch.cat(pred_tokens, dim=1)             # [B, 3, 25]

    @torch.no_grad()
    def predict_next_salve(self, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict both next state (13 tokens) and next action (3 tokens).
        Returns (state [B, 13, 25], action [B, 3, 25])."""
        state = self.predict_next_state(ctx)
        ctx_with_state = torch.cat([ctx, state], dim=1)
        action = self.predict_next_action(ctx_with_state)
        return state, action


# =============================================================================
# Policy
# =============================================================================
class Policy(nn.Module):
    """π(a|s): a Gaussian over the 5 actuator consignes, conditioned on the
    intermediate latent of the world model (d_model dimensions) concatenated
    with the 47 non-reward sensory scalars (normalized to [0,1]).

    Output means are squashed into valid consigne ranges:
      limb theta / tail theta -> tanh * THETA_RANGE  in [-pi, pi]
      limb d                   -> LIMB_MIN + sigmoid*(LIMB_MAX-LIMB_MIN)
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
        theta_front = torch.tanh(raw[:, 0])
        d_front = torch.sigmoid(raw[:, 1])
        theta_back = torch.tanh(raw[:, 2])
        d_back = torch.sigmoid(raw[:, 3])
        tail = torch.tanh(raw[:, 4])
        return torch.stack([theta_front, d_front, theta_back, d_back, tail], dim=1)

    def sample(self, state_vec: torch.Tensor, std: float):
        """Sample an action and return (action [B,5], pre-squash raw [B,5]).

        Sampling is done in the *pre-squash* (logit) space with a fixed isotropic
        Gaussian, then squashed. log_prob is computed against the same raw
        distribution (a standard simplification: we optimize in raw space)."""
        h = self.net(state_vec)
        raw_mean = self.mean_head(h)             # [B, 5]
        # print("raw[:,1]", raw_mean[:, 1])
        # print("raw[:,3]", raw_mean[:, 3])
        # print("sigmoid", torch.sigmoid(raw_mean[:, [1, 3]]))
        # print("action", self._squash(raw_mean)[:, [1, 3]])
        dist = torch.distributions.Normal(raw_mean, std)
        raw = dist.rsample()
        action = self._squash(raw)
        return action, raw, raw_mean
