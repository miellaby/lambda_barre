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
    if isinstance(device_preference, torch.device):
        device_preference = device_preference.type
    elif device_preference is None:
        if env_dev:
            device_preference = env_dev.lower().strip()
        elif env_accel and env_accel not in ("0", "false", "no"):
            device_preference = "auto"
        else:
            device_preference = "cpu"

    pref = str(device_preference).lower().strip()

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
import torch.nn.functional as F

from .tokenize import (_COST_IDX, _REWARD_IDX, DENSE_DIM, N_SIGNAL, STATE_TOKENS, SALVE_TOKENS,
                       ACTION_TOKENS, TYPE_DIM, MOD_DIM, CANAL_DIM, N_POLICY_STATE,
                       _LAYOUT, _prefix, _N_SIGNALS)


# --- actuator consigne ranges (mirror body.py) -------------------------------
LIMB_MIN = 10.0
LIMB_MAX = 32.0
THETA_RANGE = math.pi   # limb / tail theta consignes in [-pi, +pi]

_SALVE_MASK_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def build_salve_mask(
    L: int,
    device
) -> torch.Tensor:
    """Classical (strictly causal) Transformer mask. Cached per (length,
    device): it is queried on every training step and every rollout pass,
    and the allocation is pure overhead. Read-only — never mutate."""
    key = (L, str(device))
    m = _SALVE_MASK_CACHE.get(key)
    if m is None:
        m = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=device),
            diagonal=1,
        )
        _SALVE_MASK_CACHE[key] = m
    return m


class KVCache:
    """Incremental key/value cache for the WorldModel's attention layers.

    Stores, per layer, the K and V projections of every already-processed
    token, plus the next absolute token index and the head prediction at the
    last processed position. Because the input projections are position-wise
    and the attention is strictly causal, cached entries are immutable: the
    cached path is mathematically identical to a full recomputation, whatever
    the (continuous) token values.

    One cache belongs to one generation branch. Clone it where trajectories
    diverge (candidate actions, imagination regimes); caches are short-lived
    and hold ~1.5 KB per token per batch item (fp32, d_model 64, 3 layers).

    INVARIANT: a cache is only valid while the WorldModel weights are frozen.
    Any optimizer step on the WM invalidates every cache (K/V projections,
    norms and head all move). Trainable code paths must build a fresh cache
    after each WM update — never reuse one across gradient steps.
    """

    def __init__(self, n_layers: int, d_model: int, batch: int, device):
        self.k = [torch.zeros(batch, 0, d_model, device=device)
                  for _ in range(n_layers)]
        self.v = [torch.zeros(batch, 0, d_model, device=device)
                  for _ in range(n_layers)]
        self.pos = 0
        self.last_pred = None  # head output at the last processed position [B, 16]

    def __len__(self) -> int:
        return self.pos

    def clone(self) -> "KVCache":
        new = KVCache.__new__(KVCache)
        new.k = [t.clone() for t in self.k]
        new.v = [t.clone() for t in self.v]
        new.pos = self.pos
        new.last_pred = self.last_pred.clone() if self.last_pred is not None else None
        return new


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
        self.nhead = nhead
        self.dropout_p = dropout
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
        self._mid_layer = mid_layer
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
        # strictly causal by construction: the is_causal hint skips the
        # per-call mask comparison (and its .item() graph break under
        # torch.compile) inside nn.TransformerEncoder
        h = self.transformer(h, mask=attn_mask, is_causal=True)
        # self.head produit les logits bruts, la sigmoid les transforme en [0,1]
        return self.head(h)                   # [B, L, 16]

    # --- cached inference path ------------------------------------------------
    # The cached path replicates the TransformerEncoder computation exactly
    # (norm_first pre-norm, GELU FFN, per-head 1/sqrt(head_dim) scaling,
    # dropout 0) but computes each layer's attention with explicit K/V
    # storage, so already-processed tokens are never re-projected. It is
    # inference-only and must not run with dropout active.

    def _project_qkv(self, layer: nn.TransformerEncoderLayer,
                     x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack the MultiheadAttention packed in_proj into q, k, v."""
        w = layer.self_attn.in_proj_weight          # [3*d, d]
        b = layer.self_attn.in_proj_bias            # [3*d]
        d = self.d_model
        q = F.linear(x, w[:d], b[:d])
        k = F.linear(x, w[d:2 * d], b[d:2 * d])
        v = F.linear(x, w[2 * d:], b[2 * d:])
        return q, k, v

    @torch.inference_mode()
    def cache_forward(self, x: torch.Tensor, cache: KVCache) -> torch.Tensor:
        """Process a block of tokens [B, T, 25] through the cached attention
        path, extending ``cache`` in place. Returns the head predictions
        [B, T, 16] for the block — identical to ``forward`` on the
        concatenated stream (subject to float rounding).
        """
        if self.training and self.dropout_p > 0:
            raise RuntimeError("cached inference path is only valid without dropout")
        B, T, _ = x.shape
        d = self.d_model
        nh = self.nhead
        hd = d // nh
        device = x.device
        h = self.in_proj(x)                                     # [B, T, d]
        salve_pos = torch.arange(cache.pos, cache.pos + T, device=device) // SALVE_TOKENS
        h = h + self._sinusoidal_pe(salve_pos).unsqueeze(0)
        for i, layer in enumerate(self.transformer.layers):
            xn = layer.norm1(h)                                  # [B, T, d]
            q, k, v = self._project_qkv(layer, xn)               # [B, T, d] each
            K = torch.cat([cache.k[i], k], dim=1)                # [B, P+T, d]
            V = torch.cat([cache.v[i], v], dim=1)                # [B, P+T, d]
            cache.k[i], cache.v[i] = K, V
            qh = q.view(B, T, nh, hd).transpose(1, 2)           # [B, nh, T, hd]
            Kh = K.view(B, -1, nh, hd).transpose(1, 2)          # [B, nh, P+T, hd]
            Vh = V.view(B, -1, nh, hd).transpose(1, 2)
            P = K.shape[1] - T
            if T > 1:
                # New token t attends to all P cached tokens and to new tokens <= t.
                mask = torch.zeros(T, P + T, device=device)
                mask[:, P:] = torch.full((T, T), float("-inf"), device=device).triu(diagonal=1)
                out = F.scaled_dot_product_attention(qh, Kh, Vh, attn_mask=mask)
            else:
                out = F.scaled_dot_product_attention(qh, Kh, Vh)
            out = out.transpose(1, 2).reshape(B, T, d)           # [B, T, d]
            h = h + layer.self_attn.out_proj(out)
            h = h + layer.linear2(F.gelu(layer.linear1(layer.norm2(h))))
            if i == self._mid_layer:
                self._latent = h                                # same capture as the forward hook
        cache.pos += T
        out = self.head(h)                                       # [B, T, 16]
        cache.last_pred = out[:, -1, :]
        return out

    def _gen_tokens_cached(self, cache: KVCache, n_tokens: int,
                           template: torch.Tensor, valid_mask: torch.Tensor,
                           first_pred: torch.Tensor) -> torch.Tensor:
        """Sequentially generate ``n_tokens`` tokens continuing a populated
        cache: the head prediction at the last processed position yields
        token 0, whose feedback yields token 1, and so on. The cache is
        extended in place with every generated token."""
        B = first_pred.shape[0]
        device = first_pred.device
        preds = []

        def build(i: int, sig: torch.Tensor) -> torch.Tensor:
            sig = sig.clamp(0.0, 1.0) * valid_mask[i].to(device)
            tok = template[i].to(device).unsqueeze(0).expand(B, -1).clone()
            tok[:, TYPE_DIM + MOD_DIM + CANAL_DIM:] = sig
            return tok

        preds.append(build(0, first_pred))
        out = self.cache_forward(preds[0].unsqueeze(1), cache)   # [B, 1, 16]
        for i in range(1, n_tokens):
            preds.append(build(i, out[:, 0, :]))
            out = self.cache_forward(preds[-1].unsqueeze(1), cache)
        return torch.stack(preds, dim=1)                         # [B, n_tokens, 25]

    @torch.inference_mode()
    def predict_next_state_cached(self, cache: KVCache,
                                  extra_tokens: torch.Tensor | None = None) -> torch.Tensor:
        """Cached variant of ``predict_next_state``: continue a populated
        ``cache``. If ``extra_tokens`` [B, T, 25] is given (typically the 3
        action tokens), they are processed first. Returns [B, 13, 25]; the
        cache then ends with the generated state tokens."""
        if extra_tokens is not None:
            self.cache_forward(extra_tokens, cache)
        return self._gen_tokens_cached(cache, STATE_TOKENS, self.state_template,
                                        self.state_valid_mask, cache.last_pred)

    @torch.inference_mode()
    def predict_next_action_cached(self, cache: KVCache) -> torch.Tensor:
        """Cached variant of ``predict_next_action``: generate the 3 action
        tokens continuing a populated cache that ends with state tokens.
        Returns [B, 3, 25]; the cache then ends with the generated action
        tokens."""
        return self._gen_tokens_cached(cache, ACTION_TOKENS, self.action_template,
                                       self.action_valid_mask, cache.last_pred)

    @torch.inference_mode()
    def predict_next_state(self, ctx: torch.Tensor) -> torch.Tensor:
        """Autoregressive prediction of the next state's 13 tokens.

        Sequentially generates tokens 0..12, feeding each predicted token
        back into the context with its known structural prefix and valid signal slots.
        Returns [B, 13, 25]."""
        B = ctx.shape[0]
        device = ctx.device
        curr = ctx
        preds = []
        for i in range(STATE_TOKENS):
            out = self.forward(curr)
            sig = out[:, -1, :].clamp(0.0, 1.0) * self.state_valid_mask[i].to(device)
            tok = self.state_template[i].to(device).unsqueeze(0).expand(B, -1).clone()
            tok[:, TYPE_DIM + MOD_DIM + CANAL_DIM:] = sig
            preds.append(tok)
            curr = torch.cat([curr, tok.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1)

    @torch.inference_mode()
    def predict_next(self, ctx: torch.Tensor) -> torch.Tensor:
        """Alias for ``predict_next_state`` for backwards compatibility."""
        return self.predict_next_state(ctx)

    @torch.inference_mode()
    def predict_next_action(self, ctx: torch.Tensor) -> torch.Tensor:
        """Autoregressive prediction of the 3 action tokens from a context
        ending in state tokens.

        Sequentially generates action tokens 0..2, feeding each predicted token
        back into the context with its known structural prefix and valid signal slots.
        Returns [B, 3, 25]."""
        B = ctx.shape[0]
        device = ctx.device
        curr = ctx
        preds = []
        for i in range(ACTION_TOKENS):
            out = self.forward(curr)
            sig = out[:, -1, :].clamp(0.0, 1.0) * self.action_valid_mask[i].to(device)
            tok = self.action_template[i].to(device).unsqueeze(0).expand(B, -1).clone()
            tok[:, TYPE_DIM + MOD_DIM + CANAL_DIM:] = sig
            preds.append(tok)
            curr = torch.cat([curr, tok.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1)

    @torch.inference_mode()
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

    def __init__(self, n_state: int = N_POLICY_STATE + 96, hidden: int = 256):
        super().__init__()
        self.N_STATE = n_state
        self.net = nn.Sequential(
            nn.Linear(n_state, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
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
