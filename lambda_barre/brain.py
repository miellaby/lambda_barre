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

import bisect
import itertools
import math
import os
import random
import time
from collections import deque

import torch

from . import models as M
from .dream import DreamStep, DreamTrajectory, DreamRecord, REGIME_NAMES
from .models import build_salve_mask, KVCache
from .tokenize import (STATE_TOKENS, ACTION_TOKENS, SALVE_TOKENS, DENSE_DIM,
                       N_SIGNAL, _N_SIGNALS, N_POLICY_STATE, SIG_OFFSET,
                       _M_INTERO, _EOS_CHANNEL_W,
                       _COST_IDX, _REWARD_IDX, _INTERO_IDX, _COST_KEYS, _COST_WEIGHTS,
                       _BY_KEY, _LAYOUT, _prefix,
                       state_tokens, action_tokens, salve_cost)


# Default sequence layout parameters (transitions per sequence, and past context transitions)
DEFAULT_SEQ_STEPS = 10
DEFAULT_N_CTX = 4

SEQ_STEPS = DEFAULT_SEQ_STEPS
N_CTX = DEFAULT_N_CTX
_N_CTX = N_CTX  # backward-compatible alias


def compute_seq_layout(seq_steps: int = DEFAULT_SEQ_STEPS, n_ctx: int = DEFAULT_N_CTX) -> dict:
    """Compute geometric positions and token dimensions derived from seq_steps and n_ctx."""
    assert n_ctx < seq_steps, f"n_ctx ({n_ctx}) must be strictly less than seq_steps ({seq_steps})"
    seq_len = seq_steps * SALVE_TOKENS + STATE_TOKENS
    decision_pos = n_ctx * SALVE_TOKENS + _INTERO_IDX
    eos_pos = seq_steps * SALVE_TOKENS + _INTERO_IDX
    n_imagine = seq_steps - n_ctx
    return {
        "seq_steps": seq_steps,
        "n_ctx": n_ctx,
        "seq_len": seq_len,
        "decision_pos": decision_pos,
        "eos_pos": eos_pos,
        "n_imagine": n_imagine,
    }


def compute_eos_time_weights(n_imagine: int) -> list[float]:
    """Compute uniform (flat) normalized time weights over n_imagine consequence steps."""
    if n_imagine <= 0:
        return []
    return [1.0 / n_imagine] * n_imagine


def compute_context_time_weights(n_early: int, decay: float = 0.5) -> list[float]:
    """Compute normalized time weights over the pre-decision context window (s0..s_{n_ctx}),
    decaying backwards in time away from the decision state s_{n_ctx}.
    The decision state reflects the immediate crisis driving the action, so it carries
    the majority of the baseline weight (~52% for n=5 with decay=0.5)."""
    if n_early <= 0:
        return []
    w = [decay ** (n_early - 1 - i) for i in range(n_early)]
    total = sum(w)
    return [x / total for x in w]


def build_target_masks(seq_steps: int = DEFAULT_SEQ_STEPS) -> tuple[torch.Tensor, torch.Tensor]:
    """Build _target_valid and _target_weight tensors for a given seq_steps."""
    seq_len = seq_steps * SALVE_TOKENS + STATE_TOKENS
    target_valid = torch.zeros(seq_len - 1, N_SIGNAL, dtype=torch.bool)
    for p in range(seq_len - 1):
        tidx = (p + 1) % SALVE_TOKENS
        if tidx == _INTERO_IDX and (p + 1) < (seq_len - 1):
            continue
        n = _EOS_NSLOTS if (p + 1) == (seq_len - 1) else _N_SIGNALS[tidx]
        target_valid[p, :n] = True

    target_weight = torch.zeros(seq_len - 1, N_SIGNAL, dtype=torch.float32)
    for p in range(seq_len - 1):
        tidx = (p + 1) % SALVE_TOKENS
        if tidx == _INTERO_IDX and (p + 1) < (seq_len - 1):
            continue
        n = _EOS_NSLOTS if (p + 1) == (seq_len - 1) else _N_SIGNALS[tidx]
        target_weight[p, :n] = 1.0
    for tidx in (_COST_IDX, _REWARD_IDX, _INTERO_IDX):
        p = seq_steps * SALVE_TOKENS + tidx - 1
        n = _EOS_NSLOTS if tidx == _INTERO_IDX else _N_SIGNALS[tidx]
        target_weight[p, :n] = _WM_REWARD_BOOST

    return target_valid, target_weight


def build_lookahead_mask(seq_steps: int = DEFAULT_SEQ_STEPS, n_ctx: int = DEFAULT_N_CTX,
                         device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Attention mask for training WM with lookahead on a sequence."""
    layout = compute_seq_layout(seq_steps, n_ctx)
    mask = build_salve_mask(layout["seq_len"], device).clone()
    dp, ep = layout["decision_pos"], layout["eos_pos"]
    mask[dp, ep] = False
    mask[ep, dp + 1: ep] = True
    return mask


def build_lookahead_gen_mask(ctx_len: int, device, decision_pos: int | None = None) -> torch.Tensor:
    """Attention mask for hindsight generation on a [context, EOS] input of
    ctx_len + 1 tokens. If decision_pos is None, defaults to _DECISION_POS."""
    if decision_pos is None:
        decision_pos = _DECISION_POS
    m = build_salve_mask(ctx_len + 1, device).clone()
    m[decision_pos, ctx_len] = False
    return m


_LAYOUT_DEFAULT = compute_seq_layout(DEFAULT_SEQ_STEPS, DEFAULT_N_CTX)
_SEQ_LEN = _LAYOUT_DEFAULT["seq_len"]
_DECISION_POS = _LAYOUT_DEFAULT["decision_pos"]
_EOS_POS = _LAYOUT_DEFAULT["eos_pos"]
_EOS_IMAGINE = _LAYOUT_DEFAULT["n_imagine"]
_EXPLORE_MARGIN_PCT = 0.001                                # 0.1% cost reduction margin required for alternative candidates
_WM_SALIENCY_MAX_LOSS = 0.02                               # Coreset WM loss below which EOS-saliency distances are trusted
_WM_IMPACT_FLOOR = 0.05                                     # floor weight added to every sequence's cost impact
_WM_REWARD_BOOST = 16.0                                      # gradient boost on S10 reward slots and terminal EOS delta
_WM_LOOKAHEAD_PERIOD = 4                                     # WM mask alternation: 1 lookahead pass every 4 (3:1 causal:lookahead)
_EOS_FORCE_EPS = (0.02, 0.10)                                # relative hindsight forcing: eps ~ U(lo, hi) per slot
_EOS_CANAL = (0.0, 0.0, 1.0, 0.0)
_EOS_NSLOTS = 6
_EOS_DELTA_SCALE = 0.15
_EOS_TIME_W = compute_eos_time_weights(_EOS_IMAGINE)
_EOS_TIME_W_T = torch.tensor(_EOS_TIME_W, dtype=torch.float32)
_CONTEXT_TIME_W = compute_context_time_weights(DEFAULT_N_CTX + 1)
_CONTEXT_TIME_W_T = torch.tensor(_CONTEXT_TIME_W, dtype=torch.float32)
_EOS_CHANNEL_W_T = (torch.tensor(_EOS_CHANNEL_W, dtype=torch.float32)
                    / sum(_EOS_CHANNEL_W))                   # sums to 1


def _eos_delta_slot(d: float) -> float:
    """Raw cost-slot delta -> EOS slot: 0.5 = no change, below = recovery,
    above = worsening; full scale = +/- _EOS_DELTA_SCALE."""
    t = max(-1.0, min(1.0, d / _EOS_DELTA_SCALE))
    return (t + 1.0) / 2.0


# Precomputed cost metadata (from the dense layout) for the vectorized
# trajectory-cost: 5 unsigned cost signals (token 10) + 1 signed confort
# signal (token 11). Denormalize: unsigned -> t*scale, signed -> (2t-1)*scale.
_COST_SCALES = torch.tensor([_BY_KEY[k].max_val for k in _COST_KEYS],
                            dtype=torch.float32)              # [5]
_COST_SIGNED = torch.tensor([_BY_KEY[k].signed for k in _COST_KEYS],
                            dtype=torch.bool)                  # [5]
_COST_W = torch.tensor([_COST_WEIGHTS[k] for k in _COST_KEYS],
                      dtype=torch.float32)                    # [5]
_CONFORT_SCALE = _BY_KEY["confort"].max_val
_CONFORT_W = _COST_WEIGHTS["confort"]

# Flat signal-grid indices for the 45 policy scalars: tokens 0..9, first
# _N_SIGNALS[i] slots each. Used to gather them in one op from a [B, 13, 16]
# signal grid.
_POL_SCALAR_IDX = []
for _i in range(_COST_IDX):               # tokens 0..9
    for _k in range(_N_SIGNALS[_i]):
        _POL_SCALAR_IDX.append(_i * N_SIGNAL + _k)
assert len(_POL_SCALAR_IDX) == N_POLICY_STATE  # 45
# print(f"policy scalars: {_POL_SCALAR_IDX} (len={len(_POL_SCALAR_IDX)})")

# Action encoding metadata: 5 consignes → (token-within-action-block, slot,
# scale, signed). The action block is tokens 13..15 (3 tokens).
_ACT_KEYS = ("membre_avant_theta", "membre_avant_d", "membre_arriere_theta", "membre_arriere_d",
             "queue_theta")
_ACT_LAYOUT = []   # (tok_in_block, slot)
for _i in range(STATE_TOKENS, SALVE_TOKENS):    # 13, 14, 15
    for _k, _key in enumerate(_LAYOUT[_i].signals):
        _ACT_LAYOUT.append((_i - STATE_TOKENS, _k, _key))
_ACT_TOK = torch.tensor([t for t, _, _ in _ACT_LAYOUT], dtype=torch.long)
_ACT_SLOT = torch.tensor([s for _, s, _ in _ACT_LAYOUT], dtype=torch.long)
_ACT_SCALES = torch.tensor([_BY_KEY[k].max_val for _, _, k in _ACT_LAYOUT],
                           dtype=torch.float32)
_ACT_SIGNED = torch.tensor([_BY_KEY[k].signed for _, _, k in _ACT_LAYOUT],
                           dtype=torch.bool)
# Template: 3 action tokens with structural prefix, zeroed signals.
_ACT_TEMPLATE = torch.tensor(
    [_prefix(_LAYOUT[i]) + [0.0] * N_SIGNAL
     for i in range(STATE_TOKENS, SALVE_TOKENS)], dtype=torch.float32)

_target_valid, _target_weight = build_target_masks(DEFAULT_SEQ_STEPS)
_MASK = build_salve_mask(_SEQ_LEN, torch.device("cpu"))
_MASK_LOOKAHEAD = build_lookahead_mask(DEFAULT_SEQ_STEPS, DEFAULT_N_CTX, torch.device("cpu"))


# Structural template of the terminal EOS token: intero modality, dedicated
# canal, state type, zeroed signal slots (filled with the 6 cost-aggregate
# slots at batch time / generation time).
_EOS_TEMPLATE = torch.tensor(
    [0.0] + list(_M_INTERO) + list(_EOS_CANAL) + [0.0] * N_SIGNAL,
    dtype=torch.float32)

# Channels tracked for static detection:
#   - animal: proprioception (tokens 0..3) and touch (tokens 8..9)
#   - environnement: vision (tokens 4, 6), ball/sound (tokens 5, 7), touch (tokens 8..9)
#   - consignes: action (tokens 13..15)
# Excludes reward/cost accumulators (tokens 10, 11) and interoception (token 12).
_STATIC_SOURCES = {"proprio", "touch", "ball", "vision", "action"}
_STATIC_TOKENS = tuple(ch.idx for ch in _LAYOUT if ch.source in _STATIC_SOURCES)


def _salve_moved(s1: list[list[float]], s2: list[list[float]],
                 checked_tokens: tuple[int, ...] = _STATIC_TOKENS,
                 tol: float = 1e-4) -> bool:
    """Check whether any signal in animal, environment, or consigne tokens moved by >= tol."""
    valid_tokens = [t for t in checked_tokens if t < len(s1) and t < len(s2)]
    if not valid_tokens:
        valid_tokens = list(range(min(len(s1), len(s2))))
    for t in valid_tokens:
        tok1 = s1[t]
        tok2 = s2[t]
        for k in range(min(len(tok1), len(tok2))):
            if abs(tok1[k] - tok2[k]) >= tol:
                return True
    return False


def sequence_relief(seq: list, n_ctx: int = DEFAULT_N_CTX,
                    eos_time_w: list[float] | None = None,
                    context_time_w: list[float] | None = None) -> float:
    """Signed biological cost relief of one sequence: pre-decision window (s0..s_{n_ctx})
    minus time-weighted consequence window (s_{n_ctx+1}..s_end).
    The pre-decision window is weighted towards s_{n_ctx} (the decision state).
    Positive = long-term denouement is cheaper (recovery).
    Negative = long-term denouement is worse (aggravation / fall).
    """
    if len(seq) <= n_ctx + 1:
        return 0.0
    n_early = n_ctx + 1
    if eos_time_w is None:
        eos_time_w = compute_eos_time_weights(len(seq) - 1 - n_ctx)
    if context_time_w is None:
        context_time_w = compute_context_time_weights(n_early)
    try:
        early = [0.0] * _EOS_NSLOTS
        for i, salve in enumerate(seq[:n_early]):
            w = context_time_w[i] if i < len(context_time_w) else (1.0 / n_early)
            for j in range(5):
                early[j] += w * salve[_COST_IDX][SIG_OFFSET + j]
            early[5] += w * salve[_REWARD_IDX][SIG_OFFSET]
        late = [0.0] * _EOS_NSLOTS
        for i, salve in enumerate(seq[n_early:]):
            w = eos_time_w[i] if i < len(eos_time_w) else 0.0
            for j in range(5):
                late[j] += w * salve[_COST_IDX][SIG_OFFSET + j]
            late[5] += w * salve[_REWARD_IDX][SIG_OFFSET]
        return sum(w * (e - l) for w, e, l in zip(_EOS_CHANNEL_W, early, late)) / sum(_EOS_CHANNEL_W)
    except (IndexError, TypeError):
        return 0.0


def sequence_impact(sequences: list[list[list[float]]], n_ctx: int = DEFAULT_N_CTX,
                    eos_time_w: list[float] | None = None,
                    context_time_w: list[float] | None = None) -> torch.Tensor:
    """Per-sequence biological cost impact [B] for World Model training weighting:
    |relief| = absolute magnitude of the consequence vs decision difference.
    Symmetric: a sequence where the creature falls or recovers has high impact;
    a steady-state sequence has near-zero impact.
    """
    if not sequences:
        return torch.empty(0)
    impacts = [abs(sequence_relief(seq, n_ctx=n_ctx, eos_time_w=eos_time_w, context_time_w=context_time_w)) for seq in sequences]
    return torch.tensor(impacts, dtype=torch.float32)


# --- naive sequence distance & clustering across all channels -----------------
def _sequence_signals_tensor(sequences: list[list[list[float]]],
                             device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Flatten all signal slots across all tokens in each sequence into a [N, D] float tensor.
    D = len(seq) * SALVE_TOKENS * N_SIGNAL (e.g. 11 * 16 * 16 = 2816).
    Naive distance takes all channels into account."""
    if not sequences:
        return torch.empty(0, 0, device=device)
    flat_list = []
    for seq in sequences:
        seq_sigs = []
        for salve in seq:
            for tok in salve:
                seq_sigs.extend(tok[SIG_OFFSET:])
        flat_list.append(seq_sigs)
    return torch.tensor(flat_list, dtype=torch.float32, device=device)


def pairwise_sequence_distances(sequences: list[list[list[float]]],
                                device: torch.device = torch.device("cpu"),
                                weights: torch.Tensor | None = None) -> torch.Tensor:
    """Pairwise RMSE distance matrix [N, N] across all channels and tokens, optionally weighted by saliency."""
    X = _sequence_signals_tensor(sequences, device)
    N, D = X.shape
    if N <= 1 or D == 0:
        return torch.zeros(N, N, device=device)
    if weights is not None:
        w = weights.to(device=device, dtype=X.dtype).view(1, D)
        X = X * torch.sqrt(w)
    return torch.cdist(X, X) / (D ** 0.5)


def cross_sequence_distances(seqs_a: list[list[list[float]]],
                             seqs_b: list[list[list[float]]],
                             device: torch.device = torch.device("cpu"),
                             weights: torch.Tensor | None = None) -> torch.Tensor:
    """Cross RMSE distance matrix [N_a, N_b] across all channels and tokens, optionally weighted by saliency."""
    Xa = _sequence_signals_tensor(seqs_a, device)
    Xb = _sequence_signals_tensor(seqs_b, device)
    if Xa.shape[0] == 0 or Xb.shape[0] == 0 or Xa.shape[1] == 0:
        return torch.empty(Xa.shape[0], Xb.shape[0], device=device)
    D = Xa.shape[1]
    if weights is not None:
        w = weights.to(device=device, dtype=Xa.dtype).view(1, D)
        w_sqrt = torch.sqrt(w)
        Xa = Xa * w_sqrt
        Xb = Xb * w_sqrt
    return torch.cdist(Xa, Xb) / (D ** 0.5)


def complete_linkage_clustering(dist_matrix: torch.Tensor, eps: float) -> list[list[int]]:
    """Complete linkage agglomerative clustering with threshold eps.
    Guarantees that all members within a cluster have pairwise distance <= eps."""
    n = dist_matrix.shape[0]
    clusters = [[i] for i in range(n)]
    if n <= 1:
        return clusters

    active = list(range(n))
    cdist = dist_matrix.clone()

    while len(active) > 1:
        sub_dist = cdist[active][:, active]
        diag_mask = torch.eye(len(active), dtype=torch.bool, device=dist_matrix.device)
        sub_dist[diag_mask] = float("inf")
        min_val, flat_idx = torch.min(sub_dist.view(-1), dim=0)
        if min_val.item() > eps:
            break

        i_idx = flat_idx.item() // len(active)
        j_idx = flat_idx.item() % len(active)
        ci = active[i_idx]
        cj = active[j_idx]

        clusters[ci].extend(clusters[cj])
        clusters[cj] = []

        cdist[ci] = torch.maximum(cdist[ci], cdist[cj])
        cdist[:, ci] = cdist[ci]

        active.remove(cj)

    return [c for c in clusters if c]


def select_medoids(clusters: list[list[int]], dist_matrix: torch.Tensor) -> list[int]:
    """Select the medoid (most central representative) for each cluster."""
    medoids = []
    for c in clusters:
        if len(c) == 1:
            medoids.append(c[0])
        else:
            sub_d = dist_matrix[c][:, c]
            sum_d = sub_d.sum(dim=1)
            best_idx = torch.argmin(sum_d).item()
            medoids.append(c[best_idx])
    return medoids


class ExperienceBuffer:
    """Two-tier experience memory: Coreset (long-term consolidated) + Addendum (wake session journal).

    Conforming to the Lambda barre specification:
      - Wake session: transitions are journaled into linear segments.
      - At sleep time:
          1. extract_addendum() extracts all sliding-window sequences of length seq_len
             from the wake session into _addendum.
          2. prune_coreset(pct) randomly moves ~33% of _coreset sequences into _addendum
             as active forgetting candidates.
          3. filter_addendum() retains only surprising sequences evaluated by the World Model.
          4. commit_addendum() merges the surviving sequences into _coreset.
    """

    def __init__(self, seq_len: int = DEFAULT_SEQ_STEPS + 1, capacity: int = 1000,
                 pool_capacity: int = 1000, seed: int = 0,
                 n_ctx: int | None = None):
        self.seq_len = seq_len
        self.capacity = capacity
        self.pool_capacity = pool_capacity
        self.seq_steps = seq_len - 1
        if n_ctx is None:
            self.n_ctx = min(DEFAULT_N_CTX, max(1, self.seq_steps // 2))
        else:
            self.n_ctx = n_ctx
        self.n_imagine = max(1, self.seq_steps - self.n_ctx)
        self._eos_time_w = compute_eos_time_weights(self.n_imagine)
        self._context_time_w = compute_context_time_weights(self.n_ctx + 1)
        self._segments: deque[list[list[float]]] = deque()
        self._current: list[list[float]] = []
        self._coreset: list[list[list[float]]] = []
        self._coreset_vivacity: list[float] = []
        self._addendum: list[list[list[float]]] = []
        self._addendum_meta: list[str] = []
        self._rng = random.Random(seed)

    @property
    def _persistent_pool(self) -> list[list[list[float]]]:
        """Backward compatibility alias for _coreset."""
        return self._coreset

    @_persistent_pool.setter
    def _persistent_pool(self, val) -> None:
        self._coreset = list(val)[:self.pool_capacity]
        self._coreset_vivacity = [1.0] * len(self._coreset)

    @property
    def _valid_segments(self) -> list[list[list[float]]]:
        """Return all segments (archived or active) that have at least ``seq_len`` salves."""
        segs = list(self._segments)
        if len(self._current) >= self.seq_len:
            segs.append(self._current)
        return segs

    @property
    def num_segments(self) -> int:
        """Number of active + archived segments in the current session journal with at least 1 salve."""
        n = len(self._segments)
        if self._current:
            n += 1
        return n

    @property
    def pool_size(self) -> int:
        """Backward-compatible alias for coreset_size."""
        return len(self._coreset)

    @property
    def coreset_size(self) -> int:
        """Number of sequences stored in the long-term consolidated Coreset."""
        return len(self._coreset)

    @property
    def addendum_size(self) -> int:
        """Number of sequences currently in the Addendum (including extractable wake journal)."""
        return len(self._addendum) + self.journal_len

    @property
    def journal_len(self) -> int:
        """Number of extractable sliding-window sequences in the session journal."""
        total = sum(len(s) - self.seq_len + 1 for s in self._segments)
        if len(self._current) >= self.seq_len:
            total += len(self._current) - self.seq_len + 1
        return total

    def __len__(self) -> int:
        """Total number of sequences available (coreset + addendum + session journal)."""
        return len(self._coreset) + len(self._addendum) + self.journal_len

    def push(self, salve: list[list[float]]) -> None:
        """Append one transition salve to the current active segment."""
        self._current.append(salve)
        self._enforce_capacity()

    def boundary(self) -> None:
        """Seal the active segment and start a new one.

        If the active segment contains at least ``seq_len`` salves, it is archived
        into the completed segments deque. Otherwise it is discarded, as it cannot
        form a valid training sequence.
        """
        if len(self._current) >= self.seq_len:
            self._segments.append(self._current)
        self._current = []

    def new_segment(self) -> None:
        """Alias for boundary()."""
        self.boundary()

    def extract_addendum(self, stride: int = 1) -> int:
        """Extract sliding-window sequences from all valid segments in the session
        journal into the Addendum. Clears the session segments so they are not re-processed.

        Returns the number of new sequences added.
        """
        valid_segs = self._valid_segments
        added = 0
        for seg in valid_segs:
            n_windows = len(seg) - self.seq_len + 1
            if n_windows <= 0:
                continue
            for offset in range(0, n_windows, stride):
                self._addendum.append(seg[offset : offset + self.seq_len])
                self._addendum_meta.append("wake")
                added += 1
            if (n_windows - 1) % stride != 0:
                self._addendum.append(seg[n_windows - 1 : n_windows - 1 + self.seq_len])
                self._addendum_meta.append("wake")
                added += 1
        self._segments.clear()
        self._current.clear()
        return added

    def prune_coreset(self, pct: float = 0.33, beta: float = 2.0) -> int:
        """Move pct (default 33%) of sequences from Coreset into Addendum.
        These become candidates for active forgetting.

        Uses stochastic sampling without replacement (Efraimidis-Spirakis)
        biased towards memories with lower vivacity:
          weight w_i = exp(-beta * vivacity_i)
        Lower vivacity yields higher probability, while still allowing any memory
        to be stochastically revisited.

        Returns the number of sequences moved.
        """
        if len(self._coreset) <= 1 or pct <= 0.0:
            return 0
        n_prune = max(1, int(len(self._coreset) * pct))
        n_prune = min(n_prune, len(self._coreset) - 1)
        while len(self._coreset_vivacity) < len(self._coreset):
            self._coreset_vivacity.append(1.0)
        # Weighted stochastic keys: key_i = -ln(u_i) / w_i with u_i ~ U(0, 1).
        # Smallest keys correspond to selected items without replacement.
        keys = [
            -math.log(max(1e-12, self._rng.random())) / math.exp(-beta * v)
            for v in self._coreset_vivacity
        ]
        order = sorted(range(len(self._coreset)), key=lambda i: keys[i])
        prune_indices = set(order[:n_prune])
        new_coreset = []
        new_viv = []
        for i, seq in enumerate(self._coreset):
            if i in prune_indices:
                self._addendum.append(seq)
                self._addendum_meta.append("coreset")
            else:
                new_coreset.append(seq)
                new_viv.append(self._coreset_vivacity[i])
        self._coreset = new_coreset
        self._coreset_vivacity = new_viv
        return n_prune

    def deduplicate_addendum(self, eps: float = 0.04,
                             weights: torch.Tensor | None = None) -> int:
        """Intra-Addendum deduplication via complete linkage clustering on RMSE distances.
        Groups sequences at distance <= eps across all channels and tokens, and keeps only the medoid of each group.
        Optionally uses saliency feature weights.
        Returns the number of duplicate sequences removed."""
        if len(self._addendum) <= 1:
            return 0
        D = pairwise_sequence_distances(self._addendum, weights=weights)
        clusters = complete_linkage_clustering(D, eps=eps)
        medoid_indices = select_medoids(clusters, D)
        n_dropped = len(self._addendum) - len(medoid_indices)
        if n_dropped > 0:
            self._addendum = [self._addendum[i] for i in medoid_indices]
            if len(self._addendum_meta) >= len(medoid_indices):
                self._addendum_meta = [self._addendum_meta[i] for i in medoid_indices]
        return n_dropped

    def deduplicate_against_coreset(self, eps: float = 0.04,
                                    weights: torch.Tensor | None = None) -> dict:
        """Cross-deduplicate Addendum against Coreset and refresh vivacity of existing memories.

        Sequences in Addendum that are within distance < eps of an existing Coreset memory
        are dropped from Addendum, while the corresponding Coreset memory has its vivacity
        refreshed to 1.0. Surviving Addendum sequences are kept.
        """
        if not self._coreset or not self._addendum:
            return {"dropped": 0, "refreshed": 0, "surviving": len(self._addendum)}

        while len(self._coreset_vivacity) < len(self._coreset):
            self._coreset_vivacity.append(1.0)

        cross_D = cross_sequence_distances(self._addendum, self._coreset, weights=weights)
        min_dists, min_indices = torch.min(cross_D, dim=1)

        surviving = []
        surviving_meta = []
        n_dropped = 0
        n_refreshed = 0

        for i, seq in enumerate(self._addendum):
            d_min = min_dists[i].item()
            idx_core = min_indices[i].item()
            if d_min < eps:
                self._coreset_vivacity[idx_core] = 1.0
                n_refreshed += 1
                n_dropped += 1
            else:
                surviving.append(seq)
                if i < len(self._addendum_meta):
                    surviving_meta.append(self._addendum_meta[i])
                else:
                    surviving_meta.append("wake")

        self._addendum = surviving
        self._addendum_meta = surviving_meta

        return {
            "dropped": n_dropped,
            "refreshed": n_refreshed,
            "surviving": len(self._addendum),
        }

    def consolidate_addendum_into_coreset(self, eps: float = 0.04, decay: float = 0.01,
                                          dedup_intra: bool = True,
                                          weights: torch.Tensor | None = None) -> dict:
        """Commit surviving Addendum sequences into Coreset with decay and eviction.

        If dedup_intra is True, removes redundant duplicates within Addendum.
        Deduplicates against Coreset if not already done, appends survivors,
        applies memory decay across Coreset vivacities, and evicts lowest-vivacity
        memories if exceeding pool_capacity.
        """
        n_intra_dropped = 0
        if dedup_intra and len(self._addendum) > 1:
            n_intra_dropped = self.deduplicate_addendum(eps=eps, weights=weights)

        cross_res = self.deduplicate_against_coreset(eps=eps, weights=weights)
        n_cross_dropped = cross_res["dropped"]
        n_refreshed = cross_res["refreshed"]

        while len(self._coreset_vivacity) < len(self._coreset):
            self._coreset_vivacity.append(1.0)

        n_added = 0
        for seq in self._addendum:
            self._coreset.append(seq)
            self._coreset_vivacity.append(1.0)
            n_added += 1

        if decay > 0.0:
            self._coreset_vivacity = [max(0.0, v - decay) for v in self._coreset_vivacity]

        n_evicted = 0
        if len(self._coreset) > self.pool_capacity:
            excess = len(self._coreset) - self.pool_capacity
            paired = sorted(zip(self._coreset_vivacity, self._coreset), key=lambda x: x[0], reverse=True)
            kept = paired[:self.pool_capacity]
            self._coreset_vivacity = [v for v, _ in kept]
            self._coreset = [s for _, s in kept]
            n_evicted = excess

        self._addendum.clear()
        self._addendum_meta.clear()

        return {
            "dedup_intra_dropped": n_intra_dropped,
            "dedup_cross_dropped": n_cross_dropped,
            "added": n_added,
            "refreshed": n_refreshed,
            "evicted": n_evicted,
            "coreset_size": len(self._coreset),
        }

    def filter_addendum(self, kept: list[list[list[float]]],
                        kept_meta: list[str] | None = None) -> None:
        """Replace Addendum with only the retained sequences."""
        self._addendum = list(kept)
        if kept_meta is not None:
            self._addendum_meta = list(kept_meta)
        else:
            self._addendum_meta = self._addendum_meta[:len(kept)]

    def commit_addendum(self, dedup: bool = False, eps: float = 0.04, decay: float = 0.01,
                        weights: torch.Tensor | None = None) -> int:
        """Merge all surviving Addendum sequences into Coreset, then clear Addendum.
        If dedup=True, uses consolidate_addendum_into_coreset; otherwise uses direct extend.
        Returns the number of sequences committed.
        """
        if dedup:
            stats = self.consolidate_addendum_into_coreset(eps=eps, decay=decay, weights=weights)
            return stats["added"]
        n = len(self._addendum)
        self._coreset.extend(self._addendum)
        self._coreset_vivacity.extend([1.0] * n)
        self._addendum.clear()
        self._addendum_meta.clear()
        return n

    def consolidate(self, stride: int = 1) -> int:
        """Backward-compatible extraction + immediate commit."""
        added = self.extract_addendum(stride=stride)
        self.commit_addendum()
        return added

    def clear_journal(self) -> None:
        """Discard active and archived segments in the session journal, preserving Coreset and Addendum."""
        self._segments.clear()
        self._current.clear()

    def clear(self) -> None:
        """Discard all session segments, Addendum, and Coreset."""
        self.clear_journal()
        self._coreset.clear()
        self._coreset_vivacity.clear()
        self._addendum.clear()
        self._addendum_meta.clear()

    def save(self, path: str) -> None:
        """Save Coreset, Addendum, and active segments to disk."""
        torch.save({
            "seq_len": self.seq_len,
            "n_ctx": self.n_ctx,
            "coreset": list(self._coreset),
            "coreset_vivacity": list(self._coreset_vivacity),
            "persistent_pool": list(self._coreset),
            "addendum": list(self._addendum),
            "addendum_meta": list(self._addendum_meta),
            "segments": list(self._segments),
            "current": self._current,
        }, path)

    def load(self, path: str) -> bool:
        """Load Coreset, Addendum, and segments from disk. Returns True if loaded."""
        if os.path.exists(path):
            data = torch.load(path, weights_only=False)
            pool = data.get("coreset", data.get("persistent_pool", []))
            self._coreset = [list(seq) for seq in pool][:self.pool_capacity]
            viv = data.get("coreset_vivacity", None)
            if viv is not None and len(viv) == len(self._coreset):
                self._coreset_vivacity = [float(v) for v in viv]
            else:
                self._coreset_vivacity = [1.0] * len(self._coreset)
            self._addendum = list(data.get("addendum", []))
            self._addendum_meta = list(data.get("addendum_meta", []))
            self._segments = deque(data.get("segments", []))
            self._current = list(data.get("current", []))
            return True
        return False

    def _enforce_capacity(self) -> None:
        """Ensure total extractable sequences in the session journal do not exceed ``capacity``."""
        excess = self.journal_len - self.capacity
        while excess > 0 and self._segments:
            s0 = self._segments[0]
            removable = len(s0) - self.seq_len + 1
            if excess >= removable:
                self._segments.popleft()
                excess -= removable
            else:
                self._segments[0] = s0[excess:]
                excess = 0
        if excess > 0 and len(self._current) >= self.seq_len:
            removable = len(self._current) - self.seq_len + 1
            trim = min(excess, removable)
            self._current = self._current[trim:]

    def sample_coreset(self, batch: int) -> list[list[list[float]]]:
        """Sample ``batch`` sequences from the Coreset."""
        if not self._coreset:
            return []
        k = min(batch, len(self._coreset))
        return self._rng.sample(list(self._coreset), k)

    def sample_addendum(self, batch: int) -> list[list[list[float]]]:
        """Sample ``batch`` sequences from the Addendum."""
        if not self._addendum:
            return []
        k = min(batch, len(self._addendum))
        return self._rng.sample(self._addendum, k)

    def sample(self, batch: int) -> list[list[list[float]]]:
        """Sample ``batch`` sequences of length ``seq_len``.

        Prioritizes Coreset + Addendum if available.
        Otherwise falls back to sliding windows in the journal.
        Returns [] if empty.
        """
        pool = list(self._coreset) + self._addendum
        if pool:
            k = min(batch, len(pool))
            return self._rng.sample(pool, k)

        valid_segs = self._valid_segments
        if not valid_segs:
            return []

        counts = [len(s) - self.seq_len + 1 for s in valid_segs]
        total = sum(counts)
        if total == 0:
            return []

        k = min(batch, total)
        indices = self._rng.sample(range(total), k)

        # Prefix sums for fast segment lookup
        cum = list(itertools.accumulate(counts))
        samples = []
        for idx in indices:
            seg_idx = bisect.bisect_right(cum, idx)
            offset = idx - (cum[seg_idx - 1] if seg_idx > 0 else 0)
            seq = valid_segs[seg_idx][offset:offset + self.seq_len]
            samples.append(seq)
        return samples

    def _sequence_relief(self, seq: list) -> float:
        """Relief score of one stored sequence under this buffer's geometry."""
        if len(seq) < self.seq_len:
            return 0.0
        return sequence_relief(seq, n_ctx=self.n_ctx, eos_time_w=self._eos_time_w,
                               context_time_w=self._context_time_w)

    def sample_for_policy(self, batch: int, beta: float = 2.0) -> list[list[list[float]]]:
        """Sample ``batch`` sequences prioritized by long-term cost relief.

        Higher probability is given to sequences whose consequence window
        (s5..s10, time-weighted like the EOS — the long term dominates) is
        cheaper than the decision window before it (s0..s4): recovery
        trajectories. Channels carry equally (no salve_cost weighting).
        Uses weighted stochastic sampling without replacement (Efraimidis-Spirakis).
        """
        pool = list(self._coreset) + self._addendum
        if not pool:
            pool = self.sample(batch)
            if not pool or len(pool) <= batch:
                return pool

        k = min(batch, len(pool))
        weights = []
        for seq in pool:
            relief = self._sequence_relief(seq)
            relief_clamped = max(-1.0, min(1.0, relief))
            weights.append(math.exp(beta * relief_clamped))

        # Efraimidis-Spirakis weighted sampling without replacement
        keys = [
            -math.log(max(1e-12, self._rng.random())) / w
            for w in weights
        ]
        order = sorted(range(len(pool)), key=lambda i: keys[i])
        return [pool[i] for i in order[:k]]



class Brain:
    """Holds both networks, drives the live loop, and runs sleep training.

    Lifecycle in the main loop:

        brain.act(salve_t)            -> (theta_l, d_l, theta_r, d_r, tail_t)
        ... physics steps, sensors ...
        brain.record(salve_t, salve_next)
        brain.sleep()                 # on demand: offline training of both models
    """

    def __init__(self, lr_wm: float = 3e-4, lr_pol: float = 2e-4, act_std: float = 0.05,
                 explore_std: float = 0.2, device: str | None = None, seed: int = 0,
                 compile_wm: bool = False,
                 seq_steps: int = DEFAULT_SEQ_STEPS,
                 n_ctx: int = DEFAULT_N_CTX):
        torch.manual_seed(seed)
        self.device, self.device_desc = M.configure_hardware(device)
        # Intra-op parallelism cap: 2 threads wins or ties in both regimes on
        # this class of CPU (measured: batch-1 inference 3-6x slower with 4
        # threads; batch-64 training slightly slower with 4 than with 2).
        if self.device.type == "cpu":
            torch.set_num_threads(min(2, torch.get_num_threads()))
        self.seq_steps = seq_steps
        self.n_ctx = n_ctx
        layout = compute_seq_layout(seq_steps, n_ctx)
        self.seq_len = layout["seq_len"]
        self.decision_pos = layout["decision_pos"]
        self.eos_pos = layout["eos_pos"]
        self.n_imagine = layout["n_imagine"]

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
            lr=lr_pol, weight_decay=1e-4)
        self.buffer = ExperienceBuffer(seq_len=seq_steps + 1, seed=seed, n_ctx=n_ctx)
        if seq_steps == DEFAULT_SEQ_STEPS and n_ctx == DEFAULT_N_CTX:
            self.mask = _MASK.to(self.device)
            self.mask_lookahead = _MASK_LOOKAHEAD.to(self.device)
            self.target_valid = _target_valid.to(self.device)
            self.target_weight = _target_weight.to(self.device)
            self._eos_w = _EOS_TIME_W_T.to(self.device)
            self._context_w = _CONTEXT_TIME_W_T.to(self.device)
        else:
            self.mask = build_salve_mask(self.seq_len, self.device)
            self.mask_lookahead = build_lookahead_mask(seq_steps, n_ctx, self.device)
            t_valid, t_weight = build_target_masks(seq_steps)
            self.target_valid = t_valid.to(self.device)
            self.target_weight = t_weight.to(self.device)
            eos_w = compute_eos_time_weights(self.n_imagine)
            self._eos_w = torch.tensor(eos_w, dtype=torch.float32, device=self.device)
            context_w = compute_context_time_weights(self.n_ctx + 1)
            self._context_w = torch.tensor(context_w, dtype=torch.float32, device=self.device)
        self._eos_tmpl = _EOS_TEMPLATE.to(self.device)
        self._eos_cw = _EOS_CHANNEL_W_T.to(self.device)  # innate channel weights
        # WM training step: eager by default, optionally compiled as one
        # fused graph (static shapes [B, 173, 25] — the training loop is the
        # only caller, so no recompilation churn). The module itself stays
        # eager: the latent-capture hook and the checkpoints remain intact.
        # Separate compiled entry for the lookahead pass: its mask is baked
        # in, so causal and lookahead keep disjoint compiled graphs.
        self._wm_loss = (torch.compile(self._wm_loss_impl, dynamic=False)
                         if compile_wm else self._wm_loss_impl)
        self._wm_loss_lookahead = (torch.compile(self._wm_loss_lookahead_impl, dynamic=False)
                                   if compile_wm else self._wm_loss_lookahead_impl)
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
        self._act_min = torch.where(self._act_signed, -1.0, 0.0).to(self.device)
        self._act_max = torch.tensor(1.0, device=self.device)
        self._act_tmpl = _ACT_TEMPLATE.to(self.device)
        self._kalman_max_v = torch.tensor([0.20, 0.15, 0.20, 0.15, 0.20], device=self.device)
        # Rolling history of recent (state_tokens, action_tokens) for the
        # wake-time world model context. The policy reads the world model's
        # intermediate latent, which needs the last SEQ_STEPS-1 transitions.
        self._wake_history: list[tuple[list, list]] = []
        self._cached_latent = None   # [1, d_model] — refreshed at 1 Hz by wake_tick
        self._cached_scalars = None  # [1, 47] — scalar state at last wake_tick
        # last sleep stats, for the HUD
        self.last_wm_loss = float("nan")
        self.last_wm_impact = float("nan")
        self.last_wm_coreset_loss = float("nan")
        self.last_wm_addendum_loss = float("nan")
        self.last_wm_lookahead_loss = float("nan")
        self.last_filter_stats: dict | None = None
        self.last_pol_loss = float("nan")
        self.last_pol_stats: dict | None = None
        self.last_dream_record: DreamRecord | None = None
        self.sleep_cycle_stats: dict = {}
        self.sleep_wm_epochs = 0
        self.sleep_wm_step = 0
        self.mode = "wake"
        # inference timing (exponential moving average, ms)
        self._wm_time = 0.0
        self._pol_time = 0.0
        self._time_alpha = 0.1
        # Experience recording control (pause when stationary)
        self._recording = True
        self._static_ticks = 0
        self._last_salve: list[list[float]] | None = None
        self._static_threshold = 6
        self._static_tol = 1e-4

    # --- live loop -----------------------------------------------------------
    def _sync_device(self) -> None:
        """Block until queued device work finishes. CPU is eager, but
        CUDA/MPS launch kernels asynchronously: without this, perf_counter
        timings measure launch overhead (~0.1 ms) instead of execution."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    @torch.inference_mode()
    def wake_tick(self, salve) -> None:
        """World model tick: run a forward pass on the rolling history +
        current salve to produce a fresh latent for the policy. This is the
        perceptual pass — it runs every WM tick, brain on or off, so the
        policy's observation is never stale (journaling is done by record)."""
        self.mode = "wake"
        cur_state = state_tokens(salve)
        a_toks = action_tokens(salve)
        ctx = self._build_context(cur_state)
        self.world.eval()
        t0 = time.perf_counter()
        self.world(ctx)
        self._cached_latent = self.latent_norm(self.world.last_latent())  # [1, d_model]
        self._sync_device()
        dt_ms = (time.perf_counter() - t0) * 1000
        # seed the EMA with the first measurement: blending into a zero
        # prior under-reports the first ~10 ticks (a brain that ticks only
        # once would otherwise display 10% of its real inference time).
        if self._wm_time <= 0.0:
            self._wm_time = dt_ms
        else:
            self._wm_time = (1 - self._time_alpha) * self._wm_time \
                + self._time_alpha * dt_ms
        self._cached_scalars = self._policy_scalars_batch(
            torch.tensor(cur_state, dtype=torch.float32,
                         device=self.device).unsqueeze(0))   # [1, 47]
        # update history with this transition's state + action
        self._wake_history.append((cur_state, a_toks))
        if len(self._wake_history) > self.n_ctx:
            self._wake_history.pop(0)

    @torch.inference_mode()
    def act(self, salve, std=None) -> tuple[float, float, float, float, float]:
        """Return 5 actuator consignes. Uses fresh scalar sensory readings
        from ``salve`` directly and reuses the cached latent from the last
        wake_tick(). If no latent yet (first tick), runs a fallback forward."""
        if self._cached_latent is None:
            self.wake_tick(salve)   # has to run wm at least once
        t0 = time.perf_counter()
        cur_state = state_tokens(salve)
        cur_state_t = torch.tensor(cur_state, dtype=torch.float32,
                                   device=self.device).unsqueeze(0)
        cur_scalars = self._policy_scalars_batch(cur_state_t)
        self._cached_scalars = cur_scalars
        pol_in = torch.cat([cur_scalars, self._cached_latent], dim=1)
        action, _, _ = self.policy.sample(pol_in, self.act_std if std is None else std)
        self._sync_device()
        dt_ms = (time.perf_counter() - t0) * 1000
        if self._pol_time <= 0.0:
            self._pol_time = dt_ms
        else:
            self._pol_time = (1 - self._time_alpha) * self._pol_time \
                + self._time_alpha * dt_ms
        a = action[0].tolist()
        return a[0], a[1], a[2], a[3], a[4]

    @property
    def is_recording(self) -> bool:
        """Whether experience recording into the buffer is currently active."""
        return self._recording

    def record(self, salve) -> bool:
        """Journal one transition into the experience buffer.

        If the current sequence is long enough (>= buffer.seq_len) and the
        tokens (animal, environment, consignes) do not move for 6 successive
        ticks, recording is stopped. Recording resumes as soon as tokens move again.
        The context and in-progress segment are preserved without clearing or boundary.
        Returns True if the salve was recorded into the buffer, False if skipped.
        """
        if hasattr(salve, "tolist"):
            salve = salve.tolist()

        if self._last_salve is None:
            moved = True
        else:
            moved = _salve_moved(salve, self._last_salve, tol=self._static_tol)

        if moved:
            self._static_ticks = 0
        else:
            self._static_ticks += 1

        recorded = False
        if self._recording:
            self.buffer.push(salve)
            recorded = True
            if (len(self.buffer._current) >= self.buffer.seq_len
                    and self._static_ticks >= self._static_threshold):
                self._recording = False
        else:
            if moved:
                self._recording = True
                self.buffer.push(salve)
                recorded = True

        self._last_salve = salve
        return recorded

    def clear_history(self) -> None:
        """Clear the wake context history, cached latent, and seal the buffer segment."""
        self._wake_history.clear()
        self._cached_latent = None
        self._cached_scalars = None
        self._recording = True
        self._static_ticks = 0
        self._last_salve = None
        self.buffer.boundary()

    def boundary(self) -> None:
        """Signal a temporal discontinuity (facing flip, reset, teleportation)."""
        self.clear_history()

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
            ctx[0, pos + _INTERO_IDX, SIG_OFFSET:] = 0.0
            pos += STATE_TOKENS
            ctx[0, pos:pos + ACTION_TOKENS] = torch.tensor(
                a_toks, dtype=torch.float32, device=self.device)
            pos += ACTION_TOKENS
        ctx[0, pos:pos + STATE_TOKENS] = torch.tensor(
            cur_state, dtype=torch.float32, device=self.device)
        ctx[0, pos + _INTERO_IDX, SIG_OFFSET:] = 0.0
        return ctx

    def _refresh_wake_latent(self) -> None:
        """Recompute _cached_latent and _cached_scalars from _wake_history with
        the current world model weights (e.g. after sleep training), preventing
        wake shock from obsolete pre-sleep latents."""
        if not self._wake_history:
            return
        cur_state = self._wake_history[-1][0]
        n = len(self._wake_history) - 1
        ctx_len = n * SALVE_TOKENS + STATE_TOKENS
        ctx = torch.zeros(1, ctx_len, DENSE_DIM, device=self.device)
        pos = 0
        for s_toks, a_toks in self._wake_history[:-1]:
            ctx[0, pos:pos + STATE_TOKENS] = torch.tensor(
                s_toks, dtype=torch.float32, device=self.device)
            ctx[0, pos + _INTERO_IDX, SIG_OFFSET:] = 0.0
            pos += STATE_TOKENS
            ctx[0, pos:pos + ACTION_TOKENS] = torch.tensor(
                a_toks, dtype=torch.float32, device=self.device)
            pos += ACTION_TOKENS
        ctx[0, pos:pos + STATE_TOKENS] = torch.tensor(
            cur_state, dtype=torch.float32, device=self.device)
        ctx[0, pos + _INTERO_IDX, SIG_OFFSET:] = 0.0
        self.world.eval()
        with torch.inference_mode():
            self.world(ctx)
            self._cached_latent = self.latent_norm(self.world.last_latent())
            self._cached_scalars = self._policy_scalars_batch(
                torch.tensor(cur_state, dtype=torch.float32,
                             device=self.device).unsqueeze(0))
        self._recording = True
        self._static_ticks = 0
        self._last_salve = None

    # --- sleep: world model training ----------------------------------------
    def _batch_tensors(self, sequences):
        """Build a [B, L, 25] tensor for a batch of multi-step transition
        sequences.

        Each sequence is a list of seq_steps salves of 16 Sensor tokens and 3 Actuator tokens:
        [s0, a0, s1, a1, ..., a_{N-1}, sN]"""
        B = len(sequences)

        vals = torch.zeros(B, self.seq_len, DENSE_DIM, device=self.device)
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
            assert pos == self.seq_len, f"Expected pos={self.seq_len}, got {pos}"

            # Zero-padding for intermediate Token 12 (interoception) in salves
            # 0..seq_steps-1. The fatigue/souffrance absolute level is the
            # CARRIER (onde porteuse) of the outcome signal: a slow baseline
            # with no effect on the simulation (excluded from the policy
            # inputs). Masking it from every input keeps the carrier from
            # leaking back in as a spurious baseline.
            for k in range(self.seq_steps):
                p_intero = k * SALVE_TOKENS + _INTERO_IDX
                vals[b, p_intero, SIG_OFFSET:] = 0.0

            # Terminal EOS token (position eos_pos): the DENOUEMENT DELTA the
            # hindsight mechanism conditions on — per innate cost channel,
            # the time-weighted consequence window, hyperbolic weights: the long term
            # dominates, minus the flat decision window. Slots normalized 0.5 = neutre,
            # < 0.5 = rétablissement, > 0.5 = aggravation.
            p_final = self.eos_pos
            early_cost = [k * SALVE_TOKENS + _COST_IDX for k in range(self.n_ctx + 1)]
            late_cost = [k * SALVE_TOKENS + _COST_IDX
                         for k in range(self.n_ctx + 1, self.seq_steps + 1)]
            early_rew = [k * SALVE_TOKENS + _REWARD_IDX for k in range(self.n_ctx + 1)]
            late_rew = [k * SALVE_TOKENS + _REWARD_IDX
                        for k in range(self.n_ctx + 1, self.seq_steps + 1)]
            vals[b, p_final, :SIG_OFFSET] = self._eos_tmpl[:SIG_OFFSET]
            early_c = (vals[b, early_cost, SIG_OFFSET:SIG_OFFSET + 5]
                       * self._context_w[:, None]).sum(dim=0)
            late_c = (vals[b, late_cost, SIG_OFFSET:SIG_OFFSET + 5]
                      * self._eos_w[:, None]).sum(dim=0)
            d_c = ((late_c - early_c) / _EOS_DELTA_SCALE).clamp(-1.0, 1.0)
            vals[b, p_final, SIG_OFFSET:SIG_OFFSET + 5] = (d_c + 1.0) / 2.0
            early_k = (vals[b, early_rew, SIG_OFFSET] * self._context_w).sum()
            late_k = (vals[b, late_rew, SIG_OFFSET] * self._eos_w).sum()
            d_k = ((late_k - early_k) / _EOS_DELTA_SCALE).clamp(-1.0, 1.0)
            vals[b, p_final, SIG_OFFSET + 5] = (d_k + 1.0) / 2.0
            vals[b, p_final, SIG_OFFSET + _EOS_NSLOTS:] = 0.0
        return vals

    def _wm_loss_impl(self, vals: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Teacher-forced forward + impact-weighted masked-MSE loss over valid
        signal slots.

        ``weights`` [B] scales each sequence's contribution to the loss, so
        its descent step is proportional to its measured cost impact
        (|cost(S10) - cost(S4)|). The weighted mean keeps the overall step
        magnitude on the same scale as the uniform loss, independent of the
        cost-unit scale of the impacts.
        Side-effect free so torch.compile can capture it as one graph; the
        backward pass through this graph is also compiled (AOTAutograd)."""
        return self._wm_masked_mse(self.world(vals), vals, weights)

    def _wm_loss_lookahead_impl(self, vals: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Same masked-MSE loss under the EOS-lookahead attention mask: the
        decision position (76) also attends the realized terminal EOS (172),
        so the head there learns P(a4 | context, EOS) — the hindsight action —
        while every other position stays causal. No forcing: each sequence
        carries its own realized EOS, good or bad outcomes alike."""
        return self._wm_masked_mse(
            self.world(vals, attn_mask=self.mask_lookahead, is_causal=False),
            vals, weights)

    def _wm_masked_mse(self, pred: torch.Tensor, vals: torch.Tensor,
                       weights: torch.Tensor) -> torch.Tensor:
        """Masked, reward-boosted, impact-weighted MSE over valid signal slots
        (shared by the causal and lookahead passes)."""
        pred_signals = pred[:, :-1, :]                   # [B, L-1, 16]
        tgt_signals = vals[:, 1:, SIG_OFFSET:]           # [B, L-1, 16]
        w_b = self.target_weight.unsqueeze(0).expand_as(pred_signals)  # [B, L-1, 16]
        sq = (pred_signals - tgt_signals) ** 2
        sqm = (sq * w_b).sum(dim=(1, 2))                  # [B] per-sequence weighted squared error
        wsum = w_b.sum(dim=(1, 2)).clamp(min=1.0)          # [B] total slot weight per sequence
        mse = sqm / wsum                                   # [B] per-sequence weighted MSE
        return (mse * weights).sum() / weights.sum().clamp(min=1e-8)

    def _train_wm_batch(self, sequences: list[list[list[float]]],
                        lookahead: bool = False) -> float:
        """Run one training optimization step on a batch of transition sequences.

        Each sequence's descent step is proportional to its actual biological impact
        |relief|: the absolute magnitude of change between the decision window
        and the consequence window. A small floor weight (``_WM_IMPACT_FLOOR``)
        keeps neutral sequences learning at a trickle and prevents an all-neutral
        batch from yielding a zero loss that would trip the target-loss stopping criterion.

        ``lookahead=True`` runs the same loss under the EOS-lookahead
        attention mask (see ``_wm_loss_lookahead_impl``)."""
        vals = self._batch_tensors(sequences)             # [B, L, 25]
        weights = sequence_impact(
            sequences, n_ctx=self.n_ctx,
            eos_time_w=self.buffer._eos_time_w
        ).to(self.device) + _WM_IMPACT_FLOOR
        self.last_wm_impact = (weights.sum() / len(sequences)).item()
        if lookahead:
            loss = self._wm_loss_lookahead(vals, weights)
        else:
            loss = self._wm_loss(vals, weights)
        self.opt_wm.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world.parameters(), 1.0)
        self.opt_wm.step()
        return loss.item()

    def train_world(self, epochs: int = 512, batch: int = 32):
        """Train the world model. Generator: yields (label, value) after each
        epoch."""
        if len(self.buffer) < 1:
            return
        self.world.train()
        losses = []
        for ep in range(epochs):
            self.sleep_wm_step = ep + 1
            sequences = self.buffer.sample(batch)
            if not sequences:
                break
            loss_val = self._train_wm_batch(sequences)
            losses.append(loss_val)
            yield "wm", loss_val
        self.last_wm_loss = sum(losses) / len(losses) if losses else float("nan")

    def evaluate_sequences_surprise(self, sequences: list[list[list[float]]], batch_size: int = 32) -> list[float]:
        """Evaluate surprise scores for a list of sequences under the current World Model.

        Surprise score combines:
          1. Dynamical prediction error (MSE over valid sensor token positions).
          2. Discrepancy between predicted and actual innate cost/reward over the trajectory.
          Surprise = dyn_loss * (1.0 + mean_cost_divergence).
        """
        if not sequences:
            return []
        self.world.eval()
        surprises: list[float] = []
        n_cost = len(_COST_KEYS)
        with torch.inference_mode():
            for i in range(0, len(sequences), batch_size):
                chunk = sequences[i : i + batch_size]
                B = len(chunk)
                vals = self._batch_tensors(chunk)
                pred = self.world(vals, attn_mask=self.mask)
                pred_signals = pred[:, :-1, :]
                tgt_signals = vals[:, 1:, SIG_OFFSET:]
                m_b = self.target_valid.unsqueeze(0).expand_as(pred_signals)

                sq = (pred_signals - tgt_signals) ** 2
                valid_counts = m_b.sum(dim=[1, 2]).clamp(min=1)
                dyn_loss = (sq * m_b).sum(dim=[1, 2]) / valid_counts  # [B]

                cost_diff_sum = torch.zeros(B, device=self.device)
                for k in range(1, self.seq_steps + 1):
                    p_cost = k * SALVE_TOKENS + _COST_IDX - 1
                    p_conf = k * SALVE_TOKENS + _REWARD_IDX - 1

                    c_pred = pred_signals[:, p_cost, :n_cost]
                    denorm_pred = torch.where(self._cost_signed,
                                              (2 * c_pred - 1) * self._cost_scales,
                                              c_pred * self._cost_scales)
                    conf_pred = pred_signals[:, p_conf, 0]
                    conf_phys_pred = (2 * conf_pred - 1) * _CONFORT_SCALE
                    cost_pred = (denorm_pred * self._cost_w).sum(1) + _CONFORT_W * conf_phys_pred

                    c_tgt = tgt_signals[:, p_cost, :n_cost]
                    denorm_tgt = torch.where(self._cost_signed,
                                             (2 * c_tgt - 1) * self._cost_scales,
                                             c_tgt * self._cost_scales)
                    conf_tgt = tgt_signals[:, p_conf, 0]
                    conf_phys_tgt = (2 * conf_tgt - 1) * _CONFORT_SCALE
                    cost_tgt = (denorm_tgt * self._cost_w).sum(1) + _CONFORT_W * conf_phys_tgt

                    cost_diff_sum += (cost_pred - cost_tgt).abs()

                mean_cost_diff = cost_diff_sum / self.seq_steps
                seq_surprise = dyn_loss * (1.0 + mean_cost_diff)
                surprises.extend(seq_surprise.tolist())
        return surprises

    def compute_saliency_weights(self, sequences: list[list[list[float]]] | None = None,
                                 batch_size: int = 32,
                                 floor_pct: float = 0.05) -> torch.Tensor:
        """Compute sequence feature sensitivity weights backpropagated
        from the terminal EOS token.

        Uses the World Model's autograd gradients on the terminal EOS
        prediction (position seq_len - 2, predicting token eos_pos: the
        time-weighted innate cost aggregate, 6 slots).

        Returns a 1D tensor of shape [D_TOTAL] with mean == 1.0, suitable for weighted RMSE distance.
        If buffer is empty and no sequences provided, returns uniform ones tensor.
        """
        D_TOTAL = (self.seq_steps + 1) * SALVE_TOKENS * N_SIGNAL
        if sequences is None:
            sequences = self.buffer.sample(batch_size)
            if not sequences and self.buffer.coreset_size > 0:
                sequences = self.buffer.sample_coreset(batch_size)
            if not sequences and self.buffer.addendum_size > 0:
                sequences = self.buffer.sample_addendum(batch_size)
        if not sequences:
            return torch.ones(D_TOTAL, device=self.device)

        seqs_sample = sequences[:batch_size]
        self.world.eval()

        vals = self._batch_tensors(seqs_sample).detach().requires_grad_(True)
        pred = self.world(vals, attn_mask=self.mask)

        # Target: the terminal EOS token prediction (position seq_len - 2
        # predicts token eos_pos) — the denouement delta, scalarized as the
        # channel-hierarchy-weighted deviation from neutral (0.5).
        eos_delta = ((pred[:, self.seq_len - 2, :_EOS_NSLOTS] * self._eos_cw).sum(dim=-1)
                     - 0.5)

        # Vectorized backward pass across all sample sequences
        grad_out = torch.autograd.grad(eos_delta.sum(), vals)[0]
        signal_grads = grad_out[:, :, SIG_OFFSET:]
        mean_sens = signal_grads.abs().mean(dim=0)

        # Assemble into full sequence layout
        w_grid = torch.zeros(self.seq_steps + 1, SALVE_TOKENS, N_SIGNAL, device=self.device)
        w_grid[:self.seq_steps, :, :] = mean_sens[:self.seq_steps * SALVE_TOKENS].view(self.seq_steps, SALVE_TOKENS, N_SIGNAL)
        w_grid[self.seq_steps, :STATE_TOKENS, :] = mean_sens[self.seq_steps * SALVE_TOKENS : self.seq_steps * SALVE_TOKENS + STATE_TOKENS]

        w_flat = w_grid.view(-1)  # [2816]
        floor = floor_pct * w_flat.mean().clamp(min=1e-8)
        w_floored = w_flat + floor
        w_norm = w_floored / w_floored.mean().clamp(min=1e-8)

        return w_norm.detach()

    def trusted_saliency_weights(self, **kwargs) -> torch.Tensor | None:
        """EOS-saliency weights, or None when the World Model is not trusted yet.

        The saliency-weighted distance is only meaningful if the World Model has
        actually converged on the Coreset: gradients of an untrained WM carry no
        reliable sensitivity information. Returns the saliency weights only when
        the last World Model training on the Coset (``last_wm_coreset_loss``)
        exists (not NaN) and is below ``_WM_SALIENCY_MAX_LOSS`` (0.02); otherwise
        returns None, i.e. the naive uniform RMSE distance is used.
        """
        loss = self.last_wm_coreset_loss
        if math.isnan(loss) or loss >= _WM_SALIENCY_MAX_LOSS:
            return None
        return self.compute_saliency_weights(**kwargs)

    def _salve_cost_batch(self, gen: torch.Tensor) -> torch.Tensor:
        """Vectorized ``salve_cost`` over a batch of predicted states.

        gen: [B, 13, 25] (or [B, 16, 25]) float tensors. Returns [B] cost —
        the same weighted innate cost as ``tokenize.salve_cost`` but computed
        in one tensor op, no per-element Python loop or ``.tolist()`` sync."""
        # print("gen motivation:",
        #    gen[0, _COST_IDX, SIG_OFFSET:SIG_OFFSET + 5])
        # print("gen reward:",
        #     gen[0, _REWARD_IDX, SIG_OFFSET])
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
        """Vectorized normalized action [B, 5] -> [B, 3, 25].
        
        Model action space:
            signed   -> [-1, +1]
            unsigned -> [0, 1]

        Token signal space:
            [0, 1]
        """
        if action.dim() == 1:
            action = action.unsqueeze(0)

        B = action.shape[0]

        # Convert signed model outputs [-1,1] to token range [0,1].
        # Unsigned dimensions [0,1] are unchanged.
        norm = torch.where(
            self._act_signed,
            (action + 1.0) / 2.0,
            action,
        )

        toks = self._act_tmpl.unsqueeze(0).expand(B, -1, -1).clone()
        toks[:, self._act_tok, SIG_OFFSET + self._act_slot] = norm

        return toks

    def _decode_action_batch(self, act_toks: torch.Tensor) -> torch.Tensor:
        """Vectorized action token decoding: [B, 3, 25] -> [B, 5] in model action space.
        Signed consignes: [-1, +1]
        Unsigned consignes: [0, 1]
        """
        if act_toks.dim() == 2:
            act_toks = act_toks.unsqueeze(0)
        norm = act_toks[:, self._act_tok, SIG_OFFSET + self._act_slot]
        return torch.where(self._act_signed, 2.0 * norm - 1.0, norm)

    def _random_action_batch(self, B: int, device: torch.device) -> torch.Tensor:
        action = torch.rand(
            B,
            self._act_scales.numel(),
            device=device,
        )

        action[:, self._act_signed] = (
            2.0 * action[:, self._act_signed] - 1.0
        )

        return action

    def _explore_action_batch(self, base_action: torch.Tensor, sigma: float) -> torch.Tensor:
        """Generate exploratory action candidates perturbed around base_action.

        base_action: [B, 5]
        sigma: standard deviation of Gaussian perturbation
        Returns: [B, 5] valid clamped actions (signed dims in [-1, 1], unsigned in [0, 1])
        """
        noise = torch.randn_like(base_action) * sigma
        return torch.clamp(base_action + noise, self._act_min, self._act_max)

    # --- sleep: policy training via real context + imagined rollout ----------
    def _gen_state_cached(self, cache) -> torch.Tensor:
        """Generate the 13 state tokens continuing a populated ``cache``.

        Mirrors ``WorldModel.predict_next_state`` on the same token stream,
        with the rollout's intero zero-padding applied at the feedback
        boundary: the raw intero token is returned for cost evaluation while
        a zeroed copy enters the cache. The intero token is the last state
        token, so the zeroing only conditions the next salve's generation —
        exactly like the non-cached rollout context.
        """
        wm = self.world
        B = cache.k[0].shape[0]
        device = cache.k[0].device
        preds = []
        pred = cache.last_pred
        for i in range(STATE_TOKENS):
            sig = pred.clamp(0.0, 1.0) * wm.state_valid_mask[i].to(device)
            tok = wm.state_template[i].to(device).unsqueeze(0).expand(B, -1).clone()
            tok[:, SIG_OFFSET:] = sig
            feed = tok
            if i == _INTERO_IDX:
                feed = tok.clone()
                feed[:, SIG_OFFSET:] = 0.0
            out = wm.cache_forward(feed.unsqueeze(1), cache)
            preds.append(tok)
            pred = out[:, 0, :]
        return torch.stack(preds, dim=1)

    def _imagine_candidate(self, ctx, base_cache, action, past_a2, past_a3,
                           s_dec_toks, n_imagine, ctx_len, ctx_mask,
                           use_kv_cache, dream_cand_trajs,
                           candidate_idx) -> torch.Tensor:
        """Evaluate one candidate action over the 2 imagined futures under
        Bellman optimism, filling ``dream_cand_trajs[candidate_idx]`` with
        the per-regime trajectories. Returns the optimistic cost [B].

        ``use_kv_cache`` switches the rollouts from full-context recomputation
        to persistent per-branch KV caches (cloned where the futures
        diverge); both paths are numerically equivalent.
        """
        action_toks = self._encode_action_batch(action)

        # Step 0: candidate acts on decision state -> predicts first imagined state
        if use_kv_cache:
            cand_cache = base_cache.clone()
            with torch.inference_mode():
                self.world.cache_forward(action_toks, cand_cache)
                s5 = self._gen_state_cached(cand_cache)
        else:
            cand_ctx_0 = torch.cat([ctx, action_toks], dim=1)
            with torch.inference_mode():
                s5 = self.world.predict_next_state(cand_ctx_0)
        c0 = self._salve_cost_batch(s5)
        s5_zeroed = s5.clone()
        s5_zeroed[:, _INTERO_IDX, SIG_OFFSET:] = 0.0
        if use_kv_cache:
            hist = torch.cat([ctx, action_toks, s5_zeroed], dim=1)
        else:
            ctx_s5 = torch.cat([cand_ctx_0, s5_zeroed], dim=1)

        if n_imagine <= 1:
            return c0

        cand_a = action[0].tolist()
        c0_val = c0[0].item()
        s5_toks = s5[0].tolist()
        w0 = self._eos_w[0] if len(self._eos_w) > 0 else (1.0 / n_imagine)

        # Future 1: Kalman / inertia extrapolation
        v = 0.7 * (action - past_a3) + 0.3 * (past_a3 - past_a2)
        v = torch.clamp(v, -self._kalman_max_v, self._kalman_max_v)
        cur_a = action.clone()
        cost_kalman = w0 * c0
        cur_cache = cand_cache.clone() if use_kv_cache else None
        cur_ctx = None if use_kv_cache else ctx_s5
        cur_hist = hist if use_kv_cache else None
        cur_state_toks = s5_toks
        cur_cost_val = c0_val
        f1_steps = [DreamStep(state_tokens=s_dec_toks, action=cand_a, label=f"s{self.n_ctx}+cand")]
        for k in range(1, n_imagine):
            cur_a = torch.clamp(cur_a + v, self._act_min, self._act_max)
            v = v * 0.8
            next_a_toks = self._encode_action_batch(cur_a)
            with torch.inference_mode():
                if use_kv_cache:
                    self.world.cache_forward(next_a_toks, cur_cache)
                    gen = self._gen_state_cached(cur_cache)
                else:
                    cand_ctx = torch.cat([cur_ctx, next_a_toks], dim=1)
                    gen = self.world.predict_next_state(cand_ctx)
            gen_zeroed = gen.clone()
            gen_zeroed[:, _INTERO_IDX, SIG_OFFSET:] = 0.0
            sc = self._salve_cost_batch(gen)
            wk = self._eos_w[k] if k < len(self._eos_w) else (1.0 / n_imagine)
            cost_kalman = cost_kalman + wk * sc
            f1_steps.append(DreamStep(state_tokens=cur_state_toks, action=cur_a[0].tolist(), step_cost=cur_cost_val, label=f"s{self.n_ctx+k}"))
            if use_kv_cache:
                cur_hist = torch.cat([cur_hist, next_a_toks, gen_zeroed], dim=1)
            else:
                cur_ctx = torch.cat([cand_ctx, gen_zeroed], dim=1)
            cur_state_toks = gen[0].tolist()
            cur_cost_val = sc[0].item()
        f1_steps.append(DreamStep(state_tokens=cur_state_toks, action=None, step_cost=cur_cost_val, label=f"s{self.n_ctx+n_imagine}"))
        dream_cand_trajs[candidate_idx][0] = {"steps": f1_steps, "cost": cost_kalman[0].item()}

        # Future 2: Policy closed-loop reaction
        cur_state = s5
        cost_pol = w0 * c0
        cur_cache = cand_cache.clone() if use_kv_cache else None
        cur_ctx = None if use_kv_cache else ctx_s5
        cur_hist = hist if use_kv_cache else None
        cur_state_toks = s5_toks
        cur_cost_val = c0_val
        f2_steps = [DreamStep(state_tokens=s_dec_toks, action=cand_a, label=f"s{self.n_ctx}+cand")]
        for k in range(1, n_imagine):
            with torch.inference_mode():
                slide_src = cur_hist if use_kv_cache else cur_ctx
                slide_ctx = slide_src[:, -ctx_len:]
                self.world(slide_ctx, attn_mask=ctx_mask)
                next_latent = self.latent_norm(self.world.last_latent().detach())
                next_scalars = self._policy_scalars_batch(cur_state)
                next_pol_in = torch.cat([next_scalars, next_latent], dim=1)
                next_action = self.policy(next_pol_in)
                next_a_toks = self._encode_action_batch(next_action)
                if use_kv_cache:
                    self.world.cache_forward(next_a_toks, cur_cache)
                    gen = self._gen_state_cached(cur_cache)
                else:
                    cand_ctx = torch.cat([cur_ctx, next_a_toks], dim=1)
                    gen = self.world.predict_next_state(cand_ctx)
            gen_zeroed = gen.clone()
            gen_zeroed[:, _INTERO_IDX, SIG_OFFSET:] = 0.0
            sc = self._salve_cost_batch(gen)
            wk = self._eos_w[k] if k < len(self._eos_w) else (1.0 / n_imagine)
            cost_pol = cost_pol + wk * sc
            f2_steps.append(DreamStep(state_tokens=cur_state_toks, action=next_action[0].tolist(), step_cost=cur_cost_val, label=f"s{self.n_ctx+k}"))
            if use_kv_cache:
                cur_hist = torch.cat([cur_hist, next_a_toks, gen_zeroed], dim=1)
            else:
                cur_ctx = torch.cat([cand_ctx, gen_zeroed], dim=1)
            cur_state = gen
            cur_state_toks = gen[0].tolist()
            cur_cost_val = sc[0].item()
        f2_steps.append(DreamStep(state_tokens=cur_state_toks, action=None, step_cost=cur_cost_val, label=f"s{self.n_ctx+n_imagine}"))
        dream_cand_trajs[candidate_idx][1] = {"steps": f2_steps, "cost": cost_pol[0].item()}

        # Bellman optimism: optimistic minimum across the 2 futures
        return torch.minimum(cost_kalman, cost_pol)

    def train_policy(self, steps: int = 16, batch: int = 32,
                     n_imagine: int | None = None,
                     prioritize_relief: bool = True,
                     use_kv_cache: bool = True,
                     eos_lookahead: bool = True):
        """Policy training by supervised imitation of the WM's hindsight action.

        ``eos_lookahead=True`` (default) — efficient hindsight supervision: the
        policy's own action is rolled out to get the baseline cost C_pi and
        the imagined denouement delta (EOS); its deviation from neutral is
        forced eps-proportionally more favorable; the WM's lookahead mode
        generates the action associated with that outcome; a causal
        verification rollout accepts it only if it actually reduces the
        imagined cost. ~3 rollout-equivalents per batch instead of the legacy 6.

        ``eos_lookahead=False`` — legacy stochastic-candidate machinery,
        kept alive as a fallback. Also selected automatically when the
        imagination horizon cannot reach the terminal salve
        (n_imagine < self.n_imagine) or when the hindsight generation is
        systematically rejected (_LOOKAHEAD_FALLBACK_STREAK batches in a
        row without a single accepted sample).

        Generator: yields (label, value) after each step."""
        if n_imagine is None:
            n_imagine = self.n_imagine
        if len(self.buffer) < 1:
            return
        self.world.eval()
        self.policy.train()
        losses = []
        use_lookahead = eos_lookahead and n_imagine >= self.n_imagine
        if eos_lookahead and not use_lookahead:
            print(f"[sleep:policy] n_imagine={n_imagine} cannot reach the terminal salve "
                  f"(needs {self.n_imagine}) — falling back to the candidate machinery.")
        if use_lookahead:
            for step in range(steps):
                updated, loss = self._policy_step_lookahead(
                    batch, n_imagine, use_kv_cache, step, steps)
                losses.append(loss)   # 0.0 when the batch was fully rejected
                yield "pol", loss
        else:
            self.last_pol_stats = None
            for lbl, val in self._train_policy_candidates(
                    steps, batch, n_imagine,
                    prioritize_relief, use_kv_cache, step_offset=0):
                losses.append(val)
                yield lbl, val
        self.last_pol_loss = sum(losses) / len(losses) if losses else float("nan")

    # --- sleep: hindsight (EOS-lookahead) policy supervision ------------------
    def _dream_context_steps(self, ctx: torch.Tensor) -> list[DreamStep]:
        """Build the real-context storyboard row [s0+a0 ... s_{n_ctx}] for the Dream Theater."""
        n_ctx = self.n_ctx
        steps = []
        for s_i in range(n_ctx):
            s_tokens = ctx[0, s_i * SALVE_TOKENS: s_i * SALVE_TOKENS + STATE_TOKENS].tolist()
            a_vals = self._decode_action_batch(
                ctx[0:1, s_i * SALVE_TOKENS + STATE_TOKENS: (s_i + 1) * SALVE_TOKENS])[0].tolist()
            steps.append(DreamStep(state_tokens=s_tokens, action=a_vals,
                                   label=f"s{s_i}+a{s_i}"))
        steps.append(DreamStep(
            state_tokens=ctx[0, n_ctx * SALVE_TOKENS: n_ctx * SALVE_TOKENS + STATE_TOKENS].tolist(),
            action=None, label=f"s{n_ctx} (decision)"))
        return steps

    def _build_eos_token(self, eos_vals: torch.Tensor) -> torch.Tensor:
        """Terminal EOS token [B, 25] from normalized cost-aggregate slots in
        [0, 1] (lower = better): [effort, douleur, courbature, instabilite,
        vertige, confort]."""
        B = eos_vals.shape[0]
        tok = self._eos_tmpl.unsqueeze(0).expand(B, -1).clone()
        n = min(_EOS_NSLOTS, eos_vals.shape[1])
        tok[:, SIG_OFFSET:SIG_OFFSET + n] = eos_vals[:, :n]
        return tok

    def _rollout_candidate(self, ctx, base_cache, action, s_dec_toks, n_imagine,
                           ctx_len, ctx_mask, use_kv_cache):
        """Single-regime imagined rollout: ``action`` acts on s_{n_ctx}, then the
        policy reacts in closed loop to each imagined state for n_imagine - 1 more steps.

        Returns (cost [B], predicted terminal EOS [B, _EOS_NSLOTS] or None,
        dream steps). The cost is C = sum_k w_k c(s_{n_ctx+1+k}) weighted by
        the exact same time weights as the EOS token (self._eos_w); the
        predicted EOS is the DENOUEMENT DELTA: the time-weighted aggregate of
        the imagined salves' innate cost channels minus the flat decision-window
        baseline read from the real context (s0..s_{n_ctx})."""
        action_toks = self._encode_action_batch(action)
        dream_steps = [DreamStep(state_tokens=s_dec_toks, action=action[0].tolist(),
                                 label=f"s{self.n_ctx}+cand")]
        eos_pred = None
        # decision-window baseline (s0..s_{n_ctx}) from the real context, weighted towards s_{n_ctx}
        e_cost = [k * SALVE_TOKENS + _COST_IDX for k in range(self.n_ctx + 1)]
        e_rew = [k * SALVE_TOKENS + _REWARD_IDX for k in range(self.n_ctx + 1)]
        eos_early = torch.zeros(ctx.shape[0], _EOS_NSLOTS, device=ctx.device)
        eos_early[:, :5] = (ctx[:, e_cost, SIG_OFFSET:SIG_OFFSET + 5]
                            * self._context_w[None, :, None]).sum(dim=1)
        eos_early[:, 5] = (ctx[:, e_rew, SIG_OFFSET] * self._context_w[None, :]).sum(dim=1)
        if use_kv_cache:
            cache = base_cache.clone()
            with torch.inference_mode():
                self.world.cache_forward(action_toks, cache)
                s = self._gen_state_cached(cache)
        else:
            cand_ctx = torch.cat([ctx, action_toks], dim=1)
            with torch.inference_mode():
                s = self.world.predict_next_state(cand_ctx)
        c0 = self._salve_cost_batch(s)
        w0 = self._eos_w[0] if len(self._eos_w) > 0 else (1.0 / n_imagine)
        cost = w0 * c0
        eos_acc = torch.zeros(s.shape[0], _EOS_NSLOTS, device=s.device)
        eos_acc[:, :5] += w0 * s[:, _COST_IDX, SIG_OFFSET:SIG_OFFSET + 5]
        eos_acc[:, 5] += w0 * s[:, _REWARD_IDX, SIG_OFFSET]
        eos_wsum = w0

        def _eos_finalize():
            d = ((eos_acc / eos_wsum - eos_early) / _EOS_DELTA_SCALE).clamp(-1.0, 1.0)
            return ((d + 1.0) / 2.0).clamp(0.0, 1.0)

        s_zeroed = s.clone()
        s_zeroed[:, _INTERO_IDX, SIG_OFFSET:] = 0.0
        if use_kv_cache:
            hist = torch.cat([ctx, action_toks, s_zeroed], dim=1)
        else:
            cand_ctx = torch.cat([cand_ctx, s_zeroed], dim=1)
        if n_imagine == 1:
            eos_pred = _eos_finalize()
        cur_state = s
        cur_state_toks = s[0].tolist()
        cur_cost_val = c0[0].item()
        for k in range(1, n_imagine):
            with torch.inference_mode():
                slide_src = hist if use_kv_cache else cand_ctx
                slide_ctx = slide_src[:, -ctx_len:]
                self.world(slide_ctx, attn_mask=ctx_mask)
                next_latent = self.latent_norm(self.world.last_latent().detach())
                next_scalars = self._policy_scalars_batch(cur_state)
                next_pol_in = torch.cat([next_scalars, next_latent], dim=1)
                next_action = self.policy(next_pol_in)
                next_a_toks = self._encode_action_batch(next_action)
                if use_kv_cache:
                    self.world.cache_forward(next_a_toks, cache)
                    gen = self._gen_state_cached(cache)
                else:
                    cand_ctx = torch.cat([cand_ctx, next_a_toks], dim=1)
                    gen = self.world.predict_next_state(cand_ctx)
            gen_zeroed = gen.clone()
            gen_zeroed[:, _INTERO_IDX, SIG_OFFSET:] = 0.0
            sc = self._salve_cost_batch(gen)
            wk = self._eos_w[k] if k < len(self._eos_w) else (1.0 / n_imagine)
            cost = cost + wk * sc
            dream_steps.append(DreamStep(state_tokens=cur_state_toks,
                                         action=next_action[0].tolist(),
                                         step_cost=cur_cost_val, label=f"s{self.n_ctx + k}"))
            if use_kv_cache:
                hist = torch.cat([hist, next_a_toks, gen_zeroed], dim=1)
            else:
                cand_ctx = torch.cat([cand_ctx, gen_zeroed], dim=1)
            eos_acc[:, :5] += wk * gen[:, _COST_IDX, SIG_OFFSET:SIG_OFFSET + 5]
            eos_acc[:, 5] += wk * gen[:, _REWARD_IDX, SIG_OFFSET]
            eos_wsum = eos_wsum + wk
            if k + 1 == self.n_imagine:
                eos_pred = _eos_finalize()
            cur_state = gen
            cur_state_toks = gen[0].tolist()
            cur_cost_val = sc[0].item()
        dream_steps.append(DreamStep(state_tokens=cur_state_toks, action=None,
                                      step_cost=cur_cost_val, label=f"s{self.n_ctx + n_imagine}"))
        return cost, eos_pred, dream_steps

    def _lookahead_generate(self, ctx, base_cache, eos_forced, use_kv_cache):
        """Generate the hindsight action conditioned on the forced terminal EOS."""
        B, ctx_len, _ = ctx.shape
        device = ctx.device
        wm = self.world
        assert ctx_len == self.decision_pos + 1, f"hindsight generation expects the {self.decision_pos + 1}-token context"
        eos_tok = self._build_eos_token(eos_forced).unsqueeze(1)     # [B, 1, 25]
        x = torch.cat([ctx, eos_tok], dim=1)                          # [B, ctx_len + 1, 25]
        salve_pos = torch.arange(ctx_len + 1, device=device) // SALVE_TOKENS
        salve_pos[ctx_len] = self.seq_steps                           # EOS at its terminal salve
        gen_mask = build_lookahead_gen_mask(ctx_len, device, decision_pos=self.decision_pos)
        toks = []
        with torch.inference_mode():
            out = wm(x, attn_mask=gen_mask, is_causal=False,
                     salve_positions=salve_pos)
            sig = out[:, self.decision_pos, :].clamp(0.0, 1.0) * wm.action_valid_mask[0].to(device)
            tok = wm.action_template[0].to(device).unsqueeze(0).expand(B, -1).clone()
            tok[:, SIG_OFFSET:] = sig
            toks.append(tok)
            if use_kv_cache:
                cache = base_cache.clone()
                for i in range(1, ACTION_TOKENS):
                    wm.cache_forward(toks[-1].unsqueeze(1), cache)
                    sig = cache.last_pred.clamp(0.0, 1.0) * wm.action_valid_mask[i].to(device)
                    tok = wm.action_template[i].to(device).unsqueeze(0).expand(B, -1).clone()
                    tok[:, SIG_OFFSET:] = sig
                    toks.append(tok)
            else:
                curr = torch.cat([ctx, toks[0].unsqueeze(1)], dim=1)
                for i in range(1, ACTION_TOKENS):
                    o = wm(curr)
                    sig = o[:, -1, :].clamp(0.0, 1.0) * wm.action_valid_mask[i].to(device)
                    tok = wm.action_template[i].to(device).unsqueeze(0).expand(B, -1).clone()
                    tok[:, SIG_OFFSET:] = sig
                    toks.append(tok)
                    curr = torch.cat([curr, tok.unsqueeze(1)], dim=1)
        return torch.stack(toks, dim=1)                              # [B, 3, 25]

    def _policy_step_lookahead(self, batch, n_imagine, use_kv_cache,
                                step_idx, total_steps):
        """One policy step under hindsight supervision (see ``train_policy``).

        Baseline rollout of the policy action -> (C_pi, imagined EOS delta);
        multiplicative forcing of its deviation from neutral; hindsight
        generation of a4; causal verification rollout; purely comparative
        acceptance. Returns (updated, loss): updated=False means no sample
        was accepted and no optimizer step was taken."""
        sequences = self.buffer.sample_for_policy(batch)
        if not sequences:
            return True, 0.0
        B = len(sequences)
        device = self.device
        n_ctx = self.n_ctx
        ctx_len = n_ctx * SALVE_TOKENS + STATE_TOKENS
        vals = self._batch_tensors(sequences)
        ctx = vals[:, :self.decision_pos + 1, :].clone()
        s_dec_toks = ctx[0, n_ctx * SALVE_TOKENS: n_ctx * SALVE_TOKENS + STATE_TOKENS].tolist()
        ctx_mask = build_salve_mask(ctx_len, device)

        # 1. real context -> latent + deterministic policy baseline action
        base_cache = None
        with torch.inference_mode():
            if use_kv_cache:
                base_cache = KVCache(len(self.world.transformer.layers),
                                      self.world.d_model, B, device)
                self.world.cache_forward(ctx, base_cache)
            else:
                self.world(ctx, attn_mask=ctx_mask)
            latent = self.latent_norm(self.world.last_latent().detach())
        cur_scalars = self._policy_scalars_batch(ctx[:, -STATE_TOKENS:])
        pol_in = torch.cat([cur_scalars, latent], dim=1)
        with torch.inference_mode():
            policy_action = self.policy(pol_in)                     # [B, 5]

        # 2-4. Shared hindsight lookahead pipeline
        hl = self._hindsight_lookahead_pipeline(
            ctx, policy_action, s_dec_toks, n_imagine,
            ctx_len, ctx_mask, base_cache=base_cache,
            use_kv_cache=use_kv_cache)
        c_pi, eos_pi, steps_pi = hl["c_pi"], hl["eos_pi"], hl["steps_pi"]
        eos_forced = hl["eos_forced"]
        gen_action = hl["gen_action"]
        c_gen, eos_gen, steps_gen = hl["c_gen"], hl["eos_gen"], hl["steps_gen"]

        # 5. Purely comparative acceptance: the generated action must beat the baseline cost
        accepted = c_gen < c_pi                                     # [B]
        n_accepted = int(accepted.sum().item())
        self.last_pol_stats = {"accepted": n_accepted, "total": B}

        # 6. supervised update toward the accepted hindsight actions only
        if n_accepted > 0:
            pred_action = self.policy(pol_in)
            loss = torch.nn.functional.mse_loss(pred_action[accepted],
                                                 gen_action[accepted])
            self.opt_pol.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt_pol.step()
            loss_val = loss.item()
            updated = True
        else:
            loss_val = 0.0
            updated = False

        # 7. Dream Theater record (sample 0): lived sequence, WM hindsight
        #    completion, and the baseline-vs-hindsight decision
        acc0 = bool(accepted[0].item())
        context_steps = self._dream_context_steps(ctx)
        seq0 = sequences[0]
        lived_actions = self._decode_action_batch(torch.tensor(
            [salve[STATE_TOKENS:SALVE_TOKENS] for salve in seq0[:-1]],
            dtype=torch.float32, device=device))                  # [len(seq0)-1, 5]
        lived_steps = [DreamStep(
            state_tokens=[list(tok) for tok in salve[:STATE_TOKENS]],
            action=lived_actions[k].tolist() if k < len(seq0) - 1 else None,
            step_cost=salve_cost(salve), label=f"s{k}")
            for k, salve in enumerate(seq0)]
        predicted_steps = context_steps[:self.n_ctx] + steps_gen
        self.last_dream_record = DreamRecord(
            step_idx=step_idx,
            total_steps=total_steps,
            loss=loss_val,
            facing=1,
            context_steps=context_steps,
            candidate_actions=[policy_action[0].tolist(), gen_action[0].tolist()],
            trajectories=[
                DreamTrajectory(candidate_idx=0, regime_idx=0, regime_name="Policy",
                                steps=steps_pi, total_cost=c_pi[0].item(),
                                is_best_future=False, is_winner=False,
                                eos_pred=eos_pi[0].tolist() if eos_pi is not None else None),
                DreamTrajectory(candidate_idx=1, regime_idx=0, regime_name="Policy",
                                steps=steps_gen, total_cost=c_gen[0].item(),
                                is_best_future=True, is_winner=acc0,
                                eos_pred=eos_gen[0].tolist() if eos_gen is not None else None),
            ],
            best_candidate_idx=1 if acc0 else -1,
            best_future_indices=[0, 0],
            candidate_names=["0: POLICY", "1: HINDSIGHT"],
            eos_forced=eos_forced[0].tolist(),
            lived_steps=lived_steps,
            predicted_steps=predicted_steps,
            eos_realized=vals[0, self.eos_pos, SIG_OFFSET:SIG_OFFSET + _EOS_NSLOTS].tolist(),
        )
        return updated, loss_val

    def _hindsight_lookahead_pipeline(self, ctx: torch.Tensor,
                                      base_action: torch.Tensor,
                                      s_dec_toks: list[list[float]],
                                      n_imagine: int,
                                      ctx_len: int,
                                      ctx_mask: torch.Tensor,
                                      base_cache = None,
                                      eps: torch.Tensor | float | None = None,
                                      use_kv_cache: bool = False) -> dict:
        """Core hindsight lookahead pipeline (shared between policy training and diagnostics/theater):
        1. Baseline rollout of base_action -> (c_pi, eos_pi, steps_pi)
        2. Multiplicative forcing on the deviation from neutral -> eos_forced
        3. Hindsight action generation under forced EOS -> (gen_toks, gen_action)
        4. Causal verification rollout -> (c_gen, eos_gen, steps_gen)
        """
        B = ctx.shape[0]
        device = ctx.device

        # 1. baseline rollout -> accumulated cost C_pi + imagined terminal EOS
        c_pi, eos_pi, steps_pi = self._rollout_candidate(
            ctx, base_cache, base_action, s_dec_toks, n_imagine,
            ctx_len, ctx_mask, use_kv_cache)

        # 2. multiplicative forcing on the deviation from neutral: make the
        #    predicted denouement deviation eps-proportionally more favorable
        #    (worsening shrinks toward 0.5, recovery deepens below it) — the
        #    request scales with what the baseline itself predicts, at any
        #    delta magnitude (automatic curriculum)
        if eps is None:
            lo, hi = _EOS_FORCE_EPS
            eps_val = torch.rand(B, _EOS_NSLOTS, device=device) * (hi - lo) + lo
        elif isinstance(eps, (float, int)):
            eps_val = torch.full((B, _EOS_NSLOTS), float(eps), device=device)
        else:
            eps_val = eps
        # Neutrality floor: in case of future pain/worsening, clamp at least to neutral (0.5),
        # then push further into recovery by eps.
        # Slots 0..4 (costs): lower is better; neutral is 0.5.
        # If eos_pi > 0.5 (worsening), cap at 0.5 then subtract eps.
        # If eos_pi <= 0.5 (already recovery), subtract eps to deepen recovery.
        eos_forced = eos_pi.clone()
        eos_base_c = torch.minimum(eos_pi[:, :5], torch.tensor(0.5, device=device))
        eos_forced[:, :5] = (eos_base_c - eps_val[:, :5]).clamp(0.0, 1.0)

        # Slot 5 (reward/comfort): higher is better; neutral is 0.5.
        # If eos_pi < 0.5 (drop in comfort), floor at 0.5 then add eps.
        # If eos_pi >= 0.5 (already comfort gain), add eps to increase gain.
        eos_base_r = torch.maximum(eos_pi[:, 5], torch.tensor(0.5, device=device))
        eos_forced[:, 5] = (eos_base_r + eps_val[:, 5]).clamp(0.0, 1.0)

        # 3. hindsight generation of action under the forced EOS
        with torch.inference_mode():
            gen_toks = self._lookahead_generate(ctx, base_cache, eos_forced, use_kv_cache)
            gen_action = self._decode_action_batch(gen_toks)        # [B, 5]

        # 4. causal verification rollout (causality, reachability and graceful
        #    overshoot discovery in one test)
        c_gen, eos_gen, steps_gen = self._rollout_candidate(
            ctx, base_cache, gen_action, s_dec_toks, n_imagine,
            ctx_len, ctx_mask, use_kv_cache)

        return {
            "c_pi": c_pi,
            "eos_pi": eos_pi,
            "steps_pi": steps_pi,
            "eos_forced": eos_forced,
            "gen_toks": gen_toks,
            "gen_action": gen_action,
            "c_gen": c_gen,
            "eos_gen": eos_gen,
            "steps_gen": steps_gen,
        }

    def rollout_lookahead_from_sequence(self, seq: list[list[list[float]]],
                                        eps: float | None = None) -> dict:
        """Roll out counterfactual hindsight trajectory from sequence context s0..s4:
        1. Baseline rollout with recorded a4 -> eos_pi
        2. Reinforced EOS -> eos_better
        3. Hindsight action a4* generated by World Model conditioned on [ctx, eos_better]
        4. Verification rollout of a4* and closed-loop policy reactions -> steps_gen
        """
        device = self.device
        vals = self._batch_tensors([seq])
        n_ctx = self.n_ctx
        ctx_len = n_ctx * SALVE_TOKENS + STATE_TOKENS
        ctx = vals[:, :self.decision_pos + 1, :].clone()
        s_dec_toks = ctx[0, n_ctx * SALVE_TOKENS: n_ctx * SALVE_TOKENS + STATE_TOKENS].tolist()
        ctx_mask = build_salve_mask(ctx_len, device)

        rec_a_toks = torch.tensor(seq[n_ctx][STATE_TOKENS:SALVE_TOKENS],
                                  dtype=torch.float32, device=device).unsqueeze(0)
        recorded_a4 = self._decode_action_batch(rec_a_toks)

        if eps is None:
            eps = (_EOS_FORCE_EPS[0] + _EOS_FORCE_EPS[1]) / 2.0

        hl = self._hindsight_lookahead_pipeline(
            ctx, recorded_a4, s_dec_toks, self.n_imagine,
            ctx_len, ctx_mask, base_cache=None,
            eps=eps, use_kv_cache=False)
        hl["recorded_a4"] = recorded_a4
        hl["eos_better"] = hl["eos_forced"]
        return hl

    # --- sleep: legacy policy training via candidate rollouts (fallback) ----
    def _train_policy_candidates(self, steps: int = 16, batch: int = 32,
                                 n_imagine: int | None = None,
                                 prioritize_relief: bool = True,
                                 use_kv_cache: bool = True,
                                 step_offset: int = 0):
        """Legacy stochastic-candidate machinery: 3 candidates (policy,
        recorded demonstration, Gaussian noise sigma 0.30) x 2 imagined
        regimes (Kalman inertia, policy closed-loop) under
        Bellman optimism — 6 rollouts per sample. Kept alive as the fallback
        of the EOS-lookahead supervision.
        Generator: yields (label, value) after each step."""
        if n_imagine is None:
            n_imagine = self.n_imagine
        n_ctx = self.n_ctx
        losses = []
        for step in range(steps):
            sequences = (self.buffer.sample_for_policy(batch)
                         if prioritize_relief else self.buffer.sample(batch))
            if not sequences:
                break

            B = len(sequences)
            device = self.device

            # build real context: [s0, a0, s1, ..., a_{n_ctx-1}, s_{n_ctx}]
            ctx_len = n_ctx * SALVE_TOKENS + STATE_TOKENS
            vals = self._batch_tensors(sequences)
            ctx = vals[:, :ctx_len, :].clone()
            real_actions = self._decode_action_batch(
                vals[:, n_ctx * SALVE_TOKENS + STATE_TOKENS : (n_ctx + 1) * SALVE_TOKENS, :])

            # 1. world model forward on real context → intermediate latent
            ctx_mask = build_salve_mask(ctx_len, device)
            base_cache = None
            with torch.inference_mode():
                if use_kv_cache:
                    base_cache = KVCache(len(self.world.transformer.layers),
                                         self.world.d_model, B, device)
                    self.world.cache_forward(ctx, base_cache)
                else:
                    self.world(ctx, attn_mask=ctx_mask)
                latent = self.latent_norm(self.world.last_latent().detach())

            # 2. policy input
            cur_scalars = self._policy_scalars_batch(ctx[:, -STATE_TOKENS:])

            pol_in = torch.cat([cur_scalars, latent], dim=1)

            # 3. generate 3 candidate actions:
            #    candidate 0 = deterministic action from current policy
            #    candidate 1 = recorded demonstrated action from dataset
            #    candidate 2 = exploration noise around policy action (sigma = 0.30)
            with torch.inference_mode():
                policy_action = self.policy(pol_in)
                cand_noise = self._explore_action_batch(policy_action, sigma=0.30)

            candidates = torch.stack(
                [
                    policy_action,
                    real_actions,
                    cand_noise,
                ],
                dim=1,
            )  # [B, 3, 5]

            # Decode past actions from the context for trajectory extrapolation:
            p_a_prev2 = (n_ctx - 2) * SALVE_TOKENS + STATE_TOKENS
            p_a_prev1 = (n_ctx - 1) * SALVE_TOKENS + STATE_TOKENS
            past_a2 = self._decode_action_batch(ctx[:, p_a_prev2 : p_a_prev2 + ACTION_TOKENS, :])
            past_a3 = self._decode_action_batch(ctx[:, p_a_prev1 : p_a_prev1 + ACTION_TOKENS, :])

            # 4. evaluate every candidate with 2 futures under Bellman optimism
            candidate_costs = []
            dream_cand_trajs = [{} for _ in range(3)]
            s_dec_toks = ctx[0, n_ctx * SALVE_TOKENS : n_ctx * SALVE_TOKENS + STATE_TOKENS].tolist()

            for candidate_idx in range(3):
                cand_cost = self._imagine_candidate(
                    ctx, base_cache, candidates[:, candidate_idx, :],
                    past_a2, past_a3, s_dec_toks, n_imagine,
                    ctx_len, ctx_mask, use_kv_cache,
                    dream_cand_trajs, candidate_idx)
                candidate_costs.append(cand_cost)

            # [B, 3]
            candidate_costs = torch.stack(candidate_costs, dim=1)

            # 5. Select the best action
            cost_baseline = candidate_costs[:, 0:1]  # [B, 1]
            margin = _EXPLORE_MARGIN_PCT * cost_baseline.abs().clamp(min=1.0)  # [B, 1]
            adjusted_costs = candidate_costs.clone()
            adjusted_costs[:, 1:] += margin
            best_idx = adjusted_costs.argmin(dim=1)
            batch_idx = torch.arange(B, device=device)
            best_action = candidates[batch_idx, best_idx].detach()  # [B, 5]

            # 6. Supervised learning
            #    The action selected by the WM becomes the target.
            #    Re-run the policy without no_grad so that loss propagates.
            pred_action = self.policy(pol_in)
            loss = torch.nn.functional.mse_loss(pred_action, best_action)

            self.opt_pol.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt_pol.step()

            # Build DreamRecord for visualization (sample 0)
            if B > 0 and n_imagine > 1:
                context_steps = []
                for s_i in range(n_ctx):
                    s_tokens = ctx[0, s_i * 16 : s_i * 16 + 13].tolist()
                    a_vals = self._decode_action_batch(ctx[0:1, s_i * 16 + 13 : (s_i + 1) * 16])[0].tolist()
                    context_steps.append(DreamStep(
                        state_tokens=s_tokens,
                        action=a_vals,
                        label=f"s{s_i}+a{s_i}"
                    ))
                context_steps.append(DreamStep(
                    state_tokens=s_dec_toks,
                    action=None,
                    label=f"s{n_ctx} (decision)"
                ))

                all_trajs = []
                best_futures_per_cand = []
                win_cand = int(best_idx[0].item())
                for c_idx in range(3):
                    c_costs = [dream_cand_trajs[c_idx][f]["cost"] for f in range(2)]
                    b_f_idx = int(torch.tensor(c_costs).argmin().item())
                    best_futures_per_cand.append(b_f_idx)
                    for f_idx in range(2):
                        traj_info = dream_cand_trajs[c_idx][f_idx]
                        is_b = (f_idx == b_f_idx)
                        is_w = (c_idx == win_cand and is_b)
                        all_trajs.append(DreamTrajectory(
                            candidate_idx=c_idx,
                            regime_idx=f_idx,
                            regime_name=REGIME_NAMES[f_idx],
                            steps=traj_info["steps"],
                            total_cost=traj_info["cost"],
                            is_best_future=is_b,
                            is_winner=is_w,
                        ))

                self.last_dream_record = DreamRecord(
                    step_idx=step + step_offset,
                    total_steps=step_offset + steps,
                    loss=loss.item(),
                    facing=1,
                    context_steps=context_steps,
                    candidate_actions=[candidates[0, c].tolist() for c in range(3)],
                    trajectories=all_trajs,
                    best_candidate_idx=win_cand,
                    best_future_indices=best_futures_per_cand,
                )

            losses.append(loss.item())
            yield "pol", loss.item()

    # --- sleep entry point ---------------------------------------------------
    def sleep(self, wm_epochs: int = 128, wm_batch: int = 64,
              pol_steps: int = 64, pol_batch: int = 32,
              pol_n_imagine: int | None = None, clear_buffer: bool = True,
              prune_pct: float = 0.33, surprise_factor: float = 1.0,
              wm_target_loss: float | None = 0.01,
              wm_coreset_target_loss: float | None = None,
              use_kv_cache: bool = True,
              pol_eos_lookahead: bool = True):
        """One full sleep cycle with Coreset / Addendum and active forgetting.

        Phases:
          1. Consolidate wake session journal into Addendum (added_wake).
          2. Prune ~33% of Coreset into Addendum as candidates for active forgetting.
          3. Phase 1: Train World Model on the pruned Coreset (yielding 'wm_coreset')
             until quota is validated AND loss <= wm_target_loss. Causal passes
             alternate with EOS-lookahead passes (1 in _WM_LOOKAHEAD_PERIOD):
             the decision position also attends the realized terminal EOS, so
             the WM learns the hindsight action.
          4. Filter Addendum: evaluate surprise under updated WM (dynamics + cost divergence).
             Discard familiar sequences; retain surprising sequences.
          5. Phase 2: Train World Model on the surviving Addendum (yielding 'wm_addendum')
             until quota is validated AND loss <= wm_target_loss (same alternation).
          6. Policy training via WM hindsight supervision (yielding 'pol').
          7. Consolidation: commit Addendum into Coreset, refresh wake latent, yield 'done'.
        """
        if pol_n_imagine is None:
            pol_n_imagine = self.n_imagine
        if wm_coreset_target_loss is not None:
            wm_target_loss = wm_coreset_target_loss
        self.mode = "sleep"
        self.last_dream_record = None

        # Coreset avant sommeil
        coreset_before = self.buffer.coreset_size

        # 1. Extraction des nouvelles expériences de la session de veille
        n_wake = self.buffer.extract_addendum()

        # 2. Souvenirs remis en jeu (oubli actif : 33% du Coreset)
        n_pruned = 0
        if self.buffer.coreset_size > 1 and prune_pct > 0.0:
            n_pruned = self.buffer.prune_coreset(prune_pct)

        addendum_initial = self.buffer.addendum_size

        # 3. Déduplication géométrique :
        # 3a. Intra-Addendum
        saliency_weights = self.trusted_saliency_weights()
        n_intra_dropped = self.buffer.deduplicate_addendum(eps=0.04, weights=saliency_weights)
        if n_intra_dropped > 0:
            print(f"[sleep:intra-addendum] {n_intra_dropped} redundant duplicates eliminated from Addendum.")

        # 3b. Addendum vs Coreset
        cross_res = self.buffer.deduplicate_against_coreset(eps=0.04, weights=saliency_weights)
        n_cross_dropped = cross_res["dropped"]
        n_cross_refreshed = cross_res["refreshed"]
        if n_cross_dropped > 0:
            print(f"[sleep:cross-coreset] {n_cross_dropped} Coreset duplicates eliminated from Addendum ({n_cross_refreshed} refreshed).")

        self.sleep_cycle_stats = {
            "coreset_before": coreset_before,
            "wake": n_wake,
            "pruned": n_pruned,
            "addendum_initial": addendum_initial,
            "dedup_intra": n_intra_dropped,
            "dedup_coreset": n_cross_dropped,
            "refreshed": n_cross_refreshed,
            "filter_dropped": None,
            "filter_kept": None,
            "coreset_after": None,
        }

        yield "prune", {
            "pruned": n_pruned,
            "coreset": self.buffer.coreset_size,
            "addendum": self.buffer.addendum_size,
            "coreset_before": coreset_before,
            "wake": n_wake,
            "dedup_intra": n_intra_dropped,
            "dedup_coreset": n_cross_dropped,
            "refreshed": n_cross_refreshed,
        }

        # 3. Epoch allocation between Coreset and Addendum
        has_coreset = (self.buffer.coreset_size > 0)
        has_addendum = (self.buffer.addendum_size > 0)

        if not has_coreset and not has_addendum:
            self.mode = "wake"
            yield "done", {"wm_loss": float("nan"), "pol_loss": float("nan")}
            return

        # 3. Target epoch quota for Coreset and Addendum
        epochs_core = wm_epochs if has_coreset else 0
        epochs_add = wm_epochs if has_addendum else 0

        # 4. Phase 1: Train World Model on Coreset
        core_losses = []
        lookahead_losses = []
        if epochs_core > 0 or (has_coreset and wm_target_loss is not None):
            self.world.train()
            self.sleep_wm_epochs = epochs_core
            ep = 0
            while True:
                ep += 1
                self.sleep_wm_step = ep
                if ep > self.sleep_wm_epochs:
                    self.sleep_wm_epochs = ep
                seqs = self.buffer.sample_coreset(wm_batch)
                if not seqs:
                    break
                # Alternated mask: 1 EOS-lookahead pass every _WM_LOOKAHEAD_PERIOD
                lookahead = (ep % _WM_LOOKAHEAD_PERIOD == 0)
                loss_val = self._train_wm_batch(seqs, lookahead=lookahead)
                core_losses.append(loss_val)
                if lookahead:
                    lookahead_losses.append(loss_val)
                yield "wm_coreset", loss_val

                # Stopping criterion: must have completed the quota AND reached target loss
                quota_done = (ep >= epochs_core)
                target_done = (wm_target_loss is None) or (loss_val <= wm_target_loss)
                if quota_done and target_done:
                    if ep > epochs_core:
                        print(f"[sleep:wm_coreset] Target loss {wm_target_loss:.4f} reached in overtime at epoch {ep} (loss={loss_val:.5f}).")
                    break
        recent_losses = core_losses[-min(len(core_losses), 5):]
        self.last_wm_coreset_loss = (sum(recent_losses) / len(recent_losses)) if recent_losses else float("nan")

        # 5. Filter Addendum using the updated WM
        n_initial = self.buffer.addendum_size
        n_kept = n_initial
        n_dropped = 0
        wake_kept = 0
        wake_dropped = 0
        core_kept = 0
        core_dropped = 0
        mean_kept = 0.0
        mean_drop = 0.0
        pct_drop = 0.0
        threshold = 0.0

        if self.buffer.addendum_size > 0:
            if has_coreset and not math.isnan(self.last_wm_coreset_loss):
                threshold = max(self.last_wm_coreset_loss * surprise_factor, 1e-4)
                surprises = self.evaluate_sequences_surprise(self.buffer._addendum)
                surviving = []
                surviving_meta = []
                kept_surprises = []
                dropped_surprises = []

                for i, (seq, s) in enumerate(zip(self.buffer._addendum, surprises)):
                    origin = self.buffer._addendum_meta[i] if i < len(self.buffer._addendum_meta) else "wake"
                    if s >= threshold:
                        surviving.append(seq)
                        surviving_meta.append(origin)
                        kept_surprises.append(s)
                        if origin == "coreset":
                            core_kept += 1
                        else:
                            wake_kept += 1
                    else:
                        dropped_surprises.append(s)
                        if origin == "coreset":
                            core_dropped += 1
                        else:
                            wake_dropped += 1

                n_kept = len(surviving)
                n_dropped = n_initial - n_kept
                pct_drop = (n_dropped / n_initial * 100.0) if n_initial > 0 else 0.0
                mean_kept = (sum(kept_surprises) / len(kept_surprises)) if kept_surprises else 0.0
                mean_drop = (sum(dropped_surprises) / len(dropped_surprises)) if dropped_surprises else 0.0

                self.buffer.filter_addendum(surviving, surviving_meta)

                # Console log (Option 2)
                print("\n" + "=" * 62)
                print(f"[sleep:filtrage addendum] Seuil de surprise: {threshold:.5f}")
                print(f"  Total addendum: {n_initial} | Retirées: {n_dropped} ({pct_drop:.1f}%) | Retenues: {n_kept}")
                if n_pruned > 0:
                    print(f"  - 33% Coreset (oubli actif) : {core_dropped}/{n_pruned} oubliées ({core_kept} réinjectées)")
                    print(f"  - Veille brute              : {wake_dropped}/{wake_dropped + wake_kept} éliminées ({wake_kept} retenues)")
                print(f"  - Surprise moyenne : rejetées={mean_drop:.5f} | retenues={mean_kept:.5f}")
                print("=" * 62 + "\n")
            else:
                # Cold start: keep everything in Addendum
                n_kept = n_initial
                n_dropped = 0
                pct_drop = 0.0
                print(f"[sleep:addendum] Démarrage initial: {n_initial} séquences conservées (pas de filtrage initial).")

        self.last_filter_stats = {
            "initial": n_initial,
            "kept": n_kept,
            "dropped": n_dropped,
            "pct_dropped": pct_drop,
            "wake_kept": wake_kept,
            "wake_dropped": wake_dropped,
            "core_kept": core_kept,
            "core_dropped": core_dropped,
            "mean_kept_surprise": mean_kept,
            "mean_dropped_surprise": mean_drop,
            "threshold": threshold,
        }
        if hasattr(self, "sleep_cycle_stats"):
            self.sleep_cycle_stats["filter_dropped"] = n_dropped
            self.sleep_cycle_stats["filter_kept"] = n_kept
            expected_final = min(self.buffer.pool_capacity, self.buffer.coreset_size + n_kept)
            self.sleep_cycle_stats["coreset_after"] = expected_final
        yield "filter", self.last_filter_stats

        # 6. Phase 2: Train World Model on surviving Addendum
        add_losses = []
        if self.buffer.addendum_size > 0 and (epochs_add > 0 or wm_target_loss is not None):
            self.world.train()
            self.sleep_wm_epochs = epochs_add
            ep = 0
            while True:
                ep += 1
                self.sleep_wm_step = ep
                if ep > self.sleep_wm_epochs:
                    self.sleep_wm_epochs = ep
                seqs = self.buffer.sample_addendum(wm_batch)
                if not seqs:
                    break
                lookahead = (ep % _WM_LOOKAHEAD_PERIOD == 0)
                loss_val = self._train_wm_batch(seqs, lookahead=lookahead)
                add_losses.append(loss_val)
                if lookahead:
                    lookahead_losses.append(loss_val)
                yield "wm_addendum", loss_val

                # Stopping criterion: must have completed the quota AND reached target loss
                quota_done = (ep >= epochs_add)
                target_done = (wm_target_loss is None) or (loss_val <= wm_target_loss)
                if quota_done and target_done:
                    if ep > epochs_add:
                        print(f"[sleep:wm_addendum] Target loss {wm_target_loss:.4f} reached in overtime at epoch {ep} (loss={loss_val:.5f}).")
                    break
        recent_add_losses = add_losses[-min(len(add_losses), 5):]
        self.last_wm_addendum_loss = (sum(recent_add_losses) / len(recent_add_losses)) if recent_add_losses else float("nan")

        all_wm_losses = core_losses + add_losses
        self.last_wm_loss = (sum(all_wm_losses) / len(all_wm_losses)) if all_wm_losses else float("nan")
        self.last_wm_lookahead_loss = (sum(lookahead_losses) / len(lookahead_losses)) \
            if lookahead_losses else float("nan")

        # 7. Train Policy (hindsight supervision, candidate fallback)
        for label, value in self.train_policy(pol_steps, pol_batch,
                                              n_imagine=pol_n_imagine,
                                              use_kv_cache=use_kv_cache,
                                              eos_lookahead=pol_eos_lookahead):
            yield label, value

        # 8. Consolidate surviving Addendum into Coreset with memory decay.
        saliency_weights = self.trusted_saliency_weights()
        cons_stats = self.buffer.consolidate_addendum_into_coreset(
            eps=0.04, decay=0.01, dedup_intra=False, weights=saliency_weights
        )
        cons_stats["refreshed"] = n_cross_refreshed + cons_stats.get("refreshed", 0)
        cons_stats["dedup_cross_dropped"] = n_cross_dropped
        cons_stats["dedup_intra_dropped"] = n_intra_dropped
        if hasattr(self, "sleep_cycle_stats"):
            self.sleep_cycle_stats["coreset_after"] = self.buffer.coreset_size
            self.sleep_cycle_stats["added"] = cons_stats["added"]
            self.sleep_cycle_stats["evicted"] = cons_stats["evicted"]

        self._refresh_wake_latent()
        self.mode = "wake"

        print("\n" + "=" * 62)
        print("[sleep:cycle summary] :")
        print(f"  - Coreset before sleep          : {coreset_before}")
        print(f"  - New experiences               : +{n_wake}")
        print(f"  - Memories revisited (pruned)   : +{n_pruned}")
        print(f"  - Raw addendum                  : {addendum_initial}")
        print(f"  - Intra-addendum duplicates     : -{n_intra_dropped}")
        print(f"  - Addendum/coreset duplicates   : -{n_cross_dropped} ({n_cross_refreshed} refreshed)")
        print(f"  - Familiar sequences dropped    : -{n_dropped}")
        if not math.isnan(self.last_wm_lookahead_loss):
            print(f"  - WM lookahead loss (EOS-cond.): {self.last_wm_lookahead_loss:.5f}")
        print(f"  - Coreset after sleep           : {self.buffer.coreset_size} (+{cons_stats['added']} added, -{cons_stats['evicted']} evicted)")
        print("=" * 62 + "\n")

        stats = {
            "wm_loss": self.last_wm_loss,
            "wm_coreset_loss": self.last_wm_coreset_loss,
            "wm_addendum_loss": self.last_wm_addendum_loss,
            "wm_lookahead_loss": self.last_wm_lookahead_loss,
            "filter": self.last_filter_stats,
            "consolidation": cons_stats,
            "cycle": getattr(self, "sleep_cycle_stats", {}),
            "pol_loss": self.last_pol_loss,
            "coreset_size": self.buffer.coreset_size,
        }
        yield "done", stats

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

    def save_checkpoint(self, wm_path: str, pol_path: str, buf_path: str | None = None) -> None:
        """Save world model, policy, and optionally the experience buffer to disk."""
        torch.save({
            "world": self.world.state_dict(),
            "opt_wm": self.opt_wm.state_dict(),
        }, wm_path)
        torch.save({
            "policy": self.policy.state_dict(),
            "latent_norm": self.latent_norm.state_dict(),
            "opt_pol": self.opt_pol.state_dict(),
        }, pol_path)
        if buf_path:
            self.buffer.save(buf_path)

    def load_checkpoint(self, wm_path: str, pol_path: str, buf_path: str | None = None) -> None:
        """Load world model, policy, and optionally the experience buffer from disk.
        Each is loaded independently — a mismatch in one does not affect the others."""
        if os.path.exists(wm_path):
            sd = torch.load(wm_path, map_location=self.device, weights_only=True)
            if "world" in sd:
                self.world.load_state_dict(sd["world"])
            if "opt_wm" in sd:
                self.opt_wm.load_state_dict(sd["opt_wm"])
        if os.path.exists(pol_path):
            sd = torch.load(pol_path, map_location=self.device, weights_only=True)
            if "policy" in sd:
                p_sd = sd["policy"]
                if "net.0.weight" in p_sd:
                    cur_in = self.policy.net[0].in_features
                    ckpt_in = p_sd["net.0.weight"].shape[1]
                    if ckpt_in == cur_in + 2:
                        # Adapt from 111 (45 phys + 2 intero + 64 latent) to 109 (45 phys + 64 latent)
                        w = p_sd["net.0.weight"]
                        p_sd["net.0.weight"] = torch.cat([w[:, :45], w[:, 47:]], dim=1)
                try:
                    self.policy.load_state_dict(p_sd)
                except Exception as e:
                    print(f"[warning] Policy load: {e}")
            if "latent_norm" in sd:
                try:
                    self.latent_norm.load_state_dict(sd["latent_norm"])
                except Exception:
                    pass
            if "opt_pol" in sd:
                try:
                    self.opt_pol.load_state_dict(sd["opt_pol"])
                except Exception:
                    pass
        if buf_path and os.path.exists(buf_path):
            self.buffer.load(buf_path)
            if self.buffer._coreset and len(self.buffer._coreset[0]) != self.buffer.seq_len:
                print(f"[warning] Checkpoint buffer sequences have length {len(self.buffer._coreset[0])}, but brain requires {self.buffer.seq_len} salves ({self.seq_steps} steps). Discarding mismatched checkpoint buffer.")
                self.buffer.clear()
