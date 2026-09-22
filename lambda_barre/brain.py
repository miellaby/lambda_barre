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
from .models import build_salve_mask
from .tokenize import (STATE_TOKENS, ACTION_TOKENS, SALVE_TOKENS, DENSE_DIM,
                       N_SIGNAL, _N_SIGNALS, N_POLICY_STATE, SIG_OFFSET,
                       _COST_IDX, _REWARD_IDX, _INTERO_IDX, _COST_KEYS, _COST_WEIGHTS,
                       _BY_KEY, _LAYOUT, _prefix,
                       state_tokens, action_tokens)


# Fixed layout of a training sequence: a trajectory of SEQ_STEPS transitions.
# [s0, a0, s1, a1, ..., a_{N-1}, sN] = (state + action) * N + state tokens.
SEQ_STEPS = 10
_SEQ_LEN = SEQ_STEPS * SALVE_TOKENS + STATE_TOKENS          # 173
_N_CTX = 4                                                 # 4 real transitions used as context (s0..s3, a0..a3, s4)
_EXPLORE_MARGIN_PCT = 0.001                                # 0.1% cost reduction margin required for alternative candidates

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

# Per-target-position valid-slot mask: position p predicts token p+1, whose
# token index within a salve is (p+1) % 16 — only its first ``_N_SIGNALS[tidx]``
# signal slots are real (the rest are zero-padded and excluded from the loss).
# For Token 12 (interoception delta): intermediate salves (0..SEQ_STEPS-1) are
# zero-padded and excluded from the loss; only the terminal salve
# (position _SEQ_LEN - 2, predicting the final token 172) is trained.
_target_valid = torch.zeros(_SEQ_LEN - 1, N_SIGNAL, dtype=torch.bool)
for _p in range(_SEQ_LEN - 1):
    _tidx = (_p + 1) % SALVE_TOKENS
    if _tidx == _INTERO_IDX and (_p + 1) < (_SEQ_LEN - 1):
        continue
    _target_valid[_p, :_N_SIGNALS[_tidx]] = True

# Salve-within-type attention mask for the fixed training sequence.
_MASK = build_salve_mask(_SEQ_LEN, torch.device("cpu"))

# Channels tracked for static detection:
#   - animal: proprioception (tokens 0..3) and touch (tokens 8..9)
#   - environnement: vision (tokens 4, 6), cursor/sound (tokens 5, 7), touch (tokens 8..9)
#   - consignes: action (tokens 13..15)
# Excludes reward/cost accumulators (tokens 10, 11) and interoception (token 12).
_STATIC_SOURCES = {"proprio", "touch", "cursor", "vision", "action"}
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
                                device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Pairwise naive RMSE distance matrix [N, N] across all channels and tokens."""
    X = _sequence_signals_tensor(sequences, device)
    N, D = X.shape
    if N <= 1 or D == 0:
        return torch.zeros(N, N, device=device)
    return torch.cdist(X, X) / (D ** 0.5)


def cross_sequence_distances(seqs_a: list[list[list[float]]],
                             seqs_b: list[list[list[float]]],
                             device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Cross naive RMSE distance matrix [N_a, N_b] across all channels and tokens."""
    Xa = _sequence_signals_tensor(seqs_a, device)
    Xb = _sequence_signals_tensor(seqs_b, device)
    if Xa.shape[0] == 0 or Xb.shape[0] == 0 or Xa.shape[1] == 0:
        return torch.empty(Xa.shape[0], Xb.shape[0], device=device)
    D = Xa.shape[1]
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

    def __init__(self, seq_len: int = SEQ_STEPS + 1, capacity: int = 1000,
                 pool_capacity: int = 1000, seed: int = 0):
        self.seq_len = seq_len
        self.capacity = capacity
        self.pool_capacity = pool_capacity
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

    def prune_coreset(self, pct: float = 0.33) -> int:
        """Move pct (default 33%) of sequences from Coreset into Addendum.
        These become candidates for active forgetting.
        Sequences with lower vivacity are selected in priority.
        Returns the number of sequences moved.
        """
        if len(self._coreset) <= 1 or pct <= 0.0:
            return 0
        n_prune = max(1, int(len(self._coreset) * pct))
        n_prune = min(n_prune, len(self._coreset) - 1)
        while len(self._coreset_vivacity) < len(self._coreset):
            self._coreset_vivacity.append(1.0)
        # Order by vivacity ascending with random tie-breaker: coldest memories pruned first
        order = sorted(range(len(self._coreset)), key=lambda i: (self._coreset_vivacity[i], self._rng.random()))
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

    def deduplicate_addendum(self, eps: float = 0.04) -> int:
        """Intra-Addendum deduplication via complete linkage clustering on naive RMSE distances.
        Groups sequences at distance <= eps across all channels and tokens, and keeps only the medoid of each group.
        Returns the number of duplicate sequences removed."""
        if len(self._addendum) <= 1:
            return 0
        D = pairwise_sequence_distances(self._addendum)
        clusters = complete_linkage_clustering(D, eps=eps)
        medoid_indices = select_medoids(clusters, D)
        n_dropped = len(self._addendum) - len(medoid_indices)
        if n_dropped > 0:
            self._addendum = [self._addendum[i] for i in medoid_indices]
            if len(self._addendum_meta) >= len(medoid_indices):
                self._addendum_meta = [self._addendum_meta[i] for i in medoid_indices]
        return n_dropped

    def consolidate_addendum_into_coreset(self, eps: float = 0.04, decay: float = 0.01,
                                          dedup_intra: bool = True) -> dict:
        """Cross-deduplicate Addendum against Coreset and update memory vivacity traces.

        1. If dedup_intra is True, first removes redundant duplicates within Addendum.
        2. Compares each surviving Addendum sequence against Coreset with naive distance.
           - If dist < eps: Duplicate of existing memory -> refreshes Coreset memory vivacity to 1.0.
           - If dist >= eps: Novel memory -> added to Coreset with vivacity 1.0.
        3. Applies memory decay (-decay, default -0.01) across all Coreset vivacities.
        4. Evicts lowest-vivacity memories if Coreset exceeds pool_capacity.
        5. Clears Addendum.
        """
        n_intra_dropped = 0
        if dedup_intra and len(self._addendum) > 1:
            n_intra_dropped = self.deduplicate_addendum(eps=eps)

        n_refreshed = 0
        n_added = 0

        while len(self._coreset_vivacity) < len(self._coreset):
            self._coreset_vivacity.append(1.0)

        if not self._coreset:
            for seq in self._addendum:
                self._coreset.append(seq)
                self._coreset_vivacity.append(1.0)
                n_added += 1
        elif self._addendum:
            cross_D = cross_sequence_distances(self._addendum, self._coreset)
            min_dists, min_indices = torch.min(cross_D, dim=1)

            for i, seq in enumerate(self._addendum):
                d_min = min_dists[i].item()
                idx_core = min_indices[i].item()
                if d_min < eps:
                    self._coreset_vivacity[idx_core] = 1.0
                    n_refreshed += 1
                else:
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

    def commit_addendum(self, dedup: bool = False, eps: float = 0.04, decay: float = 0.01) -> int:
        """Merge all surviving Addendum sequences into Coreset, then clear Addendum.
        If dedup=True, uses consolidate_addendum_into_coreset; otherwise uses direct extend.
        Returns the number of sequences committed.
        """
        if dedup:
            stats = self.consolidate_addendum_into_coreset(eps=eps, decay=decay)
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


class Brain:
    """Holds both networks, drives the live loop, and runs sleep training.

    Lifecycle in the main loop:

        brain.act(salve_t)            -> (theta_l, d_l, theta_r, d_r, tail_t)
        ... physics steps, sensors ...
        brain.record(salve_t, salve_next)
        brain.sleep()                 # on demand: offline training of both models
    """

    def __init__(self, lr_wm: float = 3e-4, lr_pol: float = 2e-4, act_std: float = 0.05,
                 explore_std: float = 0.2, device: str | None = None, seed: int = 0):
        torch.manual_seed(seed)
        self.device, self.device_desc = M.configure_hardware(device)
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
        self.buffer = ExperienceBuffer(seq_len=SEQ_STEPS + 1, seed=seed)
        self.mask = _MASK.to(self.device)
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
        self.last_wm_coreset_loss = float("nan")
        self.last_wm_addendum_loss = float("nan")
        self.last_filter_stats: dict | None = None
        self.last_pol_loss = float("nan")
        self.last_dream_record: DreamRecord | None = None
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
    @torch.no_grad()
    def wake_tick(self, salve) -> None:
        """World model tick: run a forward pass on the rolling history +
        current salve to produce a fresh latent for the policy. Also journals
        the transition into the experience buffer."""
        self.mode = "wake"
        cur_state = state_tokens(salve)
        a_toks = action_tokens(salve)
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
        self._wake_history.append((cur_state, a_toks))
        if len(self._wake_history) > _N_CTX:
            self._wake_history.pop(0)

    @torch.no_grad()
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
        self._pol_time = (1 - self._time_alpha) * self._pol_time \
            + self._time_alpha * (time.perf_counter() - t0) * 1000
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
            pos += STATE_TOKENS
            ctx[0, pos:pos + ACTION_TOKENS] = torch.tensor(
                a_toks, dtype=torch.float32, device=self.device)
            pos += ACTION_TOKENS
        ctx[0, pos:pos + STATE_TOKENS] = torch.tensor(
            cur_state, dtype=torch.float32, device=self.device)
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
            pos += STATE_TOKENS
            ctx[0, pos:pos + ACTION_TOKENS] = torch.tensor(
                a_toks, dtype=torch.float32, device=self.device)
            pos += ACTION_TOKENS
        ctx[0, pos:pos + STATE_TOKENS] = torch.tensor(
            cur_state, dtype=torch.float32, device=self.device)
        self.world.eval()
        with torch.no_grad():
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
            assert pos == _SEQ_LEN, f"Expected pos={_SEQ_LEN}, got {pos}"

            # Zero-padding for intermediate Token 12 (interoception) in salves 0..SEQ_STEPS-1
            for k in range(SEQ_STEPS):
                p_intero = k * SALVE_TOKENS + _INTERO_IDX
                vals[b, p_intero, SIG_OFFSET:] = 0.0

            # Terminal Token 12 in salve SEQ_STEPS: compute delta between s_last and s_0
            p_final = SEQ_STEPS * SALVE_TOKENS + _INTERO_IDX
            s0_intero = seq[0][_INTERO_IDX]
            s_last_intero = seq[-1][_INTERO_IDX]
            f0, sf0 = s0_intero[SIG_OFFSET], s0_intero[SIG_OFFSET + 1]
            f_last, sf_last = s_last_intero[SIG_OFFSET], s_last_intero[SIG_OFFSET + 1]
            # Signed delta normalized to [0, 1]: (delta + 1.0) / 2.0
            d_fat = max(0.0, min(1.0, (f_last - f0 + 1.0) / 2.0))
            d_sf = max(0.0, min(1.0, (sf_last - sf0 + 1.0) / 2.0))
            vals[b, p_final, SIG_OFFSET] = d_fat
            vals[b, p_final, SIG_OFFSET + 1] = d_sf
            vals[b, p_final, SIG_OFFSET + 2:] = 0.0
        return vals

    def _train_wm_batch(self, sequences: list[list[list[float]]]) -> float:
        """Run one training optimization step on a batch of transition sequences."""
        vals = self._batch_tensors(sequences)             # [B, L, 25]
        pred = self.world(vals, attn_mask=self.mask)      # [B, L, 16]
        pred_signals = pred[:, :-1, :]                   # [B, L-1, 16]
        tgt_signals = vals[:, 1:, SIG_OFFSET:]           # [B, L-1, 16]
        m = self.target_valid                            # [L-1, 16]
        m_b = m.unsqueeze(0).expand_as(pred_signals)     # [B, L-1, 16]
        sq = (pred_signals - tgt_signals) ** 2
        loss = (sq * m_b).sum() / m_b.sum().clamp(min=1)
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
        with torch.no_grad():
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
                for k in range(1, SEQ_STEPS + 1):
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

                mean_cost_diff = cost_diff_sum / SEQ_STEPS
                seq_surprise = dyn_loss * (1.0 + mean_cost_diff)
                surprises.extend(seq_surprise.tolist())
        return surprises

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
    def train_policy(self, steps: int = 16, batch: int = 32,
                     gamma: float = 1.1, n_imagine: int = 6):
        """Policy training via candidate evaluation in WM imagination.
        Generator: yields (label, value) after each step."""
        if len(self.buffer) < 1:
            return
        self.world.eval()
        self.policy.train()
        n_ctx = _N_CTX  # real transitions used as context (s0..s3, a0..a3, s4)
        # n_imagine = 6 steps: 2.0s forward horizon at 3 Hz
        losses = []
        for step in range(steps):
            sequences = self.buffer.sample(batch)
            if not sequences:
                break
            B = len(sequences)
            device = self.device

            # build real context: [s0, a0, s1, ..., a3, s4]
            ctx_len = n_ctx * SALVE_TOKENS + STATE_TOKENS
            ctx = torch.zeros(B, ctx_len, DENSE_DIM, device=device)
            real_actions = torch.zeros(B, 5, device=device)
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

                # Demonstrated action taken at step n_ctx (tokens 13..15)
                a_toks = seq[n_ctx][STATE_TOKENS:SALVE_TOKENS]
                tf_norm = a_toks[0][SIG_OFFSET]
                df_norm = a_toks[0][SIG_OFFSET + 1]
                tb_norm = a_toks[1][SIG_OFFSET]
                db_norm = a_toks[1][SIG_OFFSET + 1]
                tq_norm = a_toks[2][SIG_OFFSET]
                real_actions[b] = torch.tensor([
                    2.0 * tf_norm - 1.0, df_norm,
                    2.0 * tb_norm - 1.0, db_norm,
                    2.0 * tq_norm - 1.0,
                ], device=device)

            # 1. world model forward on real context → intermediate latent
            ctx_mask = build_salve_mask(ctx_len, device)
            with torch.no_grad():
                self.world(ctx, attn_mask=ctx_mask)
                latent = self.world.last_latent().detach()
                latent = self.latent_norm(latent)

            # 2. policy input
            cur_scalars = self._policy_scalars_batch(ctx[:, -STATE_TOKENS:])

            pol_in = torch.cat([cur_scalars, latent], dim=1)

            # 3. generate 4 candidate actions:
            #    candidate 0 = deterministic action from current policy
            #    candidate 1 = recorded demonstrated action from dataset
            #    candidates 2..3 = growing-sigma exploration around policy action
            #                      (sigma1 = 0.15, sigma2 = 0.30)
            with torch.no_grad():
                policy_action = self.policy(pol_in)
                cand_noise1 = self._explore_action_batch(policy_action, sigma=0.15)
                cand_noise2 = self._explore_action_batch(policy_action, sigma=0.30)

            candidates = torch.stack(
                [
                    policy_action,
                    real_actions,
                    cand_noise1,
                    cand_noise2,
                ],
                dim=1,
            )  # [B, 4, 5]

            # Decode past actions a2 and a3 from the context for trajectory extrapolation:
            # salve 2 action is at tokens 45..47 (index 2 * 16 + 13)
            # salve 3 action is at tokens 61..63 (index 3 * 16 + 13)
            past_a2 = self._decode_action_batch(ctx[:, 45:48, :])
            past_a3 = self._decode_action_batch(ctx[:, 61:64, :])

            # 4. evaluate every candidate with 3 futures under Bellman optimism
            candidate_costs = []
            dream_cand_trajs = [{} for _ in range(4)]
            s4_toks = ctx[0, n_ctx * 16 : n_ctx * 16 + 13].tolist()

            for candidate_idx in range(4):
                action = candidates[:, candidate_idx, :]
                action_toks = self._encode_action_batch(action)

                # Step 0: candidate a4 acts on s4 -> predicts s5
                cand_ctx_0 = torch.cat([ctx, action_toks], dim=1)
                with torch.no_grad():
                    s5 = self.world.predict_next_state(cand_ctx_0)
                c0 = self._salve_cost_batch(s5)
                ctx_s5 = torch.cat([cand_ctx_0, s5], dim=1)

                if n_imagine <= 1:
                    candidate_costs.append(c0)
                    continue

                cand_a = action[0].tolist()
                c0_val = c0[0].item()
                s5_toks = s5[0].tolist()

                # Future 1: Kalman / inertia extrapolation from (a2, a3, a4)
                # Damped velocity clamped to physical limits to prevent quadrant-flipping rotations
                cur_ctx = ctx_s5
                v = 0.7 * (action - past_a3) + 0.3 * (past_a3 - past_a2)
                v = torch.clamp(v, -self._kalman_max_v, self._kalman_max_v)
                cur_a = action.clone()
                cost_kalman = c0.clone()
                cur_state_toks = s5_toks
                cur_cost_val = c0_val
                f1_steps = [DreamStep(state_tokens=s4_toks, action=cand_a, label="s4+cand")]
                for k in range(1, n_imagine):
                    cur_a = torch.clamp(cur_a + v, self._act_min, self._act_max)
                    v = v * 0.8
                    next_a_toks = self._encode_action_batch(cur_a)
                    cand_ctx = torch.cat([cur_ctx, next_a_toks], dim=1)
                    with torch.no_grad():
                        gen = self.world.predict_next_state(cand_ctx)
                    sc = self._salve_cost_batch(gen)
                    cost_kalman = cost_kalman + (gamma ** k) * sc
                    f1_steps.append(DreamStep(state_tokens=cur_state_toks, action=cur_a[0].tolist(), step_cost=cur_cost_val, label=f"s{4+k}"))
                    cur_ctx = torch.cat([cand_ctx, gen], dim=1)
                    cur_state_toks = gen[0].tolist()
                    cur_cost_val = sc[0].item()
                f1_steps.append(DreamStep(state_tokens=cur_state_toks, action=None, step_cost=cur_cost_val, label=f"s{4+n_imagine}"))
                dream_cand_trajs[candidate_idx][0] = {"steps": f1_steps, "cost": cost_kalman[0].item()}

                # Future 2: World Model autoregressive continuation
                cur_ctx = ctx_s5
                cost_wm = c0.clone()
                cur_state_toks = s5_toks
                cur_cost_val = c0_val
                f2_steps = [DreamStep(state_tokens=s4_toks, action=cand_a, label="s4+cand")]
                for k in range(1, n_imagine):
                    with torch.no_grad():
                        next_a_toks = self.world.predict_next_action(cur_ctx)
                        cand_ctx = torch.cat([cur_ctx, next_a_toks], dim=1)
                        gen = self.world.predict_next_state(cand_ctx)
                    sc = self._salve_cost_batch(gen)
                    cost_wm = cost_wm + (gamma ** k) * sc
                    a_dec = self._decode_action_batch(next_a_toks)[0].tolist()
                    f2_steps.append(DreamStep(state_tokens=cur_state_toks, action=a_dec, step_cost=cur_cost_val, label=f"s{4+k}"))
                    cur_ctx = torch.cat([cand_ctx, gen], dim=1)
                    cur_state_toks = gen[0].tolist()
                    cur_cost_val = sc[0].item()
                f2_steps.append(DreamStep(state_tokens=cur_state_toks, action=None, step_cost=cur_cost_val, label=f"s{4+n_imagine}"))
                dream_cand_trajs[candidate_idx][1] = {"steps": f2_steps, "cost": cost_wm[0].item()}

                # Future 3: Policy closed-loop reaction
                # Uses a sliding window of the last n_ctx transitions (77 tokens) with salve-within-type
                # attention mask, matching live wake_tick and avoiding out-of-distribution latents.
                cur_ctx = ctx_s5
                cur_state = s5
                cur_state_toks = s5_toks
                cur_cost_val = c0_val
                cost_pol = c0.clone()
                f3_steps = [DreamStep(state_tokens=s4_toks, action=cand_a, label="s4+cand")]
                for k in range(1, n_imagine):
                    with torch.no_grad():
                        slide_ctx = cur_ctx[:, -ctx_len:]
                        self.world(slide_ctx, attn_mask=ctx_mask)
                        next_latent = self.latent_norm(self.world.last_latent().detach())
                        next_scalars = self._policy_scalars_batch(cur_state)
                        next_pol_in = torch.cat([next_scalars, next_latent], dim=1)
                        next_action = self.policy(next_pol_in)
                        next_a_toks = self._encode_action_batch(next_action)
                        cand_ctx = torch.cat([cur_ctx, next_a_toks], dim=1)
                        gen = self.world.predict_next_state(cand_ctx)
                    sc = self._salve_cost_batch(gen)
                    cost_pol = cost_pol + (gamma ** k) * sc
                    f3_steps.append(DreamStep(state_tokens=cur_state_toks, action=next_action[0].tolist(), step_cost=cur_cost_val, label=f"s{4+k}"))
                    cur_ctx = torch.cat([cand_ctx, gen], dim=1)
                    cur_state = gen
                    cur_state_toks = gen[0].tolist()
                    cur_cost_val = sc[0].item()
                f3_steps.append(DreamStep(state_tokens=cur_state_toks, action=None, step_cost=cur_cost_val, label=f"s{4+n_imagine}"))
                dream_cand_trajs[candidate_idx][2] = {"steps": f3_steps, "cost": cost_pol[0].item()}

                # Bellman optimism: optimistic minimum across the 3 futures
                cand_cost = torch.minimum(torch.minimum(cost_kalman, cost_wm), cost_pol)
                candidate_costs.append(cand_cost)

            # [B, 4]
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
                    state_tokens=s4_toks,
                    action=None,
                    label="s4 (decision)"
                ))

                all_trajs = []
                best_futures_per_cand = []
                win_cand = int(best_idx[0].item())
                for c_idx in range(4):
                    c_costs = [dream_cand_trajs[c_idx][f]["cost"] for f in range(3)]
                    b_f_idx = int(torch.tensor(c_costs).argmin().item())
                    best_futures_per_cand.append(b_f_idx)
                    for f_idx in range(3):
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
                    step_idx=step,
                    total_steps=steps,
                    loss=loss.item(),
                    facing=1,
                    context_steps=context_steps,
                    candidate_actions=[candidates[0, c].tolist() for c in range(4)],
                    trajectories=all_trajs,
                    best_candidate_idx=win_cand,
                    best_future_indices=best_futures_per_cand,
                )

            losses.append(loss.item())
            yield "pol", loss.item()
        self.last_pol_loss = sum(losses) / len(losses) if losses else float("nan")

    # --- sleep entry point ---------------------------------------------------
    def sleep(self, wm_epochs: int = 128, wm_batch: int = 64,
              pol_steps: int = 64, pol_batch: int = 32,
              pol_n_imagine: int = 6, clear_buffer: bool = True,
              prune_pct: float = 0.33, surprise_factor: float = 1.0,
              pol_gamma: float = 1.1):
        """One full sleep cycle with Coreset / Addendum and active forgetting.

        Phases:
          1. Consolidate wake session journal into Addendum (added_wake).
          2. Prune ~33% of Coreset into Addendum as candidates for active forgetting.
          3. Phase 1: Train World Model on the pruned Coreset (yielding 'wm_coreset').
          4. Filter Addendum: evaluate surprise under updated WM (dynamics + cost divergence).
             Discard familiar sequences; retain surprising sequences.
          5. Phase 2: Train World Model on the surviving Addendum (yielding 'wm_addendum').
          6. Policy training via imagined rollouts (yielding 'pol').
          7. Consolidation: commit Addendum into Coreset, refresh wake latent, yield 'done'.
        """
        self.mode = "sleep"
        self.last_dream_record = None

        # 1. Extract recent wake session into Addendum
        self.buffer.extract_addendum()

        # 1b. Intra-Addendum deduplication via naive distance across all channels
        n_intra_dropped = self.buffer.deduplicate_addendum(eps=0.04)
        if n_intra_dropped > 0:
            print(f"[sleep:intra-addendum] {n_intra_dropped} doublons redondants éliminés de l'Addendum.")

        # 2. Prune candidates from Coreset into Addendum (active forgetting)
        n_pruned = 0
        if self.buffer.coreset_size > 1 and prune_pct > 0.0:
            n_pruned = self.buffer.prune_coreset(prune_pct)

        yield "prune", {
            "pruned": n_pruned,
            "coreset": self.buffer.coreset_size,
            "addendum": self.buffer.addendum_size,
        }

        # 3. Epoch allocation between Coreset and Addendum
        has_coreset = (self.buffer.coreset_size > 0)
        has_addendum = (self.buffer.addendum_size > 0)

        if not has_coreset and not has_addendum:
            self.mode = "wake"
            yield "done", {"wm_loss": float("nan"), "pol_loss": float("nan")}
            return

        if has_coreset:
            if has_addendum:
                epochs_core = max(1, int(wm_epochs * 0.7))
                epochs_add = wm_epochs - epochs_core
            else:
                epochs_core = wm_epochs
                epochs_add = 0
        else:
            epochs_core = 0
            epochs_add = wm_epochs

        # 4. Phase 1: Train World Model on Coreset
        core_losses = []
        if epochs_core > 0:
            self.world.train()
            self.sleep_wm_epochs = epochs_core
            for ep in range(epochs_core):
                self.sleep_wm_step = ep + 1
                seqs = self.buffer.sample_coreset(wm_batch)
                if not seqs:
                    break
                loss_val = self._train_wm_batch(seqs)
                core_losses.append(loss_val)
                yield "wm_coreset", loss_val
        self.last_wm_coreset_loss = (sum(core_losses) / len(core_losses)) if core_losses else float("nan")

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
        yield "filter", self.last_filter_stats

        # 6. Phase 2: Train World Model on surviving Addendum
        add_losses = []
        if self.buffer.addendum_size > 0 and epochs_add > 0:
            self.world.train()
            self.sleep_wm_epochs = epochs_add
            for ep in range(epochs_add):
                self.sleep_wm_step = ep + 1
                seqs = self.buffer.sample_addendum(wm_batch)
                if not seqs:
                    break
                loss_val = self._train_wm_batch(seqs)
                add_losses.append(loss_val)
                yield "wm_addendum", loss_val
        self.last_wm_addendum_loss = (sum(add_losses) / len(add_losses)) if add_losses else float("nan")

        all_wm_losses = core_losses + add_losses
        self.last_wm_loss = (sum(all_wm_losses) / len(all_wm_losses)) if all_wm_losses else float("nan")

        # 7. Train Policy (dream imagination)
        for label, value in self.train_policy(pol_steps, pol_batch, gamma=pol_gamma, n_imagine=pol_n_imagine):
            yield label, value

        # 8. Consolidate Addendum into Coreset with cross-deduplication & memory decay
        cons_stats = self.buffer.consolidate_addendum_into_coreset(eps=0.04, decay=0.01)
        self._refresh_wake_latent()
        self.mode = "wake"

        print("\n" + "=" * 62)
        print(f"[sleep:consolidation Coreset] Déduplication naïve (eps=0.04, decay=0.01) :")
        print(f"  - Nouvelles mémoires ajoutées   : {cons_stats['added']}")
        print(f"  - Mémoires existantes ravivées  : {cons_stats['refreshed']}")
        print(f"  - Mémoires froides évincées     : {cons_stats['evicted']}")
        print(f"  - Taille finale du Coreset      : {cons_stats['coreset_size']}")
        print("=" * 62 + "\n")

        stats = {
            "wm_loss": self.last_wm_loss,
            "wm_coreset_loss": self.last_wm_coreset_loss,
            "wm_addendum_loss": self.last_wm_addendum_loss,
            "filter": self.last_filter_stats,
            "consolidation": cons_stats,
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
