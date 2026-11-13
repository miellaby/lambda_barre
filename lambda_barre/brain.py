"""Cerveau de λ̄ — refonte WM à dénouement (REFONTE.md).

Pendant l'*éveil* (wake) la policy pilote les consignes et chaque transition
est journalisée dans un buffer d'expérience. Pendant le *sommeil* (sleep) le
cerveau entraîne hors ligne :

  1. le world model apprend la séquence enregistrée réordonnée :
     ``contexte | dénouement | conséquence inversée | action de décision``
     (voir ``compute_seq_layout`` / ``build_sequence``) ;
  2. la policy imite l'action de décision générée par le WM conditionné sur
     un dénouement élu (échantillonné puis élu selon f(probable, favorable)),
     la loss étant pondérée par la probabilité du dénouement élu.

Une salve est une liste de 16 tokens denses (13 capteurs + 3 moteurs), chaque
token un vecteur de 25 floats (voir ``tokenize``). Une *séquence enregistrée*
est une liste plate de ``seq_len`` tokens dans l'ordre rétrotemporel ci-dessus.
Le journal de veille reste un flux de salves temporelles ; le réordonnancement
(et le calcul du token dénouement, qui dépend des salves futures) se fait à
l'extraction des séquences (``ExperienceBuffer.extract_addendum``).

Un seul masque (strictement causal), un seul forward par batch d'entraînement :
la loss porte sur tous les tokens à partir du dénouement, le contexte sert de
prompt (poids 0).
"""
from __future__ import annotations

import itertools
import math
import os
import random
import time
from collections import deque

import torch

from . import models as M
from .dream import DreamStep, DreamTrajectory, DreamRecord
from .models import build_salve_mask
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
    """Géométrie de la séquence enregistrée (REFONTE.md) :

        contexte | dénouement | conséquence inversée | action de décision

    - contexte : ``[s0,a0, ..., a_{n_ctx-1}, s_{n_ctx}]`` (``ctx_len`` tokens) ;
    - dénouement : 1 token EOS à la position ``eos_pos = ctx_len`` ;
    - conséquence inversée : les états post-décision du plus tardif au plus
      précoce, ``s_{seq_steps} .. s_{n_ctx+1}`` (``n_imagine`` salves de 13
      tokens, ``conseq_len`` tokens) ;
    - action de décision : ``a_{n_ctx}`` (3 tokens) en fin de séquence.

    ``decision_pos`` reste ``ctx_len`` : c'est la fin du contexte (dernier
    token = s_{n_ctx}) ; l'action de décision siège à ``action_pos``.
    """
    assert n_ctx < seq_steps, f"n_ctx ({n_ctx}) must be strictly less than seq_steps ({seq_steps})"
    ctx_len = n_ctx * SALVE_TOKENS + STATE_TOKENS
    eos_pos = ctx_len
    n_imagine = seq_steps - n_ctx
    conseq_start = eos_pos + 1
    conseq_len = n_imagine * STATE_TOKENS
    action_pos = conseq_start + conseq_len
    seq_len = action_pos + ACTION_TOKENS
    return {
        "seq_steps": seq_steps,
        "n_ctx": n_ctx,
        "seq_len": seq_len,
        "ctx_len": ctx_len,
        "decision_pos": ctx_len,
        "eos_pos": eos_pos,
        "conseq_start": conseq_start,
        "conseq_len": conseq_len,
        "action_pos": action_pos,
        "n_imagine": n_imagine,
    }


def compute_salve_positions(seq_steps: int = DEFAULT_SEQ_STEPS,
                            n_ctx: int = DEFAULT_N_CTX) -> list[int]:
    """Position temporelle (indice de salve) de chaque token de la séquence
    enregistrée, pour l'encodage positionnel sinusoïdal keyed by salve :
    contexte 0..n_ctx, EOS à ``seq_steps``, conséquence inversée
    ``seq_steps..n_ctx+1`` (décroissant), action de décision à ``n_ctx``."""
    layout = compute_seq_layout(seq_steps, n_ctx)
    sp = [0] * layout["seq_len"]
    for i in range(n_ctx):
        for t in range(SALVE_TOKENS):
            sp[i * SALVE_TOKENS + t] = i
    for t in range(STATE_TOKENS):
        sp[n_ctx * SALVE_TOKENS + t] = n_ctx
    sp[layout["eos_pos"]] = seq_steps
    for j, k in enumerate(range(seq_steps, n_ctx, -1)):
        for t in range(STATE_TOKENS):
            sp[layout["conseq_start"] + j * STATE_TOKENS + t] = k
    for t in range(ACTION_TOKENS):
        sp[layout["action_pos"] + t] = n_ctx
    return sp


def build_target_tables(seq_steps: int = DEFAULT_SEQ_STEPS,
                        n_ctx: int = DEFAULT_N_CTX) -> tuple[torch.Tensor, torch.Tensor]:
    """Tables (valid, weight) de la loss WM, une par ligne prédictrice p
    (la ligne p prédit le token p+1) :

    - poids 0 sur le contexte (tokens 0..ctx_len-1) : prompt, pas appris ;
    - poids 1.0 sur tous les tokens à partir du dénouement (inclus) ;
    - ``valid`` = slots de signal valides du token cible (EOS : 6 slots,
      états : _N_SIGNALS[tidx], actions : _N_SIGNALS[13..15]).
    """
    layout = compute_seq_layout(seq_steps, n_ctx)
    seq_len = layout["seq_len"]
    ctx_len = layout["ctx_len"]
    eos_pos = layout["eos_pos"]
    conseq_start = layout["conseq_start"]
    action_pos = layout["action_pos"]
    valid = torch.zeros(seq_len - 1, N_SIGNAL, dtype=torch.bool)
    weight = torch.zeros(seq_len - 1, N_SIGNAL, dtype=torch.float32)
    for p in range(seq_len - 1):
        q = p + 1
        if q == eos_pos:
            n = _EOS_NSLOTS
        elif q < ctx_len:
            n = _N_SIGNALS[q % SALVE_TOKENS]
        elif q < action_pos:
            n = _N_SIGNALS[(q - conseq_start) % STATE_TOKENS]
        else:
            n = _N_SIGNALS[STATE_TOKENS + (q - action_pos)]
        valid[p, :n] = True
        weight[p, :n] = 0.0 if q < ctx_len else 1.0
    return valid, weight


# --- dénouement (EOS) ---------------------------------------------------------
_EOS_CANAL = (0.0, 0.0, 1.0, 0.0)
_EOS_NSLOTS = 6
_EOS_DECAY = 0.9          # décroissance temporelle de l'écart au but dans l'EOS
_EOS_POIDS_CHEMIN = 0.1   # poids du coût du chemin dans le progrès par canal
_EOS_MAX_P = 8.0          # exposant du max flou agrégeant les canaux (relief_score)
_EOS_CHANNEL_W_T = (torch.tensor(_EOS_CHANNEL_W, dtype=torch.float32)
                    / sum(_EOS_CHANNEL_W))                   # sums to 1

# Structural template of the dénouement token: intero modality, dedicated
# canal, state type, signal slots filled with the 6 cost-aggregate slots.
_EOS_TEMPLATE = torch.tensor(
    [0.0] + list(_M_INTERO) + list(_EOS_CANAL) + [0.0] * N_SIGNAL,
    dtype=torch.float32)


def _eos_slots_of_window(window: list, n_ctx: int = DEFAULT_N_CTX,
                         decay: float = _EOS_DECAY,
                         poids_chemin: float = _EOS_POIDS_CHEMIN) -> list[float]:
    """Les 6 slots du token dénouement d'une fenêtre temporelle de salves
    (seq_steps+1 salves de 16 tokens) : progrès par canal
    f = P - poids_chemin*A, P = c[s_{n_ctx}] - c[fin], A = Σ_k decay^k (c[s_{n_ctx+1+k}] - c[fin]).
    Slot 5 = inconfort (-confort). 0.5 = neutre."""
    L = len(window)
    if L <= n_ctx + 1:
        return [0.5] * _EOS_NSLOTS
    try:
        c = []
        for salve in window:
            row = [salve[_COST_IDX][SIG_OFFSET + j] for j in range(5)]
            row.append(-salve[_REWARD_IDX][SIG_OFFSET])    # inconfort = -confort
            c.append(row)
        P = [c[n_ctx][j] - c[-1][j] for j in range(_EOS_NSLOTS)]
        out = []
        for j in range(_EOS_NSLOTS):
            diffs = [(c[k][j] - c[-1][j]) * decay ** (k - n_ctx - 1) for k in range(n_ctx + 1, L)]
            A = sum(diffs)
            r = P[j] - poids_chemin * A
            out.append(0.5 + 0.5 * r / (1.0 + abs(r)))
        return out
    except (IndexError, TypeError):
        return [0.5] * _EOS_NSLOTS


def eos_token_from_slots(slots) -> list[float]:
    """Token dénouement (25 floats) depuis 6 slots [0,1] (liste ou tensor)."""
    if hasattr(slots, "tolist"):
        slots = slots.tolist()
    tok = [0.0] + list(_M_INTERO) + list(_EOS_CANAL) + list(slots[:_EOS_NSLOTS])
    tok += [0.0] * (N_SIGNAL - _EOS_NSLOTS)
    return tok


def build_sequence(window: list, n_ctx: int = DEFAULT_N_CTX,
                   decay: float = _EOS_DECAY,
                   poids_chemin: float = _EOS_POIDS_CHEMIN) -> list[list[float]]:
    """Réordonne une fenêtre temporelle de ``seq_steps+1`` salves en séquence
    enregistrée (liste plate de ``seq_len`` tokens) :

        contexte | dénouement | conséquence inversée | action de décision

    - contexte : salves complètes ``s_i+a_i`` (i < n_ctx) puis ``s_{n_ctx}`` seul ;
    - dénouement : 1 token calculé depuis la fenêtre (ground truth) ;
    - conséquence inversée : états seuls ``s_{seq_steps} .. s_{n_ctx+1}`` ;
    - action de décision : ``a_{n_ctx}`` (3 tokens), dernier token de la séquence.
    """
    seq_steps = len(window) - 1
    seq: list[list[float]] = []
    for i in range(n_ctx):
        seq.extend(window[i])                              # s_i + a_i
    seq.extend(state_tokens(window[n_ctx]))                # s_{n_ctx}
    seq.append(eos_token_from_slots(
        _eos_slots_of_window(window, n_ctx=n_ctx, decay=decay, poids_chemin=poids_chemin)))
    for k in range(seq_steps, n_ctx, -1):                  # conséquence inversée
        seq.extend(state_tokens(window[k]))
    seq.extend(action_tokens(window[n_ctx]))               # a_{n_ctx}
    return seq


def sequence_to_window(seq: list, n_ctx: int = DEFAULT_N_CTX,
                       seq_steps: int = DEFAULT_SEQ_STEPS) -> list[list[list[float]]]:
    """Inverse de ``build_sequence`` (pour l'affichage) : reconstruit la fenêtre
    temporelle de ``seq_steps+1`` salves. Les actions a_{n_ctx+1..} ne sont pas
    dans la séquence enregistrée : elles sont laissées à zéro."""
    layout = compute_seq_layout(seq_steps, n_ctx)
    ctx_len = layout["ctx_len"]
    conseq_start = layout["conseq_start"]
    action_pos = layout["action_pos"]
    window: list[list[list[float]]] = []
    for i in range(n_ctx):
        window.append([list(t) for t in seq[i * SALVE_TOKENS:(i + 1) * SALVE_TOKENS]])
    s_dec = [list(t) for t in seq[ctx_len - STATE_TOKENS:ctx_len]]
    a_dec = [list(t) for t in seq[action_pos:action_pos + ACTION_TOKENS]]
    window.append(s_dec + a_dec)
    for k in range(n_ctx + 1, seq_steps + 1):              # s_{n_ctx+1} .. s_{seq_steps}
        j = seq_steps - k                                  # position dans la conséquence inversée
        s = [list(t) for t in seq[conseq_start + j * STATE_TOKENS:
                                  conseq_start + (j + 1) * STATE_TOKENS]]
        window.append(s + [[0.0] * DENSE_DIM] * ACTION_TOKENS)
    return window


def relief_score(x, eos_pos: int | None = None, p: float = _EOS_MAX_P):
    """Score de soulagement : max flou pondéré des 6 slots du token dénouement
    (poids de canaux normalisés, exposant p). Accepte un tensor de slots
    [..., 6] ou une séquence enregistrée (liste plate : le token dénouement est
    lu à ``eos_pos``). Retourne un tensor [...] ; positif = gain."""
    if isinstance(x, torch.Tensor):
        slots = x
    else:
        if eos_pos is None:
            eos_pos = compute_seq_layout()[ "eos_pos"]
        slots = torch.tensor(x[eos_pos][SIG_OFFSET:SIG_OFFSET + _EOS_NSLOTS],
                             dtype=torch.float32)
    w = _EOS_CHANNEL_W_T.to(slots.device)
    mf = ((w * slots ** p).sum(dim=-1)) ** (1.0 / p)
    return 2.0 * (mf - 0.5)


# --- static detection ---------------------------------------------------------
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


# --- action encode/decode metadata ---------------------------------------------
_ACT_KEYS = ("membre_avant_theta", "membre_avant_d", "membre_arriere_theta", "membre_arriere_d",
             "queue_theta")
_ACT_LAYOUT = []   # (tok_in_block, slot, key)
for _i in range(STATE_TOKENS, SALVE_TOKENS):    # 13, 14, 15
    for _k, _key in enumerate(_LAYOUT[_i].signals):
        _ACT_LAYOUT.append((_i - STATE_TOKENS, _k, _key))
_ACT_TOK = torch.tensor([t for t, _, _ in _ACT_LAYOUT], dtype=torch.long)
_ACT_SLOT = torch.tensor([s for _, s, _ in _ACT_LAYOUT], dtype=torch.long)
_ACT_SIGNED = torch.tensor([_BY_KEY[k].signed for _, _, k in _ACT_LAYOUT],
                           dtype=torch.bool)
_ACT_TEMPLATE = torch.tensor(
    [_prefix(_LAYOUT[i]) + [0.0] * N_SIGNAL
     for i in range(STATE_TOKENS, SALVE_TOKENS)], dtype=torch.float32)

# Flat signal-grid indices for the 51 policy scalars (tokens 0..11, hors intero 12).
_POL_SCALAR_IDX = []
for _i in range(_INTERO_IDX):
    for _k in range(_N_SIGNALS[_i]):
        _POL_SCALAR_IDX.append(_i * N_SIGNAL + _k)
assert len(_POL_SCALAR_IDX) == N_POLICY_STATE

# Precomputed cost metadata for the vectorized salve cost.
_COST_SCALES = torch.tensor([_BY_KEY[k].max_val for k in _COST_KEYS], dtype=torch.float32)
_COST_SIGNED = torch.tensor([_BY_KEY[k].signed for k in _COST_KEYS], dtype=torch.bool)
_COST_W = torch.tensor([_COST_WEIGHTS[k] for k in _COST_KEYS], dtype=torch.float32)
_CONFORT_SCALE = _BY_KEY["confort"].max_val
_CONFORT_W = _COST_WEIGHTS["confort"]


class ExperienceBuffer:
    """Two-tier experience memory: Coreset (long-term) + Addendum (wake session).

    The wake journal is a linear stream of temporal salves (16 tokens each),
    sealed into segments at discontinuities. ``extract_addendum`` cuts
    sliding windows of ``seq_len`` salves and stores them in the Addendum as
    *recorded sequences* — flat token lists reordered by ``build_sequence``
    (contexte | dénouement | conséquence inversée | action de décision).
    """

    def __init__(self, seq_len: int = DEFAULT_SEQ_STEPS + 1, capacity: int = 1000,
                 pool_capacity: int = 1000, seed: int = 0,
                 n_ctx: int | None = None,
                 eos_decay: float = _EOS_DECAY, eos_poids_chemin: float = _EOS_POIDS_CHEMIN):
        self.seq_len = seq_len                      # salves per temporal window
        self.capacity = capacity
        self.pool_capacity = pool_capacity
        self.seq_steps = seq_len - 1
        if n_ctx is None:
            self.n_ctx = min(DEFAULT_N_CTX, max(1, self.seq_steps // 2))
        else:
            self.n_ctx = n_ctx
        self.eos_decay = eos_decay
        self.eos_poids_chemin = eos_poids_chemin
        self._segments: deque[list[list[float]]] = deque()
        self._current: list[list[float]] = []
        self._coreset: list[list[list[float]]] = []
        self._coreset_vivacity: list[float] = []
        self._coreset_meta: list[dict] = []
        self._addendum: list[list[list[float]]] = []
        self._addendum_meta: list[dict] = []
        self._rng = random.Random(seed)

    @staticmethod
    def _default_meta(origin: str = "wake") -> dict:
        return {
            "origin": origin,
            "vivacity": 1.0,
            "learn_count": 0,
            "relief_score": None,
            "surprise": None,
        }

    @property
    def _valid_segments(self) -> list[list[list[float]]]:
        segs = list(self._segments)
        if len(self._current) >= self.seq_len:
            segs.append(self._current)
        return segs

    @property
    def num_segments(self) -> int:
        n = len(self._segments)
        if self._current:
            n += 1
        return n

    @property
    def pool_size(self) -> int:
        return len(self._coreset)

    @property
    def coreset_size(self) -> int:
        return len(self._coreset)

    @property
    def addendum_size(self) -> int:
        return len(self._addendum) + self.journal_len

    @property
    def journal_len(self) -> int:
        total = sum(len(s) - self.seq_len + 1 for s in self._segments)
        if len(self._current) >= self.seq_len:
            total += len(self._current) - self.seq_len + 1
        return total

    def __len__(self) -> int:
        return len(self._coreset) + len(self._addendum) + self.journal_len

    def push(self, salve: list[list[float]]) -> None:
        """Append one temporal salve to the current active segment."""
        self._current.append(salve)
        self._enforce_capacity()

    def boundary(self) -> None:
        """Seal the active segment (discontinuity: flip, reset, teleport)."""
        if len(self._current) >= self.seq_len:
            self._segments.append(self._current)
        self._current = []

    def new_segment(self) -> None:
        self.boundary()

    def extract_addendum(self, stride: int = 1) -> int:
        """Cut sliding windows from the journal segments and store them in the
        Addendum as recorded sequences (``build_sequence``). Clears the journal."""
        added = 0
        for seg in self._valid_segments:
            n_windows = len(seg) - self.seq_len + 1
            if n_windows <= 0:
                continue
            offsets = list(range(0, n_windows, stride))
            if (n_windows - 1) % stride != 0:
                offsets.append(n_windows - 1)
            for offset in offsets:
                window = seg[offset:offset + self.seq_len]
                self._addendum.append(build_sequence(
                    window, n_ctx=self.n_ctx, decay=self.eos_decay,
                    poids_chemin=self.eos_poids_chemin))
                self._addendum_meta.append(self._default_meta("wake"))
                added += 1
        self._segments.clear()
        self._current.clear()
        return added

    def commit_addendum(self) -> int:
        """Move all Addendum sequences into the Coreset (simple FIFO capacity
        eviction), then clear the Addendum. Returns the number committed."""
        n = len(self._addendum)
        self._coreset.extend(self._addendum)
        self._coreset_vivacity.extend([1.0] * n)
        self._coreset_meta.extend(self._addendum_meta)
        self._addendum.clear()
        self._addendum_meta.clear()
        while len(self._coreset) > self.pool_capacity:
            self._coreset.pop(0)
            self._coreset_vivacity.pop(0)
            self._coreset_meta.pop(0)
        return n

    def consolidate(self, stride: int = 1) -> int:
        """Extraction + immediate commit (headless bootstrap)."""
        added = self.extract_addendum(stride=stride)
        self.commit_addendum()
        return added

    def clear_journal(self) -> None:
        self._segments.clear()
        self._current.clear()

    def clear(self) -> None:
        self.clear_journal()
        self._coreset.clear()
        self._coreset_vivacity.clear()
        self._coreset_meta.clear()
        self._addendum.clear()
        self._addendum_meta.clear()

    def save(self, path: str) -> None:
        torch.save({
            "seq_len": self.seq_len,
            "n_ctx": self.n_ctx,
            "coreset": list(self._coreset),
            "coreset_vivacity": list(self._coreset_vivacity),
            "coreset_meta": [dict(m) for m in self._coreset_meta],
            "addendum": list(self._addendum),
            "addendum_meta": [dict(m) for m in self._addendum_meta],
            "segments": list(self._segments),
            "current": self._current,
        }, path)

    def load(self, path: str) -> bool:
        """Load from disk. Returns False (and leaves the buffer empty) when the
        file is missing or holds sequences of an incompatible layout."""
        if not os.path.exists(path):
            return False
        data = torch.load(path, weights_only=False)
        pool = data.get("coreset", [])
        expected = compute_seq_layout(self.seq_steps, self.n_ctx)["seq_len"]
        if pool and len(pool[0]) != expected:
            return False  # legacy layout — checkpoints invalid after the refonte
        self._coreset = [list(seq) for seq in pool][:self.pool_capacity]
        viv = data.get("coreset_vivacity", None)
        self._coreset_vivacity = ([float(v) for v in viv] if viv is not None
                                  and len(viv) == len(self._coreset)
                                  else [1.0] * len(self._coreset))
        cmeta = data.get("coreset_meta", None)
        self._coreset_meta = ([dict(m) for m in cmeta] if cmeta is not None
                              and len(cmeta) == len(self._coreset)
                              else [self._default_meta("coreset") for _ in self._coreset])
        for m, v in zip(self._coreset_meta, self._coreset_vivacity):
            m["vivacity"] = v
        self._addendum = list(data.get("addendum", []))
        self._addendum_meta = [dict(m) for m in data.get("addendum_meta", [])]
        self._segments = deque(data.get("segments", []))
        self._current = list(data.get("current", []))
        return True

    def _enforce_capacity(self) -> None:
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

    def sample_coreset(self, batch: int) -> list:
        if not self._coreset:
            return []
        return self._rng.sample(list(self._coreset), min(batch, len(self._coreset)))

    def sample_addendum(self, batch: int) -> list:
        if not self._addendum:
            return []
        return self._rng.sample(self._addendum, min(batch, len(self._addendum)))

    def sample(self, batch: int) -> list:
        """Uniform sample over Coreset + Addendum, falling back to journal windows."""
        pool = list(self._coreset) + self._addendum
        if pool:
            return self._rng.sample(pool, min(batch, len(pool)))
        valid_segs = self._valid_segments
        if not valid_segs:
            return []
        counts = [len(s) - self.seq_len + 1 for s in valid_segs]
        total = sum(counts)
        if total <= 0:
            return []
        indices = self._rng.sample(range(total), min(batch, total))
        cum = list(itertools.accumulate(counts))
        out = []
        for idx in indices:
            seg_i = next(i for i, c in enumerate(cum) if idx < c)
            offset = idx - (cum[seg_i - 1] if seg_i > 0 else 0)
            seg = valid_segs[seg_i]
            out.append(build_sequence(seg[offset:offset + self.seq_len],
                                      n_ctx=self.n_ctx, decay=self.eos_decay,
                                      poids_chemin=self.eos_poids_chemin))
        return out


class Brain:
    """Holds both networks, drives the live loop, and runs sleep training.

    Lifecycle in the main loop:

        brain.brain_tick(salve)       -> 5 consignes (WM forward + policy action)
        brain.record_tick(salve)      # at WM cadence: wake history + journal
        brain.sleep_cycle()           # on demand: offline training of both models
    """

    def __init__(self, lr_wm: float = 3e-4, lr_pol: float = 2e-4, act_std: float = 0.0,
                 device: str | None = None, seed: int = 0,
                 compile_wm: bool = False,
                 seq_steps: int = DEFAULT_SEQ_STEPS,
                 n_ctx: int = DEFAULT_N_CTX,
                 d_model: int = 64, nhead: int = 4, layers: int = 3,
                 dim_ff: int = 384, dropout: float = 0.0,
                 pol_hidden: int = 256,
                 buffer_capacity: int = 1000, buffer_pool_capacity: int = 1000,
                 eos_decay: float = _EOS_DECAY,
                 eos_poids_chemin: float = _EOS_POIDS_CHEMIN,
                 eos_max_p: float = _EOS_MAX_P,
                 # inférence du dénouement : échantillonnage + élection
                 eos_temperature: float = 0.05,
                 eos_top_k: int = 8,
                 eos_top_p: float = 0.9,
                 eos_w_fav: float = 1.0,
                 eos_satisfying: float = 0.0,
                 **_legacy):
        torch.manual_seed(seed)
        self.device, self.device_desc = M.configure_hardware(device)
        if self.device.type == "cpu":
            torch.set_num_threads(min(2, torch.get_num_threads()))
        self.seq_steps = seq_steps
        self.n_ctx = n_ctx
        layout = compute_seq_layout(seq_steps, n_ctx)
        self.seq_len = layout["seq_len"]
        self.ctx_len = layout["ctx_len"]
        self.decision_pos = layout["decision_pos"]
        self.eos_pos = layout["eos_pos"]
        self.conseq_start = layout["conseq_start"]
        self.conseq_len = layout["conseq_len"]
        self.action_pos = layout["action_pos"]
        self.n_imagine = layout["n_imagine"]

        self.world = M.WorldModel(d_model=d_model, nhead=nhead, layers=layers,
                                  dim_ff=dim_ff, dropout=dropout).to(self.device)
        self.policy = M.Policy(n_state=N_POLICY_STATE + d_model,
                               hidden=pol_hidden).to(self.device)
        self.latent_norm = torch.nn.LayerNorm(d_model).to(self.device)
        self.opt_wm = torch.optim.Adam(self.world.parameters(), lr=lr_wm)
        self.opt_pol = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.latent_norm.parameters()),
            lr=lr_pol)
        self.buffer = ExperienceBuffer(seq_len=seq_steps + 1, seed=seed, n_ctx=n_ctx,
                                       eos_decay=eos_decay, eos_poids_chemin=eos_poids_chemin,
                                       capacity=buffer_capacity,
                                       pool_capacity=buffer_pool_capacity)
        # Positions temporelles (PE sinusoïdal) et tables de loss du layout.
        self.salve_positions = torch.tensor(
            compute_salve_positions(seq_steps, n_ctx), dtype=torch.long, device=self.device)
        self.mask = build_salve_mask(self.seq_len, self.device)
        t_valid, t_weight = build_target_tables(seq_steps, n_ctx)
        self.target_valid = t_valid.to(self.device)
        self.target_weight = t_weight.to(self.device)
        self._eos_tmpl = _EOS_TEMPLATE.to(self.device)
        self.eos_decay = eos_decay
        self.eos_poids_chemin = eos_poids_chemin
        self.eos_max_p = eos_max_p
        self.eos_temperature = eos_temperature
        self.eos_top_k = eos_top_k
        self.eos_top_p = eos_top_p
        self.eos_w_fav = eos_w_fav
        self.eos_satisfying = eos_satisfying
        n_im = max(1, self.seq_steps - self.n_ctx)
        self._eos_decay_pow = torch.tensor([eos_decay ** k for k in range(n_im)],
                                           dtype=torch.float32, device=self.device)
        self._wm_loss = (torch.compile(self._wm_loss_impl, dynamic=False)
                         if compile_wm else self._wm_loss_impl)
        self.act_std = act_std
        self._cost_scales = _COST_SCALES.to(self.device)
        self._cost_signed = _COST_SIGNED.to(self.device)
        self._cost_w = _COST_W.to(self.device)
        self._pol_idx = torch.tensor(_POL_SCALAR_IDX, dtype=torch.long, device=self.device)
        self._act_tok = _ACT_TOK.to(self.device)
        self._act_slot = _ACT_SLOT.to(self.device)
        self._act_signed = _ACT_SIGNED.to(self.device)
        self._act_tmpl = _ACT_TEMPLATE.to(self.device)
        self._wake_history: list[tuple[list, list]] = []
        # last sleep stats, for the HUD
        self.last_wm_loss = float("nan")
        self.last_wm_coreset_loss = float("nan")
        self.last_wm_addendum_loss = float("nan")
        self.wm_loss_before_sleep = float("nan")
        self.wm_loss_ratio = float("nan")
        self.wm_plateau = False
        self.last_pol_loss = float("nan")
        self.last_pol_stats: dict | None = None
        self.last_dream_record: DreamRecord | None = None
        self.sleep_cycle_stats: dict = {}
        self.sleep_wm_epochs = 0
        self.sleep_wm_step = 0
        self.sleep_wm_batch = 0
        self.sleep_wm_batches = 0
        self.sleep_wm_n = 0
        self.sleep_pol_steps = 0
        self.sleep_pol_step = 0
        self._skip_phase = False
        self.mode = "wake"
        self._wm_time = 0.0
        self._pol_time = 0.0
        self._time_alpha = 0.1
        self._recording = True
        self._static_ticks = 0
        self._last_salve: list[list[float]] | None = None
        self._static_threshold = 6
        self._static_tol = 1e-4
        # Décalage action/conséquence en mode brain-off : la paire enregistrée
        # est (a_{n-k} ; s_n) — action retardée, état courant. 0 = brain-on.
        self.act_conseq_shift = 0
        self._act_hist: list[list[list[float]]] = []

    # --- live loop -----------------------------------------------------------
    def _sync_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    @torch.inference_mode()
    def _wm_forward(self, salve) -> tuple[torch.Tensor, torch.Tensor]:
        """WM forward on the wake history + current state; returns (latent, scalars)."""
        self.mode = "wake"
        cur_state = state_tokens(salve)
        ctx = self._build_context(cur_state)
        self.world.eval()
        t0 = time.perf_counter()
        self.world(ctx)
        latent = self.latent_norm(self.world.last_latent())
        self._sync_device()
        dt_ms = (time.perf_counter() - t0) * 1000
        if self._wm_time <= 0.0:
            self._wm_time = dt_ms
        else:
            self._wm_time = (1 - self._time_alpha) * self._wm_time \
                + self._time_alpha * dt_ms
        scalars = self._policy_scalars_batch(
            torch.tensor(cur_state, dtype=torch.float32,
                         device=self.device).unsqueeze(0))
        return latent, scalars

    @torch.inference_mode()
    def brain_tick(self, salve, std=None) -> tuple[float, float, float, float, float]:
        """One brain tick at policy cadence: WM forward pass + policy action."""
        latent, scalars = self._wm_forward(salve)
        pol_in = torch.cat([scalars, latent], dim=1)
        t0 = time.perf_counter()
        action, _, _ = self.policy.sample(
            pol_in, self.act_std if std is None else std)
        self._sync_device()
        dt_ms = (time.perf_counter() - t0) * 1000
        if self._pol_time <= 0.0:
            self._pol_time = dt_ms
        else:
            self._pol_time = (1 - self._time_alpha) * self._pol_time \
                + self._time_alpha * dt_ms
        a = action[0].tolist()
        return a[0], a[1], a[2], a[3], a[4]

    def act(self, salve, std=None):
        """Legacy alias for brain_tick()."""
        return self.brain_tick(salve, std=std)

    @property
    def is_recording(self) -> bool:
        return self._recording

    def record_tick(self, salve) -> None:
        """Record a salve at WM cadence: wake history + journal."""
        cur_state = state_tokens(salve)
        a_toks = action_tokens(salve)
        self._wake_history.append((cur_state, a_toks))
        if len(self._wake_history) > self.n_ctx:
            self._wake_history.pop(0)
        self.record(salve)

    def record(self, salve) -> bool:
        """Journal one temporal salve into the experience buffer (with the
        brain-off action/consequence shift k when act_conseq_shift > 0)."""
        if hasattr(salve, "tolist"):
            salve = salve.tolist()
        if self.act_conseq_shift > 0:
            self._act_hist.append(action_tokens(salve))
            if len(self._act_hist) > self.act_conseq_shift + 1:
                self._act_hist.pop(0)
            if len(self._act_hist) > self.act_conseq_shift:
                shifted = self._act_hist[-self.act_conseq_shift - 1]
                salve = [list(t) for t in state_tokens(salve)] + [list(t) for t in shifted]
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
        self._wake_history.clear()
        self._recording = True
        self._static_ticks = 0
        self._last_salve = None
        self.buffer.boundary()

    def boundary(self) -> None:
        self.clear_history()

    def _build_context(self, cur_state: list):
        """Build a [1, L, 25] context tensor from the wake history + current
        state: [s0, a0, s1, a1, ..., s_{k-1}, a_{k-1}, s_k]."""
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

    # --- batch tensors ---------------------------------------------------------
    def _batch_tensors(self, sequences) -> torch.Tensor:
        """Stack recorded sequences (flat token lists) into [B, seq_len, 25].
        Zeroes the interoception slots of every state salve (the carrier of the
        outcome signal stays masked from the WM inputs)."""
        B = len(sequences)
        vals = torch.zeros(B, self.seq_len, DENSE_DIM, device=self.device)
        for b, seq in enumerate(sequences):
            assert len(seq) == self.seq_len, \
                f"recorded sequence has {len(seq)} tokens, expected {self.seq_len}"
            vals[b] = torch.tensor(seq, dtype=torch.float32, device=self.device)
        # intero slots: contexte (salves 0..n_ctx) + conséquence inversée
        intero_pos = [k * SALVE_TOKENS + _INTERO_IDX for k in range(self.n_ctx + 1)]
        intero_pos += [self.conseq_start + j * STATE_TOKENS + _INTERO_IDX
                       for j in range(self.n_imagine)]
        vals[:, intero_pos, SIG_OFFSET:] = 0.0
        return vals

    # --- WM training -----------------------------------------------------------
    def _wm_masked_mse(self, pred: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
        """Masked MSE over valid signal slots. The target table weights 0 the
        context rows (prompt) and 1.0 every row from the dénouement on."""
        pred_signals = pred[:, :-1, :]                   # [B, L-1, 16]
        tgt_signals = vals[:, 1:, SIG_OFFSET:]           # [B, L-1, 16]
        w = self.target_weight.unsqueeze(0).expand_as(pred_signals)
        sq = (pred_signals - tgt_signals) ** 2
        sqm = (sq * w).sum(dim=(1, 2))                   # [B]
        wsum = w.sum(dim=(1, 2)).clamp(min=1.0)          # [B]
        return (sqm / wsum).mean()

    def _wm_loss_impl(self, vals: torch.Tensor) -> torch.Tensor:
        """One forward, one loss: causal mask + salve positions over the whole
        recorded sequence, MSE on all tokens from the dénouement on."""
        pred = self.world(vals, attn_mask=self.mask, is_causal=True,
                          salve_positions=self.salve_positions)
        return self._wm_masked_mse(pred, vals)

    def _train_wm_batch(self, sequences) -> float:
        """One WM optimization step on a batch of recorded sequences."""
        vals = self._batch_tensors(sequences)
        loss = self._wm_loss(vals)
        self.opt_wm.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world.parameters(), 1.0)
        self.opt_wm.step()
        return loss.item()

    def train_wm_phase(self, epochs: int, batch: int,
                       target_loss: float | None = None,
                       source: str = "coreset",
                       label: str = "wm"):
        """One WM training phase as a generator yielding ``(label, epoch, loss)``
        after each batch. 1 epoch = the whole dataset swept in batches without
        omission or replacement; addendum is shuffled each epoch."""
        buf = self.buffer
        seqs_all, metas_all = ((buf._addendum, buf._addendum_meta) if source == "addendum"
                               else (buf._coreset, buf._coreset_meta))
        n = len(seqs_all)
        if n == 0:
            return
        self.world.train()
        self.sleep_wm_epochs = epochs
        order = list(range(n))
        ep = 0
        while True:
            ep += 1
            self.sleep_wm_step = ep
            if source == "addendum":
                buf._rng.shuffle(order)
            self.sleep_wm_n = n
            self.sleep_wm_batches = max(1, (n + batch - 1) // batch)
            self.sleep_wm_batch = 0
            epoch_losses = []
            skip = False
            for start in range(0, n, batch):
                idxs = order[start:start + batch]
                self.sleep_wm_batch = start // batch + 1
                loss_c = self._train_wm_batch([seqs_all[i] for i in idxs])
                epoch_losses.append(loss_c)
                for i in idxs:
                    if i < len(metas_all):
                        metas_all[i]["surprise"] = loss_c
                yield label, ep, loss_c
                if self._skip_phase:
                    self._skip_phase = False
                    skip = True
                    break
            if skip:
                return
            for m in metas_all:
                m["learn_count"] = m.get("learn_count", 0) + 1
            loss_val = (sum(epoch_losses) / len(epoch_losses)) if epoch_losses else float("nan")
            if ep >= epochs and (target_loss is None or loss_val <= target_loss):
                break

    def evaluate_sequences_surprise(self, sequences, batch_size: int = 32) -> list[float]:
        """Per-sequence prediction MSE under the current WM (inference only)."""
        self.world.eval()
        out = []
        with torch.inference_mode():
            for i in range(0, len(sequences), batch_size):
                chunk = sequences[i:i + batch_size]
                if not chunk:
                    continue
                vals = self._batch_tensors(chunk)
                pred = self.world(vals, attn_mask=self.mask, is_causal=True,
                                  salve_positions=self.salve_positions)
                pred_signals = pred[:, :-1, :]
                tgt_signals = vals[:, 1:, SIG_OFFSET:]
                w = self.target_weight.unsqueeze(0).expand_as(pred_signals)
                sq = ((pred_signals - tgt_signals) ** 2 * w).sum(dim=(1, 2))
                wsum = w.sum(dim=(1, 2)).clamp(min=1.0)
                out.extend((sq / wsum).tolist())
        return out

    # --- vectorized helpers ------------------------------------------------------
    def _salve_cost_batch(self, gen: torch.Tensor) -> torch.Tensor:
        """Vectorized salve_cost over [B, 13, 25] (or [B, 16, 25])."""
        n = len(_COST_KEYS)
        c = gen[:, _COST_IDX, SIG_OFFSET:SIG_OFFSET + n]
        denorm = torch.where(self._cost_signed,
                             (2 * c - 1) * self._cost_scales,
                             c * self._cost_scales)
        confort = gen[:, _REWARD_IDX, SIG_OFFSET]
        confort_phys = (2 * confort - 1) * _CONFORT_SCALE
        return (denorm * self._cost_w).sum(1) + _CONFORT_W * confort_phys

    def _policy_scalars_batch(self, state_tokens: torch.Tensor) -> torch.Tensor:
        """[B, 13, 25] -> [B, 51] policy scalars (tokens 0..11)."""
        B = state_tokens.shape[0]
        sigs = state_tokens[:, :, SIG_OFFSET:SIG_OFFSET + N_SIGNAL]
        flat = sigs.reshape(B, STATE_TOKENS * N_SIGNAL)
        return flat.index_select(1, self._pol_idx)

    def _encode_action_batch(self, action: torch.Tensor) -> torch.Tensor:
        """Action [B, 5] in model space (signed [-1,1], unsigned [0,1]) ->
        action tokens [B, 3, 25] in token signal space [0,1]."""
        if action.dim() == 1:
            action = action.unsqueeze(0)
        B = action.shape[0]
        norm = torch.where(self._act_signed, (action + 1.0) / 2.0, action)
        toks = self._act_tmpl.unsqueeze(0).expand(B, -1, -1).clone()
        flat = toks[:, :, SIG_OFFSET:SIG_OFFSET + N_SIGNAL]
        flat = flat.scatter(1, (self._act_tok * N_SIGNAL + self._act_slot).unsqueeze(0).expand(B, -1),
                            norm)
        toks[:, :, SIG_OFFSET:SIG_OFFSET + N_SIGNAL] = flat
        return toks

    def _decode_action_batch(self, act_toks: torch.Tensor) -> torch.Tensor:
        """Action tokens [B, 3, 25] -> action [B, 5] in model space."""
        B = act_toks.shape[0]
        flat = act_toks[:, :, SIG_OFFSET:SIG_OFFSET + N_SIGNAL].reshape(B, ACTION_TOKENS * N_SIGNAL)
        idx = (self._act_tok * N_SIGNAL + self._act_slot)
        vals = flat.index_select(1, idx)                       # [B, 5] in [0,1]
        return torch.where(self._act_signed, 2.0 * vals - 1.0, vals)

    # --- WM inference: dénouement sampling + generation --------------------------
    @torch.inference_mode()
    def predict_denouement(self, ctx: torch.Tensor) -> torch.Tensor:
        """Mean dénouement predicted from the context alone: forward over
        [B, ctx_len, 25], read the 6 EOS slots at the last row. Returns [B, 6]."""
        L = ctx.shape[1]
        pred = self.world(ctx, attn_mask=build_salve_mask(L, ctx.device),
                          is_causal=True,
                          salve_positions=self.salve_positions[:L])
        return pred[:, -1, :_EOS_NSLOTS].clamp(0.0, 1.0)

    @torch.inference_mode()
    def sample_denouements(self, ctx: torch.Tensor, k: int | None = None,
                           temperature: float | None = None,
                           top_p: float | None = None,
                           eos_observed: torch.Tensor | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """Échantillonnage du dénouement depuis le contexte (REFONTE.md) :

        - le WM prédit la moyenne conditionnelle µ (vecteur continu, 6 slots) ;
        - on tire K vecteurs v ~ N(µ, T²) (température T), top-K/top-p sur les
          tirages (densité gaussienne) ;
        - chaque tirage est noté f(probable, favorable) = log p(v) + w_fav * relief(v) ;
        - le meilleur est élu (pas nécessairement le plus probable).

        Règle de l'observé : si ``eos_observed`` est fourni et que son relief
        atteint ``eos_satisfying``, on reste sur l'observé (p = 1).

        Returns (eos_elu [B, 6], p_elu [B]) — p_elu = softmax(log p) du tirage élu.
        """
        B = ctx.shape[0]
        device = ctx.device
        K = k or self.eos_top_k
        T = temperature if temperature is not None else self.eos_temperature
        mu = self.predict_denouement(ctx)                       # [B, 6]
        eps = torch.randn(B, K, _EOS_NSLOTS, device=device)
        v = (mu.unsqueeze(1) + T * eps).clamp(0.0, 1.0)          # [B, K, 6]
        logp = (-0.5 * ((v - mu.unsqueeze(1)) / T) ** 2).sum(dim=-1) \
            - _EOS_NSLOTS * math.log(T * math.sqrt(2.0 * math.pi))   # [B, K]
        # top-p : plus petit préfixe des tirages triés par logp dont la masse
        # softmax cumulée atteint top_p
        if top_p is not None and 0.0 < top_p < 1.0:
            p_sort, order = torch.sort(logp, dim=1, descending=True)
            cum = torch.softmax(p_sort, dim=1).cumsum(dim=1)
            keep = cum <= top_p
            keep[:, 0] = True                                   # au moins le plus probable
            logp = logp.scatter(1, order, torch.where(keep, p_sort, torch.full_like(p_sort, float("-inf"))))
        relief = relief_score(v, p=self.eos_max_p)              # [B, K]
        f = logp + self.eos_w_fav * relief
        elu = f.argmax(dim=1)                                   # [B]
        b_idx = torch.arange(B, device=device)
        eos_elu = v[b_idx, elu]                                 # [B, 6]
        p_elu = torch.softmax(logp, dim=1)[b_idx, elu]          # [B]
        if eos_observed is not None and self.eos_satisfying is not None:
            rel_obs = relief_score(eos_observed, p=self.eos_max_p)
            use_obs = rel_obs >= self.eos_satisfying
            eos_elu = torch.where(use_obs.unsqueeze(1), eos_observed, eos_elu)
            p_elu = torch.where(use_obs, torch.ones_like(p_elu), p_elu)
        return eos_elu, p_elu

    @torch.inference_mode()
    def generate_from_denouement(self, ctx: torch.Tensor,
                                 eos_vec: torch.Tensor) -> dict:
        """Génération auto-régressive depuis ``[contexte, dénouement élu]`` :
        déroule la conséquence inversée (``conseq_len`` tokens d'état, du plus
        tardif au plus précoce) jusqu'à l'action de décision (3 derniers
        tokens de la séquence). Masque strictement causal, positions explicites.

        Returns dict with ``conseq`` [B, conseq_len, 25], ``action`` [B, 3, 25],
        ``full`` [B, seq_len, 25]."""
        B = ctx.shape[0]
        device = ctx.device
        eos_toks = self._eos_tmpl.unsqueeze(0).expand(B, -1).clone()
        eos_toks[:, SIG_OFFSET:SIG_OFFSET + _EOS_NSLOTS] = eos_vec
        curr = torch.cat([ctx, eos_toks.unsqueeze(1)], dim=1)   # [B, ctx_len+1, 25]
        n_gen = self.conseq_len + ACTION_TOKENS
        gen = []
        for t in range(n_gen):
            L = curr.shape[1]
            out = self.world(curr, attn_mask=build_salve_mask(L, device),
                             is_causal=True,
                             salve_positions=self.salve_positions[:L])
            sig = out[:, -1, :].clamp(0.0, 1.0)
            q = self.ctx_len + 1 + t                             # token en cours de génération
            if q < self.action_pos:
                tidx = (q - self.conseq_start) % STATE_TOKENS
                tmpl = self.world.state_template[tidx].to(device)
                vm = self.world.state_valid_mask[tidx].to(device)
            else:
                i = q - self.action_pos
                tmpl = self.world.action_template[i].to(device)
                vm = self.world.action_valid_mask[i].to(device)
            sig = sig * vm
            tok = tmpl.unsqueeze(0).expand(B, -1).clone()
            tok[:, SIG_OFFSET:SIG_OFFSET + N_SIGNAL] = sig
            gen.append(tok)
            curr = torch.cat([curr, tok.unsqueeze(1)], dim=1)
        gen = torch.stack(gen, dim=1)                            # [B, n_gen, 25]
        return {
            "conseq": gen[:, :self.conseq_len, :],
            "action": gen[:, self.conseq_len:, :],
            "full": torch.cat([ctx, eos_toks.unsqueeze(1), gen], dim=1),
        }

    @torch.inference_mode()
    def rollout_from_sequence(self, seq: list) -> dict:
        """Diagnostic: depuis le contexte d'une séquence enregistrée, échantillonner
        + élire un dénouement (règle de l'observé appliquée), puis générer la
        conséquence inversée et l'action de décision."""
        vals = self._batch_tensors([seq])
        ctx = vals[:, :self.ctx_len, :]
        eos_obs = vals[:, self.eos_pos, SIG_OFFSET:SIG_OFFSET + _EOS_NSLOTS]
        eos_elu, p_elu = self.sample_denouements(ctx, eos_observed=eos_obs)
        gen = self.generate_from_denouement(ctx, eos_elu)
        return {
            "eos_observed": eos_obs[0].tolist(),
            "eos_elu": eos_elu[0].tolist(),
            "p_elu": float(p_elu[0]),
            "relief_observed": float(relief_score(eos_obs, p=self.eos_max_p)[0]),
            "relief_elu": float(relief_score(eos_elu, p=self.eos_max_p)[0]),
            "action": self._decode_action_batch(gen["action"])[0].tolist(),
            "action_recorded": self._decode_action_batch(
                vals[:, self.action_pos:self.action_pos + ACTION_TOKENS, :])[0].tolist(),
            "gen": gen,
        }

    # --- policy training ---------------------------------------------------------
    def train_policy(self, steps: int = 32, batch: int = 32):
        """Policy training by imitation of the WM-generated decision action.

        Per batch: latent = WM over the context part (the policy's input latent
        is produced by the WM on the context alone); a dénouement is sampled
        and elected; the WM generates the decision action from
        [contexte, dénouement élu]; the policy loss toward that action is
        weighted linearly by the elected dénouement's probability,
        ``loss *= p.detach()`` per sequence.

        Generator: yields ("pol", loss) after each step."""
        if len(self.buffer) < 1:
            return
        self.world.eval()
        self.policy.train()
        self.sleep_pol_steps = steps
        losses = []
        for step in range(steps):
            self.sleep_pol_step = step + 1
            seqs = self.buffer.sample(batch)
            if not seqs:
                break
            loss = self._policy_step(seqs, step, steps)
            losses.append(loss)
            yield "pol", loss
            if self._skip_phase:
                self._skip_phase = False
                break
        self.last_pol_loss = sum(losses) / len(losses) if losses else float("nan")

    def _policy_step(self, sequences, step_idx: int, total_steps: int) -> float:
        B = len(sequences)
        device = self.device
        vals = self._batch_tensors(sequences)
        ctx = vals[:, :self.ctx_len, :].clone()
        eos_obs = vals[:, self.eos_pos, SIG_OFFSET:SIG_OFFSET + _EOS_NSLOTS]
        with torch.inference_mode():
            self.world(ctx, attn_mask=build_salve_mask(self.ctx_len, device),
                       is_causal=True,
                       salve_positions=self.salve_positions[:self.ctx_len])
            latent_inf = self.latent_norm(self.world.last_latent().detach())
            scalars_inf = self._policy_scalars_batch(ctx[:, -STATE_TOKENS:])
            pol_in_inf = torch.cat([scalars_inf, latent_inf], dim=1)
            pol_action = self.policy(pol_in_inf)                 # [B, 5]
            eos_elu_inf, p_elu_inf = self.sample_denouements(ctx, eos_observed=eos_obs)
            gen_action_inf = self.generate_from_denouement(ctx, eos_elu_inf)["action"]
        # clone hors du bloc inference_mode : tensors normaux, utilisables en autograd
        latent = latent_inf.clone()
        scalars = scalars_inf.clone()
        pol_in = torch.cat([scalars, latent], dim=1)
        eos_elu = eos_elu_inf.clone()
        a_star = self._decode_action_batch(gen_action_inf.clone())   # [B, 5]
        pred = self.policy(pol_in)
        per_seq = ((pred - a_star) ** 2).mean(dim=1)             # [B]
        w = p_elu_inf.clone().detach()
        loss = (per_seq * w).sum() / w.sum().clamp(min=1e-8)
        self.opt_pol.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt_pol.step()
        self.last_pol_stats = {"accepted": B, "total": B}
        self.last_dream_record = self._build_dream_record(
            sequences[0], ctx[:1], pol_action[:1], a_star[:1],
            eos_obs[:1], eos_elu[:1], float(p_elu_inf[0]), loss.item(),
            step_idx, total_steps)
        return loss.item()

    def _build_dream_record(self, seq, ctx, pol_action, a_star, eos_obs, eos_elu,
                            p_elu, loss_val, step_idx, total_steps) -> DreamRecord:
        """Minimal Dream Theater record for the refonte policy step: lived
        sequence (rebuilt temporal window), context steps, and the WM-generated
        decision action under the elected dénouement."""
        device = self.device
        n_ctx = self.n_ctx
        context_steps = []
        for i in range(n_ctx):
            s_toks = ctx[0, i * SALVE_TOKENS: i * SALVE_TOKENS + STATE_TOKENS].tolist()
            a_vals = self._decode_action_batch(
                ctx[0:1, i * SALVE_TOKENS + STATE_TOKENS:(i + 1) * SALVE_TOKENS])[0].tolist()
            context_steps.append(DreamStep(state_tokens=s_toks, action=a_vals,
                                           label=f"s{i}+a{i}"))
        context_steps.append(DreamStep(
            state_tokens=ctx[0, n_ctx * SALVE_TOKENS: n_ctx * SALVE_TOKENS + STATE_TOKENS].tolist(),
            action=None, label=f"s{n_ctx} (decision)"))
        window = sequence_to_window(seq, n_ctx=n_ctx, seq_steps=self.seq_steps)
        lived_steps = []
        for k, salve in enumerate(window):
            a_vals = (self._decode_action_batch(torch.tensor(
                [salve[STATE_TOKENS:SALVE_TOKENS]], dtype=torch.float32,
                device=device))[0].tolist()
                if any(any(v != 0.0 for v in tok[SIG_OFFSET:]) for tok in salve[STATE_TOKENS:])
                else None)
            lived_steps.append(DreamStep(
                state_tokens=[list(t) for t in salve[:STATE_TOKENS]],
                action=a_vals, step_cost=salve_cost(salve), label=f"s{k}"))
        real_action = self._decode_action_batch(torch.tensor(
            [seq[self.action_pos:self.action_pos + ACTION_TOKENS]],
            dtype=torch.float32, device=device))[0].tolist()
        predicted_steps = context_steps[:n_ctx] + [
            DreamStep(state_tokens=[list(t) for t in salve[:STATE_TOKENS]],
                      action=None, step_cost=salve_cost(salve), label=f"s{k} (imaginé)")
            for k, salve in enumerate(window[n_ctx + 1:], start=n_ctx + 1)]
        predicted_steps.append(DreamStep(
            state_tokens=context_steps[-1].state_tokens,
            action=a_star[0].tolist(), label="a* (décision générée)"))
        return DreamRecord(
            step_idx=step_idx,
            total_steps=total_steps,
            loss=loss_val,
            facing=1,
            context_steps=context_steps,
            candidate_actions=[pol_action[0].tolist(), a_star[0].tolist()],
            real_action=real_action,
            trajectories=[
                DreamTrajectory(candidate_idx=0, regime_idx=0, regime_name="Policy",
                                steps=[], is_best_future=False, is_winner=False,
                                eos_pred=None),
                DreamTrajectory(candidate_idx=1, regime_idx=0, regime_name="Dénouement élu",
                                steps=predicted_steps[n_ctx:], is_best_future=True,
                                is_winner=True,
                                eos_pred=float(relief_score(eos_elu, p=self.eos_max_p)[0])),
            ],
            best_candidate_idx=1,
            best_future_indices=[0, 0],
            candidate_names=["0: POLICY", "1: DÉNOUEMENT ÉLU"],
            eos_forced=eos_elu[0].tolist(),
            lived_steps=lived_steps,
            predicted_steps=predicted_steps,
            eos_realized=eos_obs[0].tolist(),
        )

    # --- sleep cycle -------------------------------------------------------------
    def sleep_cycle(self, wm_coreset_epoch: int = 5, wm_addendum_epoch: int = 5,
                    wm_batch: int = 64, pol_steps: int = 32, pol_batch: int = 32,
                    wm_target_loss: float | None = 0.01,
                    clear_buffer: bool = True):
        """One full sleep cycle (refonte v1 — dataset optimization set aside):

          1. extract wake journal windows into the Addendum (recorded sequences) ;
          2. WM training on the Coreset (yielding 'wm_coreset') ;
          3. WM training on the Addendum (yielding 'wm_addendum') ;
          4. policy training by imitation of the WM-generated decision action
             (yielding 'pol') ;
          5. commit the Addendum into the Coreset, yield 'done'.
        """
        self.mode = "sleep"
        self.last_dream_record = None
        self.wm_loss_before_sleep = self.last_wm_loss
        coreset_before = self.buffer.coreset_size

        n_wake = self.buffer.extract_addendum()
        self.sleep_cycle_stats = {
            "coreset_before": coreset_before,
            "wake": n_wake,
            "coreset_after": None,
        }

        has_coreset = self.buffer.coreset_size > 0
        has_addendum = self.buffer.addendum_size > 0
        if not has_coreset and not has_addendum:
            self.mode = "wake"
            yield "done", {"wm_loss": float("nan"), "pol_loss": float("nan"),
                           "coreset_size": self.buffer.coreset_size}
            return

        core_losses, add_losses = [], []
        if has_coreset and wm_coreset_epoch > 0:
            for _, ep, loss_val in self.train_wm_phase(
                    wm_coreset_epoch, wm_batch, wm_target_loss,
                    source="coreset", label="wm_coreset"):
                core_losses.append(loss_val)
                yield "wm_coreset", loss_val
        self.last_wm_coreset_loss = (sum(core_losses[-5:]) / len(core_losses[-5:])
                                     if core_losses else float("nan"))

        if has_addendum and wm_addendum_epoch > 0:
            for _, ep, loss_val in self.train_wm_phase(
                    wm_addendum_epoch, wm_batch, wm_target_loss,
                    source="addendum", label="wm_addendum"):
                add_losses.append(loss_val)
                yield "wm_addendum", loss_val
        self.last_wm_addendum_loss = (sum(add_losses[-5:]) / len(add_losses[-5:])
                                      if add_losses else float("nan"))

        all_losses = core_losses + add_losses
        self.last_wm_loss = (sum(all_losses) / len(all_losses)) if all_losses else float("nan")
        if not math.isnan(self.wm_loss_before_sleep) and self.wm_loss_before_sleep > 0:
            self.wm_loss_ratio = self.last_wm_loss / self.wm_loss_before_sleep
        else:
            self.wm_loss_ratio = float("nan")
        self.wm_plateau = (not math.isnan(self.wm_loss_ratio)) and self.wm_loss_ratio > 0.95

        for label, value in self.train_policy(pol_steps, pol_batch):
            yield label, value

        n_committed = self.buffer.commit_addendum()
        self.sleep_cycle_stats["coreset_after"] = self.buffer.coreset_size
        self._recording = True
        self._static_ticks = 0
        self._last_salve = None
        self.mode = "wake"

        stats = {
            "wm_loss": self.last_wm_loss,
            "wm_coreset_loss": self.last_wm_coreset_loss,
            "wm_addendum_loss": self.last_wm_addendum_loss,
            "pol_loss": self.last_pol_loss,
            "coreset_size": self.buffer.coreset_size,
            "committed": n_committed,
            "cycle": self.sleep_cycle_stats,
        }
        yield "done", stats

    # --- persistence -------------------------------------------------------------
    def state_dict(self):
        return {
            "world": self.world.state_dict(),
            "policy": self.policy.state_dict(),
            "latent_norm": self.latent_norm.state_dict(),
        }

    def load_state_dict(self, sd):
        self.world.load_state_dict(sd["world"])
        self.policy.load_state_dict(sd["policy"])
        self.latent_norm.load_state_dict(sd["latent_norm"])

    def save_checkpoint(self, wm_path: str, pol_path: str, buf_path: str | None = None) -> None:
        torch.save({"world": self.world.state_dict(),
                    "opt_wm": self.opt_wm.state_dict()}, wm_path)
        torch.save({"policy": self.policy.state_dict(),
                    "latent_norm": self.latent_norm.state_dict(),
                    "opt_pol": self.opt_pol.state_dict()}, pol_path)
        if buf_path is not None:
            self.buffer.save(buf_path)

    def load_checkpoint(self, wm_path: str, pol_path: str, buf_path: str | None = None) -> None:
        if not os.path.exists(wm_path) or not os.path.exists(pol_path):
            raise FileNotFoundError("missing WM or policy checkpoint — run with --reset")
        wm = torch.load(wm_path, weights_only=False)
        self.world.load_state_dict(wm["world"])
        if "opt_wm" in wm:
            self.opt_wm.load_state_dict(wm["opt_wm"])
        pol = torch.load(pol_path, weights_only=False)
        self.policy.load_state_dict(pol["policy"])
        self.latent_norm.load_state_dict(pol["latent_norm"])
        if "opt_pol" in pol:
            self.opt_pol.load_state_dict(pol["opt_pol"])
        if buf_path is not None and os.path.exists(buf_path):
            self.buffer.load(buf_path)
