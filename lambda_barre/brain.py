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
token un vecteur de 29 floats (voir ``tokenize``). Une *séquence enregistrée*
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
from .models import build_salve_mask, KVCache
from .tokenize import (STATE_TOKENS, ACTION_TOKENS, SALVE_TOKENS, DENSE_DIM,
                       N_SIGNAL, _N_SIGNALS, N_POLICY_STATE, SIG_OFFSET,
                       _M_EOS, _EOS_CHANNEL_W,
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

        contexte | conséquence rétro-causale | action de décision

    - contexte : ``[s0,a0, ..., a_{n_ctx-1}, s_{n_ctx}]`` (``ctx_len`` tokens,
      ordre normal) ;
    - conséquence rétro-causale, à partir de ``conseq_start = eos_pos = ctx_len`` :
      unités de ``SALVE_TOKENS`` tokens pour ``k = seq_steps .. n_ctx+1`` —
      ``s_k`` avec ses 13 tokens d'état **inversés** (token 12 en premier), puis
      ``a_{k-1}`` (3 tokens, ordre interne normal). Le token 12 de la première
      unité (``s_{seq_steps}``) est le **token de dénouement** : ``eos_pos`` ;
    - action de décision : ``a_{n_ctx}`` (3 derniers tokens de la séquence,
      ``action_pos``).

    ``decision_pos`` reste ``ctx_len`` : c'est la fin du contexte (dernier
    token = s_{n_ctx}) ; l'action de décision siège à ``action_pos``.
    """
    assert n_ctx < seq_steps, f"n_ctx ({n_ctx}) must be strictly less than seq_steps ({seq_steps})"
    ctx_len = n_ctx * SALVE_TOKENS + STATE_TOKENS
    eos_pos = ctx_len
    conseq_start = eos_pos
    n_imagine = seq_steps - n_ctx
    conseq_len = (n_imagine - 1) * SALVE_TOKENS + STATE_TOKENS
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
                            n_ctx: int = DEFAULT_N_CTX,
                            lookahead: int | None = None) -> list[int]:
    """Position temporelle (indice de salve) de chaque token de la séquence
    enregistrée, pour l'encodage positionnel sinusoïdal keyed by salve :
    contexte 0..n_ctx, puis conséquence rétro-causale — unité ``j``
    (``k = seq_steps - j``) : les 13 tokens inversés de ``s_k`` à la position
    ``k``, puis ``a_{k-1}`` à la position ``k-1`` ; action de décision
    ``a_{n_ctx}`` à ``n_ctx``.

    ``lookahead`` : la DERNIERE salve de la fenêtre temporelle n'est pas à
    +1 tick de la précédente mais à ``n_ctx + lookahead`` (saut). Défaut
    ``seq_steps - n_ctx`` = comportement historique (dernière salve à
    ``seq_steps``). L'unité j=0 (k = seq_steps dans le layout) prend la vraie
    position ``n_ctx + lookahead``. Le token de dénouement (``eos_pos``) a
    son propre PE dédié ``n_ctx + lookahead + 1`` — hors de la fenêtre
    temporelle, comme un token de synthèse."""
    if lookahead is None:
        lookahead = seq_steps - n_ctx
    layout = compute_seq_layout(seq_steps, n_ctx)
    sp = [0] * layout["seq_len"]
    for i in range(n_ctx):
        for t in range(SALVE_TOKENS):
            sp[i * SALVE_TOKENS + t] = i
    for t in range(STATE_TOKENS):
        sp[n_ctx * SALVE_TOKENS + t] = n_ctx
    for j, k in enumerate(range(seq_steps, n_ctx, -1)):
        pos = (n_ctx + lookahead) if j == 0 else k   # dernière salve : saut +lookahead
        base = layout["conseq_start"] + j * SALVE_TOKENS
        for t in range(STATE_TOKENS):
            sp[base + t] = pos
        if j < layout["n_imagine"] - 1:
            for t in range(ACTION_TOKENS):
                sp[base + STATE_TOKENS + t] = k - 1   # a_{k-1} garde sa vraie position
    # PE dédié du dénouement : hors fenêtre temporelle (n_ctx + lookahead + 1)
    sp[layout["eos_pos"]] = n_ctx + lookahead + 1
    for t in range(ACTION_TOKENS):
        sp[layout["action_pos"] + t] = n_ctx
    return sp


def build_target_tables(seq_steps: int = DEFAULT_SEQ_STEPS,
                        n_ctx: int = DEFAULT_N_CTX) -> tuple[torch.Tensor, torch.Tensor]:
    """Tables (valid, weight) de la loss WM, une par ligne prédictrice p
    (la ligne p prédit le token p+1) :

    - poids 0 sur le contexte (tokens 0..ctx_len-1) : prompt, pas appris ;
    - dénouement (EOS) : poids total ``ctx_len`` — autant que tout le contexte
      en langage naturel (un poids par token de prompt) ;
    - trajectoire (conséquence) et décision (``action_pos..+3``) : poids total
      1.0 chacun, per-slot ``1 / nb_slots_valides_du_groupe`` ;
    - ``valid`` = slots de signal valides du token cible : token de dénouement
      (``eos_pos``) : 6 slots ; décision ``a_{n_ctx}`` (``action_pos..+3``) :
      tokens 13..15 ; conséquence rétro-causale (à partir de ``conseq_start``) :
      unité de 16 tokens (``s_k`` inversée puis ``a_{k-1}``), le token d'état à
      l'offset ``off`` est le token ``STATE_TOKENS-1-off``, l'action à
      l'offset ``13+i`` est le token ``STATE_TOKENS+i``.
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
            off = (q - conseq_start) % SALVE_TOKENS
            tidx = STATE_TOKENS - 1 - off if off < STATE_TOKENS else off
            n = _N_SIGNALS[tidx]
        else:
            n = _N_SIGNALS[STATE_TOKENS + (q - action_pos)]
        valid[p, :n] = True
        weight[p, :n] = 0.0 if q < ctx_len else 1.0
    # Le modèle apprend P(action, conséquence | contexte, dénouement) :
    # contexte et dénouement sont des variables de conditionnement qui valent
    # chacune "1" — le groupe EOS pèse donc 1.0, comme la trajectoire
    # (conséquence) et la décision (action). Per-slot : 1 / nb_slots_valides.
    groups = (slice(eos_pos - 1, eos_pos),                 # dénouement
              slice(eos_pos, action_pos - 1),              # trajectoire
              slice(action_pos - 1, seq_len - 1))          # décision (dernière action)
    for rows in groups:
        # pondération PAR TOKEN (pas par slot) : chaque token du groupe a un
        # poids total 1/n_rows, réparti sur ses slots valides — sinon un token
        # à 16 slots (VISION) écraserait un token à 1 slot (ACT_TAIL)
        n_rows = rows.stop - rows.start
        per_row = valid[rows].float() / valid[rows].sum(dim=1, keepdim=True).clamp(min=1.0)
        weight[rows] = per_row / n_rows
    return valid, weight


# --- dénouement (EOS) ---------------------------------------------------------
_EOS_CANAL = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)   # canal 0 de la modalité EOS dédiée
_EOS_NSLOTS = M.EOS_NSLOTS
_EOS_DECAY = 0.91         # eos_decay : facteur d'actualisation Bellman de l'intégrale EOS.
                          # Pour lookahead=10 : trajet s7..s11 = 3.80, queue infinie de
                          # Bellman sur s16 = decay^10 / (1 - decay) = 4.33 — le dénouement
                          # final domine l'effort transitoire tout en favorisant un geste précoce.
# Poids relatifs W / W_max = [0.125, 1.0, 0.5, 0.5, 0.75, 0.25] pour relief_score en tanh
_EOS_CHANNEL_W_REL = (torch.tensor(_EOS_CHANNEL_W, dtype=torch.float32)
                      / float(max(_EOS_CHANNEL_W)))

# Structural template of the dénouement token: dedicated one-hot modality
# (_M_EOS), dedicated canal, state type, signal slots filled with the 6
# cost-aggregate slots.
_EOS_TEMPLATE = torch.tensor(
    [0.0] + list(_M_EOS) + list(_EOS_CANAL) + [0.0] * N_SIGNAL,
    dtype=torch.float32)


def _eos_slots_of_window(window: list, n_ctx: int = DEFAULT_N_CTX,
                         decay: float = _EOS_DECAY,
                         lookahead: int | None = None) -> list[float]:
    """Les 6 slots du token dénouement d'une fenêtre temporelle de salves
    (seq_steps+1 salves de 16 tokens) : intégrale par canal de l'écart au
    coût de la décision, pondérée par la VRAIE position temporelle :

        r[j] = Σ_{k>d} decay^(pos_k - d) · (c[k][j] - c[d][j]) · p_k

    avec d = n_ctx, pos_k = k sauf pour la dernière salve (k = L-1) dont la
    vraie position est n_ctx + lookahead (saut de la fenêtre : la dernière
    salve n'est pas à +1 tick de la précédente). p_k = 1 sauf dernier état
    où p_k = lookahead : le coût final est compté comme s'il se prolongeait
    sur la durée temporelle de la conséquence (lookahead ticks). Défaut
    lookahead = L-1-d (pas de saut). Slot 5 = inconfort (-confort).
    0.5 = neutre. Squash en tanh : slot[j] = 0.5 - 0.5*tanh(r[j]) — une
    amélioration (r < 0) donne slot > 0.5, cohérent avec relief_score."""
    L = len(window)
    if lookahead is None:
        lookahead = L - 1 - n_ctx
    if L <= n_ctx + 1:
        return [0.5] * _EOS_NSLOTS
    try:
        c = []
        for salve in window:
            row = [salve[_COST_IDX][SIG_OFFSET + j] for j in range(5)]
            row.append(-salve[_REWARD_IDX][SIG_OFFSET])    # inconfort = -confort
            c.append(row)
        out = []
        p_tail = (1.0 / (1.0 - decay)) if decay < 1.0 else float(lookahead)
        for j in range(_EOS_NSLOTS):
            r = 0.0
            for k in range(n_ctx + 1, L):
                pos = (n_ctx + lookahead) if k == L - 1 else k
                p = p_tail if k == L - 1 else 1.0
                r += decay ** (pos - n_ctx) * (c[k][j] - c[n_ctx][j]) * p
            out.append(0.5 - 0.5 * math.tanh(r))
        return out
    except (IndexError, TypeError):
        return [0.5] * _EOS_NSLOTS


def eos_token_from_slots(slots) -> list[float]:
    """Token dénouement (29 floats) depuis 6 slots [0,1] (liste ou tensor)."""
    if hasattr(slots, "tolist"):
        slots = slots.tolist()
    tok = [0.0] + list(_M_EOS) + list(_EOS_CANAL) + list(slots[:_EOS_NSLOTS])
    tok += [0.0] * (N_SIGNAL - _EOS_NSLOTS)
    return tok


def build_sequence(window: list, n_ctx: int = DEFAULT_N_CTX,
                   decay: float = _EOS_DECAY,
                   lookahead: int | None = None) -> list[list[float]]:
    """Réordonne une fenêtre temporelle de ``seq_steps+1`` salves en séquence
    enregistrée (liste plate de ``seq_len`` tokens) :

        contexte | conséquence rétro-causale (incluant l'action de décision)

    - contexte : salves complètes ``s_i+a_i`` (i < n_ctx) puis ``s_{n_ctx}`` seul,
      ordre normal ;
    - conséquence rétro-causale, pour ``k = seq_steps .. n_ctx+1`` : ``s_k``
      avec ses 13 tokens d'état **inversés** (token 12 en premier), puis
      ``a_{k-1}`` (3 tokens, ordre interne normal). Pour ``k = seq_steps``, le
      token 12 est remplacé par le **token de dénouement** calculé depuis la
      fenêtre (ground truth) — premier token de la séquence après le contexte.
    """
    seq_steps = len(window) - 1
    seq: list[list[float]] = []
    for i in range(n_ctx):
        seq.extend(window[i])                              # s_i + a_i
    seq.extend(state_tokens(window[n_ctx]))                # s_{n_ctx}
    eos = eos_token_from_slots(
        _eos_slots_of_window(window, n_ctx=n_ctx, decay=decay, lookahead=lookahead))
    for k in range(seq_steps, n_ctx, -1):                  # conséquence rétro-causale
        s_toks = state_tokens(window[k])
        if k == seq_steps:
            seq.append(eos)                                # token de dénouement (token 12 de s26)
            seq.extend(s_toks[:STATE_TOKENS - 1][::-1])    # tokens 11..0 inversés
        else:
            seq.extend(s_toks[::-1])                       # tokens 12..0 inversés
        seq.extend(action_tokens(window[k - 1]))           # a_{k-1} ; dernier = a_{n_ctx}
    return seq


def sequence_to_window(seq: list, n_ctx: int = DEFAULT_N_CTX,
                       seq_steps: int = DEFAULT_SEQ_STEPS) -> list[list[list[float]]]:
    """Inverse de ``build_sequence`` (pour l'affichage) : reconstruit la fenêtre
    temporelle de ``seq_steps+1`` salves. ``a_{seq_steps}`` n'est pas dans la
    séquence enregistrée : laissée à zéro."""
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
        j = seq_steps - k                                  # unité de la conséquence (k = seq_steps - j)
        base = conseq_start + j * SALVE_TOKENS
        s = [list(t) for t in seq[base:base + STATE_TOKENS][::-1]]   # ré-inverser s_k
        if k == seq_steps:
            # le token 12 de s_{seq_steps} a été remplacé par le token de
            # dénouement à l'enregistrement : intero remis à zéro
            s[_INTERO_IDX][SIG_OFFSET:] = [0.0] * N_SIGNAL
        if k < seq_steps:
            ab = conseq_start + (j - 1) * SALVE_TOKENS + STATE_TOKENS
            a = [list(t) for t in seq[ab:ab + ACTION_TOKENS]]        # a_k dans l'unité de k+1
        else:
            a = [[0.0] * DENSE_DIM] * ACTION_TOKENS
        window.append(s + a)
    return window


def relief_score(x, eos_pos: int | None = None):
    """Score de soulagement : tanh de la somme pondérée des écarts à 0.5
    (``tanh(Σ (W_j / W_max) · (2·slot_j - 1))``). Symétrique autour de 0.5 = 0,
    strictement croissant sur chaque canal (départage même en saturation),
    borné dans (-1, +1). Accepte un tensor de slots [..., 6] ou une séquence
    enregistrée (liste plate : le token dénouement est lu à ``eos_pos``)."""
    if isinstance(x, torch.Tensor):
        slots = x
    else:
        if eos_pos is None:
            eos_pos = compute_seq_layout()["eos_pos"]
        slots = torch.tensor(x[eos_pos][SIG_OFFSET:SIG_OFFSET + _EOS_NSLOTS],
                             dtype=torch.float32)
    w = _EOS_CHANNEL_W_REL.to(slots.device)
    d = 2.0 * slots - 1.0
    return torch.tanh((w * d).sum(dim=-1))


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
_ACT_SIGNED = torch.tensor(
    [_BY_KEY[k].first_val < 0 < _BY_KEY[k].last_val for _, _, k in _ACT_LAYOUT],
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

# Precomputed cost metadata for the vectorized salve cost (affine first + t * span).
_COST_FIRST = torch.tensor([_BY_KEY[k].first_val for k in _COST_KEYS], dtype=torch.float32)
_COST_SPAN = torch.tensor(
    [_BY_KEY[k].last_val - _BY_KEY[k].first_val for k in _COST_KEYS], dtype=torch.float32)
_COST_W = torch.tensor([_COST_WEIGHTS[k] for k in _COST_KEYS], dtype=torch.float32)
_CONFORT_FIRST = _BY_KEY["confort"].first_val
_CONFORT_SPAN = _BY_KEY["confort"].last_val - _BY_KEY["confort"].first_val
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
                 eos_decay: float = _EOS_DECAY,
                 lookahead: int | None = None,
                 dedup_eps: float = 0.02,
                 vivacity_decay: float = 0.95,
                 impact_floor: float = 0.10):
        self.seq_len = seq_len                      # salves per temporal window
        self.capacity = capacity
        self.pool_capacity = pool_capacity
        self.seq_steps = seq_len - 1
        if n_ctx is None:
            self.n_ctx = min(DEFAULT_N_CTX, max(1, self.seq_steps // 2))
        else:
            self.n_ctx = n_ctx
        self.eos_decay = eos_decay
        self.dedup_eps = dedup_eps
        self.vivacity_decay = vivacity_decay
        self.impact_floor = impact_floor
        # saut de la fenêtre : la dernière salve est à +lookahead de la décision
        # (défaut seq_steps - n_ctx = pas de saut, dernière salve à +1 tick)
        self.lookahead = lookahead if lookahead is not None else self.seq_steps - self.n_ctx
        # portée réelle d'une fenêtre dans le journal : au moins seq_steps
        # salves consécutives ET la dernière salve à n_ctx + lookahead
        self.window_span = max(self.seq_steps, self.n_ctx + self.lookahead + 1)
        self._layout = compute_seq_layout(self.seq_steps, self.n_ctx)
        # Masque bool [seq_len_toks, N_SIGNAL] des slots signaux valides sur
        # TOUTE la séquence enregistrée (contexte + dénouement + conséquence
        # inversée + action de décision, hors tokens intero masqués et padding),
        # avec pondération PAR TOKEN (1 / sqrt(N_tok * n_slots_tok)) pour que
        # VISION (16 slots) ne pèse pas 16x plus qu'ACT_TAIL (1 slot).
        sig_mask = torch.zeros(self._layout["seq_len"], N_SIGNAL, dtype=torch.bool)
        tok_slot_counts: list[int] = []
        ctx_len = self._layout["ctx_len"]
        eos_pos = self._layout["eos_pos"]
        conseq_start = self._layout["conseq_start"]
        action_pos = self._layout["action_pos"]
        for q in range(self._layout["seq_len"]):
            if q == eos_pos:
                sig_mask[q, :_EOS_NSLOTS] = True
                tok_slot_counts.append(_EOS_NSLOTS)
                continue
            if q < ctx_len:
                tidx = q % SALVE_TOKENS
            elif q < action_pos:
                off = (q - conseq_start) % SALVE_TOKENS
                tidx = STATE_TOKENS - 1 - off if off < STATE_TOKENS else off
            else:
                tidx = STATE_TOKENS + (q - action_pos)
            if tidx == _INTERO_IDX:
                continue
            n_s = _N_SIGNALS[tidx]
            sig_mask[q, :n_s] = True
            tok_slot_counts.append(n_s)
        n_tok = max(1, len(tok_slot_counts))
        weights = []
        for n_s in tok_slot_counts:
            w = 1.0 / math.sqrt(n_tok * n_s)
            weights.extend([w] * n_s)
        self._sig_mask = sig_mask
        self._sig_weights = torch.tensor(weights, dtype=torch.float32)
        self.last_evicted = 0
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
            "count": 1,
            "learn_count": 0,
            "relief_score": None,
            "surprise": None,
        }

    @property
    def _valid_segments(self) -> list[list[list[float]]]:
        segs = [s for s in self._segments if len(s) >= self.window_span]
        if len(self._current) >= self.window_span:
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
        total = sum(self._n_windows(s) for s in self._segments)
        total += self._n_windows(self._current)
        return total

    def __len__(self) -> int:
        return len(self._coreset) + len(self._addendum) + self.journal_len

    def push(self, salve: list[list[float]]) -> None:
        """Append one temporal salve to the current active segment."""
        self._current.append(salve)
        self._enforce_capacity()

    def boundary(self) -> None:
        """Seal the active segment (discontinuity: flip, reset, teleport)."""
        if len(self._current) >= self.window_span:
            self._segments.append(self._current)
        self._current = []

    def new_segment(self) -> None:
        self.boundary()

    def _n_windows(self, seg) -> int:
        """Number of extractable windows in a segment: the window needs
        ``window_span`` = max(seq_steps, n_ctx + lookahead + 1) salves."""
        return max(0, len(seg) - self.window_span + 1)

    def _window_at(self, seg, offset: int) -> list:
        """Temporal window of seq_len salves at ``offset`` : seq_steps
        consecutive salves (s0..s_{seq_steps-1}) plus the LAST salve taken
        at +lookahead ticks after the decision (s_{n_ctx+lookahead}) — the
        window jumps: close future + far denouement."""
        win = [seg[offset + i] for i in range(self.seq_steps)]
        win.append(seg[offset + self.n_ctx + self.lookahead])
        return win

    def extract_addendum(self, stride: int = 1) -> int:
        """Cut sliding windows from the journal segments and store them in the
        Addendum as recorded sequences (``build_sequence``). Clears the journal."""
        added = 0
        for seg in self._valid_segments:
            n_windows = self._n_windows(seg)
            if n_windows <= 0:
                continue
            offsets = list(range(0, n_windows, stride))
            if (n_windows - 1) % stride != 0:
                offsets.append(n_windows - 1)
            for offset in offsets:
                window = self._window_at(seg, offset)
                self._addendum.append(build_sequence(
                    window, n_ctx=self.n_ctx, decay=self.eos_decay,
                    lookahead=self.lookahead))
                self._addendum_meta.append(self._default_meta("wake"))
                added += 1
        self._segments.clear()
        self._current.clear()
        return added

    def _seq_sig_matrix(self, seqs: list, device=None) -> torch.Tensor:
        """Flatten all valid signal slots of full recorded sequences (context +
        dénouement + reversed consequence + decision action, excluding masked
        intero & padding) weighted equally per token into a [N, D] float32
        matrix so ``torch.cdist(A, B)`` is the per-token weighted RMSE."""
        t = torch.tensor(seqs, dtype=torch.float32, device=device)  # [N, L, 29]
        mask = self._sig_mask.to(t.device)                          # [L, 16]
        w = self._sig_weights.to(t.device)                          # [D]
        return t[:, :, SIG_OFFSET:][:, mask] * w                    # [N, D]

    def _action_and_relief(self, seqs: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract normalized decision actions [N, 5] in [0, 1] and relief
        scores [N] from recorded sequences (CPU tensors)."""
        ap = self._layout["action_pos"]
        ep = self._layout["eos_pos"]
        acts = [[s[ap][SIG_OFFSET], s[ap][SIG_OFFSET + 1],
                 s[ap + 1][SIG_OFFSET], s[ap + 1][SIG_OFFSET + 1],
                 s[ap + 2][SIG_OFFSET]] for s in seqs]
        eos = [s[ep][SIG_OFFSET:SIG_OFFSET + _EOS_NSLOTS] for s in seqs]
        a_t = torch.tensor(acts, dtype=torch.float32)
        r_t = relief_score(torch.tensor(eos, dtype=torch.float32))
        return a_t, r_t

    def _pairwise_dists(self, seqs_a: list, seqs_b: list | None = None,
                        embed_fn=None, device=None) -> torch.Tensor:
        """Pairwise distance matrix [Na, Nb] between full recorded sequences:
        - if ``embed_fn`` is given (converged WM full-sequence latents, unit
          normalized): cosine distance ``1 - A @ B.T`` in [0, 2];
        - otherwise (cold-start fallback): per-token weighted RMSE over all
          valid signal slots of the full sequence."""
        if embed_fn is not None:
            a = embed_fn(seqs_a)
            b = a if seqs_b is None else embed_fn(seqs_b)
            return (1.0 - a @ b.T).clamp(min=0.0)
        a = self._seq_sig_matrix(seqs_a, device=device)
        b = a if seqs_b is None else self._seq_sig_matrix(seqs_b, device=device)
        return torch.cdist(a, b)

    def deduplicate_addendum(self, eps: float | None = None,
                             embed_fn=None, device=None,
                             act_eps: float = 0.05,
                             relief_eps: float = 0.08) -> int:
        """Drop near-duplicate sequences within the Addendum using full-sequence
        distance (WM latent cosine distance when ``embed_fn`` is provided,
        otherwise per-token weighted RMSE). Two sequences are never merged if
        their decision action differs by more than ``act_eps`` on any actuator
        or if their relief differs by more than ``relief_eps`` (protects
        action-initiation frames and counterfactual pairs). Among true
        duplicates, keeps the representative with highest relief and refreshes
        its count/vivacity. Returns the number dropped."""
        thr = self.dedup_eps if eps is None else eps
        n = len(self._addendum)
        if thr <= 0.0 or n <= 1:
            return 0
        dists = self._pairwise_dists(self._addendum, embed_fn=embed_fn, device=device).cpu()
        acts, rels = self._action_and_relief(self._addendum)
        act_d = (acts.unsqueeze(1) - acts.unsqueeze(0)).abs().amax(dim=-1)   # [n, n]
        rel_d = (rels.unsqueeze(1) - rels.unsqueeze(0)).abs()                # [n, n]
        # Parcourir du meilleur relief au moins bon : entre vrais doublons, le
        # représentant au meilleur dénouement est toujours conservé.
        order = sorted(range(n), key=lambda idx: float(rels[idx]), reverse=True)
        kept_idx: list[int] = []
        for i in order:
            dup_of = -1
            best_d = thr
            row = dists[i]
            arow = act_d[i]
            rrow = rel_d[i]
            for j in kept_idx:
                if float(arow[j]) > act_eps or float(rrow[j]) > relief_eps:
                    continue
                d = float(row[j])
                if d < best_d:
                    best_d = d
                    dup_of = j
            if dup_of >= 0:
                m = self._addendum_meta[dup_of]
                m["vivacity"] = 1.0
                m["count"] = m.get("count", 1) + self._addendum_meta[i].get("count", 1)
            else:
                kept_idx.append(i)
        dropped = n - len(kept_idx)
        if dropped > 0:
            kept_idx.sort()
            self._addendum = [self._addendum[i] for i in kept_idx]
            self._addendum_meta = [self._addendum_meta[i] for i in kept_idx]
        return dropped

    def deduplicate_against_coreset(self, eps: float | None = None,
                                    embed_fn=None, device=None,
                                    act_eps: float = 0.05,
                                    relief_eps: float = 0.08) -> int:
        """Drop Addendum sequences within ``eps`` (full-sequence latent cosine
        or per-token RMSE) of an existing Coreset memory whose decision action
        and relief also match within ``act_eps`` / ``relief_eps``. Refreshes
        the matching Coreset memory's vivacity and count. Returns number dropped."""
        thr = self.dedup_eps if eps is None else eps
        if thr <= 0.0 or not self._addendum or not self._coreset:
            return 0
        dists = self._pairwise_dists(
            self._addendum, self._coreset, embed_fn=embed_fn, device=device).cpu()
        acts_a, rels_a = self._action_and_relief(self._addendum)
        acts_c, rels_c = self._action_and_relief(self._coreset)
        act_d = (acts_a.unsqueeze(1) - acts_c.unsqueeze(0)).abs().amax(dim=-1)
        rel_d = (rels_a.unsqueeze(1) - rels_c.unsqueeze(0)).abs()
        incompat = (act_d > act_eps) | (rel_d > relief_eps)
        dists = dists.masked_fill(incompat, float("inf"))
        min_d, nearest = dists.min(dim=1)
        min_d_list = min_d.tolist()
        nearest_list = nearest.tolist()
        kept_seqs: list = []
        kept_metas: list = []
        dropped = 0
        for i, (d, j) in enumerate(zip(min_d_list, nearest_list)):
            if d < thr:
                self._coreset_vivacity[j] = 1.0
                if j < len(self._coreset_meta):
                    self._coreset_meta[j]["vivacity"] = 1.0
                    self._coreset_meta[j]["count"] = (
                        self._coreset_meta[j].get("count", 1)
                        + self._addendum_meta[i].get("count", 1))
                dropped += 1
            else:
                kept_seqs.append(self._addendum[i])
                kept_metas.append(self._addendum_meta[i])
        if dropped > 0:
            self._addendum = kept_seqs
            self._addendum_meta = kept_metas
        return dropped

    def commit_addendum(self, embed_fn=None, device=None) -> int:
        """Consolidate Addendum into the Coreset with gentle vivacity decay and
        density-based capacity eviction (drops from the closest full-sequence
        pairs, keeping the higher-priority member so rare old memories survive).
        Returns the number of Addendum sequences committed."""
        if self._coreset and self.vivacity_decay < 1.0:
            for i in range(len(self._coreset_vivacity)):
                self._coreset_vivacity[i] *= self.vivacity_decay
                if i < len(self._coreset_meta):
                    self._coreset_meta[i]["vivacity"] = self._coreset_vivacity[i]
        n = len(self._addendum)
        self._coreset.extend(self._addendum)
        self._coreset_vivacity.extend([1.0] * n)
        self._coreset_meta.extend(self._addendum_meta)
        self._addendum.clear()
        self._addendum_meta.clear()

        excess = len(self._coreset) - self.pool_capacity
        self.last_evicted = max(0, excess)
        if excess > 0:
            m = len(self._coreset)
            dists = self._pairwise_dists(self._coreset, embed_fn=embed_fn, device=device)
            dists.fill_diagonal_(float("inf"))
            eos_pos = self._layout["eos_pos"]
            priorities = [
                max(abs(float(relief_score(seq, eos_pos=eos_pos))), self.impact_floor)
                * (0.5 + 0.5 * viv)
                for seq, viv in zip(self._coreset, self._coreset_vivacity)
            ]
            evicted: set[int] = set()
            for _ in range(excess):
                flat = int(dists.argmin().item())
                i, j = divmod(flat, m)
                victim, survivor = (i, j) if priorities[i] <= priorities[j] else (j, i)
                # Transfère le count de la victime vers son plus proche voisin survivant
                if survivor < len(self._coreset_meta) and victim < len(self._coreset_meta):
                    self._coreset_meta[survivor]["count"] = (
                        self._coreset_meta[survivor].get("count", 1)
                        + self._coreset_meta[victim].get("count", 1))
                evicted.add(victim)
                dists[victim, :] = float("inf")
                dists[:, victim] = float("inf")
            keep = [k for k in range(m) if k not in evicted]
            self._coreset = [self._coreset[k] for k in keep]
            self._coreset_vivacity = [self._coreset_vivacity[k] for k in keep]
            self._coreset_meta = [self._coreset_meta[k] for k in keep]
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
            "lookahead": self.lookahead,
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
            removable = self._n_windows(s0)
            if excess >= removable:
                self._segments.popleft()
                excess -= removable
            else:
                self._segments[0] = s0[excess:]
                excess = 0
        if excess > 0 and len(self._current) >= self.window_span:
            removable = self._n_windows(self._current)
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
        n_core = len(self._coreset)
        pool = list(self._coreset) + self._addendum
        if pool:
            idxs = self._rng.sample(range(len(pool)), min(batch, len(pool)))
            self.last_sample_tags = [
                f"#{i}" if i < n_core else f"add #{i - n_core}" for i in idxs
            ]
            return [pool[i] for i in idxs]
        valid_segs = self._valid_segments
        if not valid_segs:
            self.last_sample_tags = []
            return []
        counts = [self._n_windows(s) for s in valid_segs]
        total = sum(counts)
        if total <= 0:
            self.last_sample_tags = []
            return []
        indices = self._rng.sample(range(total), min(batch, total))
        self.last_sample_tags = [f"j#{idx}" for idx in indices]
        cum = list(itertools.accumulate(counts))
        out = []
        for idx in indices:
            seg_i = next(i for i, c in enumerate(cum) if idx < c)
            offset = idx - (cum[seg_i - 1] if seg_i > 0 else 0)
            seg = valid_segs[seg_i]
            out.append(build_sequence(self._window_at(seg, offset),
                                      n_ctx=self.n_ctx, decay=self.eos_decay,
                                      lookahead=self.lookahead))
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
                 lookahead: int | None = None,
                 d_model: int = 64, nhead: int = 4, layers: int = 3,
                 dim_ff: int = 384, dropout: float = 0.0,
                 pol_hidden: int = 256,
                 buffer_capacity: int = 1000, buffer_pool_capacity: int = 1000,
                 dedup_eps: float = 0.02,
                 dedup_latent_eps: float = 0.02,
                 vivacity_decay: float = 0.95,
                 eos_decay: float = _EOS_DECAY,
                 # tête MDN du dénouement + élection sur les K modes
                 eos_n_modes: int = M.EOS_N_MODES,
                 eos_sigma_min: float = M.EOS_SIGMA_MIN,
                 eos_nll_w: float = 0.1,
                 eos_temperature: float = 1.0,
                 eos_top_n: int | None = None,
                 eos_top_p: float = 0.9,
                 eos_min_p: float = 0.10,
                 # goulot de situation (VQ) devant la tête MDN
                 eos_vq_codes: int = M.EOS_VQ_CODES,
                 eos_vq_dim: int = M.EOS_VQ_DIM,
                 eos_vq_beta: float = 0.25,
                 eos_vq_decay: float = 0.99,
                 eos_vq_act_w: float = 1.0,
                 wm_input_noise: float = 0.015,
                 wm_impact_floor: float = 0.10,
                 act_conseq_shift: int = 0):
        torch.manual_seed(seed)
        self.device, self.device_desc = M.configure_hardware(device)
        if self.device.type == "cpu":
            torch.set_num_threads(min(2, torch.get_num_threads()))
        self.seq_steps = seq_steps
        self.n_ctx = n_ctx
        self.lookahead = lookahead if lookahead is not None else seq_steps - n_ctx
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
                                  dim_ff=dim_ff, dropout=dropout,
                                  eos_n_modes=eos_n_modes,
                                  eos_sigma_min=eos_sigma_min,
                                  eos_vq_codes=eos_vq_codes,
                                  eos_vq_dim=eos_vq_dim,
                                  eos_vq_decay=eos_vq_decay).to(self.device)
        self.policy = M.Policy(n_state=N_POLICY_STATE + d_model,
                               hidden=pol_hidden).to(self.device)
        self.opt_wm = torch.optim.Adam(self.world.parameters(), lr=lr_wm)
        self.opt_pol = torch.optim.Adam(self.policy.parameters(), lr=lr_pol)
        self.buffer = ExperienceBuffer(seq_len=seq_steps + 1, seed=seed, n_ctx=n_ctx,
                                       eos_decay=eos_decay, lookahead=self.lookahead,
                                       capacity=buffer_capacity,
                                       pool_capacity=buffer_pool_capacity,
                                       dedup_eps=dedup_eps,
                                       vivacity_decay=vivacity_decay,
                                       impact_floor=wm_impact_floor)
        # Positions temporelles (PE sinusoïdal) et tables de loss du layout.
        self.salve_positions = torch.tensor(
            compute_salve_positions(seq_steps, n_ctx, lookahead=self.lookahead),
            dtype=torch.long, device=self.device)
        self.wm_impact_floor = wm_impact_floor
        self.dedup_latent_eps = dedup_latent_eps
        self.mask = build_salve_mask(self.seq_len, self.device)
        t_valid, t_weight = build_target_tables(seq_steps, n_ctx)
        self.target_valid = t_valid.to(self.device)
        self.target_weight = t_weight.to(self.device)
        # Masque [1, seq_len, DENSE_DIM] pour wm_input_noise : bruit uniquement
        # sur les slots signaux valides des capteurs/actions (0 sur préfixe,
        # padding, tokens intero masqués, et token dénouement eos_pos).
        nmask = torch.zeros(self.seq_len, DENSE_DIM, dtype=torch.float32)
        for q in range(self.seq_len):
            if q == self.eos_pos:
                continue
            if q < self.ctx_len:
                tidx = q % SALVE_TOKENS
            elif q < self.action_pos:
                off = (q - self.conseq_start) % SALVE_TOKENS
                tidx = STATE_TOKENS - 1 - off if off < STATE_TOKENS else off
            else:
                tidx = STATE_TOKENS + (q - self.action_pos)
            if tidx == _INTERO_IDX:
                continue
            nmask[q, SIG_OFFSET:SIG_OFFSET + _N_SIGNALS[tidx]] = 1.0
        self.input_noise_mask = nmask.unsqueeze(0).to(self.device)
        self._eos_tmpl = _EOS_TEMPLATE.to(self.device)
        self.eos_decay = eos_decay
        self.eos_n_modes = eos_n_modes
        self.eos_sigma_min = eos_sigma_min
        self.eos_nll_w = eos_nll_w
        self.eos_vq_beta = eos_vq_beta
        self.eos_vq_act_w = eos_vq_act_w
        self.eos_temperature = eos_temperature
        self.eos_top_n = eos_top_n
        self.eos_top_p = eos_top_p
        self.eos_min_p = eos_min_p
        self.wm_input_noise = wm_input_noise
        self._wm_loss = (torch.compile(self._wm_loss_impl, dynamic=False)
                         if compile_wm else self._wm_loss_impl)
        self.act_std = act_std
        self._cost_first = _COST_FIRST.to(self.device)
        self._cost_span = _COST_SPAN.to(self.device)
        self._cost_w = _COST_W.to(self.device)
        self._pol_idx = torch.tensor(_POL_SCALAR_IDX, dtype=torch.long, device=self.device)
        self._act_tok = _ACT_TOK.to(self.device)
        self._act_slot = _ACT_SLOT.to(self.device)
        self._act_signed = _ACT_SIGNED.to(self.device)
        self._act_tmpl = _ACT_TEMPLATE.to(self.device)
        self._wake_history: list[tuple[list, list]] = []
        # last sleep stats, for the HUD
        self.last_wm_loss = float("nan")
        self.last_batch_mse = float("nan")
        self.last_batch_nll = float("nan")
        self.last_batch_commit = float("nan")
        self.last_batch_vq_act = float("nan")
        self._last_use_obs: torch.Tensor | None = None
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
        self.sleep_wm_slingshot = False
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
        # est (s_{n-k} ; a_n) — 0 = brain-on.
        self.act_conseq_shift = act_conseq_shift

    @staticmethod
    def _normalize_latent(raw: torch.Tensor) -> torch.Tensor:
        """Parameter-free LayerNorm over the WM mid-layer residual stream."""
        return torch.nn.functional.layer_norm(raw, (raw.shape[-1],))

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
        L = ctx.shape[1]
        self.world.eval()
        t0 = time.perf_counter()
        self.world(ctx, attn_mask=build_salve_mask(L, self.device),
                   is_causal=True, salve_positions=self.salve_positions[:L])
        latent = self._normalize_latent(self.world.last_latent())
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
        brain-off action/consequence shift k when act_conseq_shift > 0:
        recorded pair (s_{n-k}, a_n) — state read k ticks back in
        _wake_history, current action kept; called from record_tick, so
        _wake_history[-1] is the current salve)."""
        if hasattr(salve, "tolist"):
            salve = salve.tolist()
        k = self.act_conseq_shift
        if k > 0 and len(self._wake_history) > k:
            s_old = self._wake_history[-k - 1][0]
            a_cur = self._wake_history[-1][1]
            salve = [list(t) for t in s_old] + [list(t) for t in a_cur]
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
            if (len(self.buffer._current) >= self.buffer.window_span
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
        """Build a [1, L, 29] context tensor from the wake history + current
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
        """Stack recorded sequences (flat token lists) into [B, seq_len, 29].
        Zeroes the interoception slots of every state salve (the carrier of the
        outcome signal stays masked from the WM inputs) — except the token de
        dénouement at ``eos_pos``, which stays visible as a WM input (teacher
        forcing of the rétro-causal consequence)."""
        B = len(sequences)
        vals = torch.zeros(B, self.seq_len, DENSE_DIM, device=self.device)
        for b, seq in enumerate(sequences):
            assert len(seq) == self.seq_len, \
                f"recorded sequence has {len(seq)} tokens, expected {self.seq_len}"
            vals[b] = torch.tensor(seq, dtype=torch.float32, device=self.device)
        # intero slots: contexte (salves 0..n_ctx) + conséquence rétro-causale
        # (offset 0 de chaque unité = token 12 inversé), sauf eos_pos (unité 0)
        intero_pos = [k * SALVE_TOKENS + _INTERO_IDX for k in range(self.n_ctx + 1)]
        intero_pos += [self.conseq_start + j * SALVE_TOKENS
                       for j in range(1, self.n_imagine)]
        vals[:, intero_pos, SIG_OFFSET:] = 0.0
        return vals

    # --- WM training -----------------------------------------------------------
    def _wm_masked_mse(self, pred: torch.Tensor, vals: torch.Tensor,
                       seq_weights: torch.Tensor | None = None) -> torch.Tensor:
        """Masked MSE over valid signal slots. The target table weights 0 the
        context rows (prompt) and 1.0 every row from the dénouement on.
        ``seq_weights`` (optionnel) pondère chaque séquence du batch — utilisé
        pour l'impact-proportional training : poids = max(|relief(seq)|, floor)."""
        pred_signals = pred[:, :-1, :]                   # [B, L-1, 16]
        tgt_signals = vals[:, 1:, SIG_OFFSET:]           # [B, L-1, 16]
        w = self.target_weight.unsqueeze(0).expand_as(pred_signals)
        sq = (pred_signals - tgt_signals) ** 2
        sqm = (sq * w).sum(dim=(1, 2))                   # [B]
        wsum = w.sum(dim=(1, 2)).clamp(min=1.0)          # [B]
        per_seq = sqm / wsum                             # [B]
        if seq_weights is None:
            return per_seq.mean()
        sw = seq_weights.to(per_seq.device)
        return (per_seq * sw).sum() / sw.sum().clamp(min=1e-8)

    def _wm_loss_impl(self, vals: torch.Tensor,
                      seq_weights: torch.Tensor | None = None,
                      nll_weights: torch.Tensor | None = None
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One forward, two losses over the whole recorded sequence (causal
        mask + salve positions):
        - MSE on all tokens from the dénouement on (signal slots, weighted by
          ``seq_weights`` = impact);
        - exact shifted non-negative MDN NLL (per slot) on the 6 continuous
          dénouement slots predicted at row ``eos_pos - 1`` (weighted only by
          ``nll_weights`` = log1p(count), never by impact, to preserve
          calibrated frequencies P(EOS | ctx)).
        Returns ``(total, mse, nll, commit)``: optimization backprops ``total``,
        while convergence checks (``wm_target_loss``, plateau, latent dedup)
        track ``mse`` so the irreducible multimodal NLL entropy floor cannot
        pin the phase in overtime.
        Le token dénouement reste inconditionnellement dans l'entrée (teacher
        forcing) : la conséquence est conditionnée par contexte ET dénouement."""
        # data augmentation : bruit léger sur les seuls slots signaux valides
        # (préfixe, padding, tokens intero masqués et dénouement eos_pos intacts) ;
        # la cible de la loss reste ``vals`` propre
        vals_in = vals
        if self.wm_input_noise > 0.0:
            noise = torch.randn_like(vals) * (self.wm_input_noise * self.input_noise_mask)
            vals_in = (vals + noise).clamp(0.0, 1.0)
        tgt = vals[:, self.eos_pos, SIG_OFFSET:SIG_OFFSET + _EOS_NSLOTS]    # [B, 6]
        pred, (logits_pi, mu, sigma) = self.world.forward_with_eos(
            vals_in, attn_mask=self.mask, is_causal=True,
            salve_positions=self.salve_positions, eos_row=self.eos_pos - 1,
            eos_cond=tgt, eos_pos=self.eos_pos)
        mse = self._wm_masked_mse(pred, vals, seq_weights)
        # MDN negative log-likelihood shifted by its theoretical minimum at
        # sigma = sigma_min (so nll >= 0 always, 0 iff exact Dirac on tgt),
        # normalized per EOS slot.
        log_pi = torch.nn.functional.log_softmax(logits_pi, dim=-1)         # [B, K]
        diff = (tgt.unsqueeze(1) - mu) / sigma                              # [B, K, 6]
        log_ratio = torch.log(sigma / self.world.eos_sigma_min)             # [B, K, 6] >= 0
        log_comp = -0.5 * (diff * diff + 2.0 * log_ratio).sum(dim=-1)       # [B, K] <= 0
        nll = -torch.logsumexp(log_pi + log_comp, dim=-1) / _EOS_NSLOTS     # [B] >= 0
        if nll_weights is None:
            nll_loss = nll.mean()
        else:
            nw = nll_weights.to(nll.device)
            nll_loss = (nll * nw).sum() / nw.sum().clamp(min=1e-8)
        # Goulot de situation (VQ) :
        # - perte d'engagement : z_e reste proche de son code (codebook EMA) ;
        # - critère d'action de décision : (z_q, EOS) -> a_{n_ctx} (5 scalaires
        #   normalisés [0,1]), force le VQ à séparer les situations qui appellent
        #   des actions différentes même quand EOS == 0.5 (debout vs couché).
        vq = self.world.last_vq
        if vq is not None:
            z_e, idx, z_q = vq
            commit = ((z_e - self.world.eos_vq_codebook[idx].detach()) ** 2).mean()
            act_flat = vals[:, self.action_pos:self.action_pos + ACTION_TOKENS,
                            SIG_OFFSET:SIG_OFFSET + N_SIGNAL].reshape(
                vals.shape[0], ACTION_TOKENS * N_SIGNAL)
            act_tgt = act_flat.index_select(1, self._act_tok * N_SIGNAL + self._act_slot)
            act_pred = self.world.predict_vq_action(z_q, tgt)
            vq_act = ((act_pred - act_tgt) ** 2).mean()
        else:
            commit = torch.zeros((), dtype=mse.dtype, device=mse.device)
            vq_act = torch.zeros((), dtype=mse.dtype, device=mse.device)
        total = (mse
                 + self.eos_nll_w * (nll_loss + self.eos_vq_beta * commit)
                 + self.eos_vq_act_w * vq_act)
        return total, mse, nll_loss, commit, vq_act

    def _seq_impact(self, seq: list) -> float:
        """Impact d'une séquence enregistrée (ADR-0001) : |relief|, plancher
        wm_impact_floor."""
        rel = abs(float(relief_score(seq, eos_pos=self.eos_pos)))
        return max(rel, self.wm_impact_floor)

    def _train_wm_batch(self, sequences, counts: list[int] | None = None) -> float:
        """One WM optimization step on a batch of recorded sequences.
        - MSE is weighted by ``_seq_impact`` (impact-proportional training);
        - MDN NLL is weighted by ``log1p(count)`` when ``counts`` are given
          (reflects deduplicated multiplicity without letting one static point
          dominate a batch).
        Backpropagates ``total = mse + eos_nll_w * (nll + beta * commit)`` and
        returns the masked ``mse`` (comparable to ``wm_target_loss`` without the
        irreducible multimodal NLL entropy floor)."""
        vals = self._batch_tensors(sequences)
        impacts = [self._seq_impact(s) for s in sequences]
        seq_w = torch.tensor(impacts, dtype=torch.float32, device=self.device)
        nll_w = (torch.tensor([math.log1p(max(1, int(c))) for c in counts],
                              dtype=torch.float32, device=self.device)
                 if counts is not None else None)
        total, mse, nll, commit, vq_act = self._wm_loss(vals, seq_w, nll_w)
        self.opt_wm.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.world.parameters(), 1.0)
        self.opt_wm.step()
        vq = self.world.last_vq
        if vq is not None:
            self.world.vq_ema_update(vq[0], vq[1])
        self.world.last_vq = None
        self.last_batch_mse = float(mse.item())
        self.last_batch_nll = float(nll.item())
        self.last_batch_commit = float(commit.item())
        self.last_batch_vq_act = float(vq_act.item())
        return self.last_batch_mse

    def train_wm_phase(self, epochs: int, batch: int,
                       target_loss: float | None = None,
                       source: str = "coreset",
                       label: str = "wm",
                       fixed_seq=None,
                       slingshot_ratio: float = 1.30,
                       slingshot_settle: int = 3,
                       slingshot_max_extra: int | None = None):
        """One WM training phase as a generator yielding ``(label, epoch, loss)``
        after each batch. 1 epoch = the whole dataset shuffled and swept in
        batches without omission or replacement. ``fixed_seq`` (a sequence or
        a callable) is prepended to every batch.

        Slingshot guard: if Adam triggers a sudden loss spike above
        ``slingshot_ratio * best_smooth`` (or ``slingshot_ratio * prev_loss``),
        stopping is deferred past ``epochs`` until the loss settles back within
        the ratio for ``slingshot_settle`` consecutive epochs (capped at
        ``slingshot_max_extra`` extra epochs)."""
        buf = self.buffer
        seqs_all, metas_all = ((buf._addendum, buf._addendum_meta) if source == "addendum"
                               else (buf._coreset, buf._coreset_meta))
        n = len(seqs_all)
        if n == 0 and fixed_seq is None:
            return
        self.world.train()
        self.sleep_wm_epochs = epochs
        self.sleep_wm_slingshot = False
        order = list(range(n))
        ep = 0
        smooth_loss: float | None = None
        best_smooth = float("inf")
        prev_loss: float | None = None
        in_slingshot = False
        sling_ref = float("inf")
        calm_streak = 0
        extra_cap = (slingshot_max_extra if slingshot_max_extra is not None
                     else max(64, epochs // 2))
        extra_used = 0
        while True:
            ep += 1
            self.sleep_wm_step = ep
            mix_coreset = (source == "addendum" and len(buf._coreset) > 0)
            step_batch = max(1, batch // 2) if mix_coreset else batch
            buf._rng.shuffle(order)
            self.sleep_wm_n = n
            self.sleep_wm_batches = max(1, (n + step_batch - 1) // step_batch)
            self.sleep_wm_batch = 0
            epoch_losses = []
            skip = False
            for start in range(0, max(n, 1), step_batch):
                idxs = order[start:start + step_batch]
                self.sleep_wm_batch = start // step_batch + 1
                batch_seqs = [seqs_all[i] for i in idxs]
                batch_counts = [
                    metas_all[i].get("count", 1) if i < len(metas_all) else 1
                    for i in idxs
                ]
                if mix_coreset and batch_seqs:
                    # Replay mixte 50/50 addendum + coreset : évite l'oubli
                    # catastrophique du coreset pendant la phase addendum.
                    n_replay = min(len(buf._coreset), len(batch_seqs))
                    r_idxs = buf._rng.sample(range(len(buf._coreset)), n_replay)
                    batch_seqs = batch_seqs + [buf._coreset[r] for r in r_idxs]
                    batch_counts = batch_counts + [
                        buf._coreset_meta[r].get("count", 1)
                        if r < len(buf._coreset_meta) else 1
                        for r in r_idxs
                    ]
                if fixed_seq is not None:
                    fs = fixed_seq() if callable(fixed_seq) else fixed_seq
                    if fs is not None:
                        batch_seqs = [fs] + batch_seqs
                        batch_counts = [1] + batch_counts
                if not batch_seqs:
                    if n == 0:
                        self.sleep_wm_slingshot = False
                        return  # source vide et fixed_seq None : rien à entraîner
                    continue
                loss_c = self._train_wm_batch(batch_seqs, counts=batch_counts)
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
                self.sleep_wm_slingshot = False
                return
            for m in metas_all:
                m["learn_count"] = m.get("learn_count", 0) + 1
            loss_val = (sum(epoch_losses) / len(epoch_losses)) if epoch_losses else float("nan")
            if not math.isnan(loss_val):
                smooth_loss = (loss_val if smooth_loss is None
                               else 0.4 * loss_val + 0.6 * smooth_loss)
                if smooth_loss < best_smooth:
                    best_smooth = smooth_loss
                ref = sling_ref if in_slingshot else best_smooth
                spiking = (loss_val > ref * slingshot_ratio) or (
                    prev_loss is not None and loss_val > prev_loss * slingshot_ratio)
                if spiking:
                    if not in_slingshot:
                        sling_ref = min(best_smooth, prev_loss) if prev_loss is not None else best_smooth
                    in_slingshot = True
                    calm_streak = 0
                elif in_slingshot:
                    calm_streak += 1
                    if calm_streak >= slingshot_settle:
                        in_slingshot = False
                        sling_ref = best_smooth
                prev_loss = loss_val
            self.sleep_wm_slingshot = in_slingshot
            quota_ok = ep >= epochs and (target_loss is None or loss_val <= target_loss)
            if quota_ok:
                if not in_slingshot or extra_used >= extra_cap:
                    self.sleep_wm_slingshot = False
                    break
                extra_used += 1

    @torch.inference_mode()
    def encode_sequence_latents(self, sequences: list, batch_size: int = 128) -> torch.Tensor:
        """Extract normalized full-sequence WM latent embeddings [N, 2*d_model]
        concatenating [mean(contexte), mean(eos..action)] from the mid-layer
        representation (50/50 balance between context and full consequence +
        decision action). Restores ``self.world.training`` afterwards."""
        if not sequences:
            return torch.zeros(0, 2 * self.world.d_model, device=self.device)
        was_training = self.world.training
        self.world.eval()
        chunks = []
        try:
            for i in range(0, len(sequences), batch_size):
                vals = self._batch_tensors(sequences[i:i + batch_size])
                eos_cond = vals[:, self.eos_pos, SIG_OFFSET:SIG_OFFSET + _EOS_NSLOTS]
                self.world(vals, attn_mask=self.mask, is_causal=True,
                           salve_positions=self.salve_positions,
                           eos_cond=eos_cond, eos_pos=self.eos_pos)
                lat = self.world._latent                                # [B, seq_len, d]
                ctx_m = torch.nn.functional.normalize(
                    lat[:, :self.eos_pos, :].mean(dim=1), dim=-1)       # [B, d]
                conseq_m = torch.nn.functional.normalize(
                    lat[:, self.eos_pos:, :].mean(dim=1), dim=-1)       # [B, d]
                emb = torch.nn.functional.normalize(
                    torch.cat([ctx_m, conseq_m], dim=-1), dim=-1)       # [B, 2*d]
                chunks.append(emb)
        finally:
            if was_training:
                self.world.train()
        return torch.cat(chunks, dim=0)

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
                eos_cond = vals[:, self.eos_pos, SIG_OFFSET:SIG_OFFSET + _EOS_NSLOTS]
                pred = self.world(vals, attn_mask=self.mask, is_causal=True,
                                  salve_positions=self.salve_positions,
                                  eos_cond=eos_cond, eos_pos=self.eos_pos)
                pred_signals = pred[:, :-1, :]
                tgt_signals = vals[:, 1:, SIG_OFFSET:]
                w = self.target_weight.unsqueeze(0).expand_as(pred_signals)
                sq = ((pred_signals - tgt_signals) ** 2 * w).sum(dim=(1, 2))
                wsum = w.sum(dim=(1, 2)).clamp(min=1.0)
                out.extend((sq / wsum).tolist())
        return out

    # --- vectorized helpers ------------------------------------------------------
    def _salve_cost_batch(self, gen: torch.Tensor) -> torch.Tensor:
        """Vectorized salve_cost over [B, 13, 29] (or [B, 16, 29])."""
        n = len(_COST_KEYS)
        c = gen[:, _COST_IDX, SIG_OFFSET:SIG_OFFSET + n]
        denorm = self._cost_first + c * self._cost_span
        confort = gen[:, _REWARD_IDX, SIG_OFFSET]
        confort_phys = _CONFORT_FIRST + confort * _CONFORT_SPAN
        return (denorm * self._cost_w).sum(1) + _CONFORT_W * confort_phys

    def _policy_scalars_batch(self, state_tokens: torch.Tensor) -> torch.Tensor:
        """[B, 13, 29] -> [B, 51] policy scalars (tokens 0..11)."""
        B = state_tokens.shape[0]
        sigs = state_tokens[:, :, SIG_OFFSET:SIG_OFFSET + N_SIGNAL]
        flat = sigs.reshape(B, STATE_TOKENS * N_SIGNAL)
        return flat.index_select(1, self._pol_idx)

    def _encode_action_batch(self, action: torch.Tensor) -> torch.Tensor:
        """Action [B, 5] in model space (signed [-1,1], unsigned [0,1]) ->
        action tokens [B, 3, 29] in token signal space [0,1]."""
        if action.dim() == 1:
            action = action.unsqueeze(0)
        B = action.shape[0]
        norm = torch.where(self._act_signed, (action + 1.0) / 2.0, action)
        toks = self._act_tmpl.unsqueeze(0).expand(B, -1, -1).clone()
        flat = toks[:, :, SIG_OFFSET:SIG_OFFSET + N_SIGNAL].reshape(
            B, ACTION_TOKENS * N_SIGNAL)
        flat = flat.scatter(1, (self._act_tok * N_SIGNAL + self._act_slot).unsqueeze(0).expand(B, -1),
                            norm)
        toks[:, :, SIG_OFFSET:SIG_OFFSET + N_SIGNAL] = flat.view(B, ACTION_TOKENS, N_SIGNAL)
        return toks

    def _decode_action_batch(self, act_toks: torch.Tensor) -> torch.Tensor:
        """Action tokens [B, 3, 29] -> action [B, 5] in model space."""
        B = act_toks.shape[0]
        flat = act_toks[:, :, SIG_OFFSET:SIG_OFFSET + N_SIGNAL].reshape(B, ACTION_TOKENS * N_SIGNAL)
        idx = (self._act_tok * N_SIGNAL + self._act_slot)
        vals = flat.index_select(1, idx)                       # [B, 5] in [0,1]
        return torch.where(self._act_signed, 2.0 * vals - 1.0, vals)

    # --- WM inference: dénouement MDN election + generation ----------------------
    @torch.inference_mode()
    def _denouement_dist(self, ctx: torch.Tensor
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MDN distribution of the dénouement from context alone:
        forward on [B, ctx_len, 29], MDN head read at the last row.
        Returns (logits_pi [B, K], mu [B, K, 6], sigma [B, K, 6]).
        Temporarily switches WM to eval() and restores its prior training flag."""
        L = ctx.shape[1]
        was_training = self.world.training
        self.world.eval()
        try:
            _, mdn = self.world.forward_with_eos(
                ctx, attn_mask=build_salve_mask(L, ctx.device), is_causal=True,
                salve_positions=self.salve_positions[:L], eos_row=-1)
        finally:
            if was_training:
                self.world.train()
        return mdn

    @torch.inference_mode()
    def predict_denouement(self, ctx: torch.Tensor) -> torch.Tensor:
        """Most probable dénouement mode predicted from context alone
        (center of the highest-weight MDN component). Returns [B, 6]."""
        logits_pi, mu, _ = self._denouement_dist(ctx)
        b_idx = torch.arange(ctx.shape[0], device=ctx.device)
        best = logits_pi.argmax(dim=-1)                                   # [B]
        return mu[b_idx, best].clamp(0.0, 1.0)

    @torch.inference_mode()
    def sample_denouements(self, ctx: torch.Tensor,
                           temperature: float | None = None,
                           top_n: int | None = None,
                           top_p: float | None = None,
                           min_p: float | None = None,
                           eos_observed: torch.Tensor | None = None,
                           obs_margin: float = 0.10,
                           return_value: bool = False):
        """Exact election over the K continuous MDN modes predicted from context:

        - ``probs = softmax(logits_pi / T)`` gives the exact probability of each
          mode ``k = 0..K-1`` with continuous center ``mu[:, k, :]`` in [0, 1]^6;
        - nucleus (top-p) keeps the smallest prefix of modes sorted by
          descending probability whose cumulative mass >= ``p``
          (``(cum - sorted_p) < p``, no off-by-one); optional ``top_n`` cap;
        - floor ``min_p`` excludes modes with ``probs < min_p`` (the argmax
          mode always stays eligible);
        - among eligible modes, elects the one with highest ``relief_score``;
        - ``p_elu`` is the exact learned probability ``probs[b, elu]``;
        - observed rule: if ``eos_observed`` is given, it replaces the elected
          mode whenever its relief is within ``obs_margin`` of (or exceeds) the
          elected mode's relief (``rel_obs >= rel_elu - obs_margin``, with
          ``p_elu = 1.0``), and sets ``self._last_use_obs`` [B] bool so policy
          training can imitate the recorded action directly instead of a drifted
          WM rollout.

        Returns ``(eos_elu [B, 6], p_elu [B])``, or ``(eos_elu, p_elu, v_ctx [B])``
        when ``return_value=True``, where ``v_ctx = sum_k probs_k * relief(mu_k)``
        is the MDN expected relief of the context (for advantage weighting)."""
        B = ctx.shape[0]
        device = ctx.device
        T = temperature if temperature is not None else self.eos_temperature
        logits_pi, mu, _ = self._denouement_dist(ctx)                     # [B,K], [B,K,6]
        K = logits_pi.shape[1]
        probs = torch.softmax(logits_pi / T, dim=-1)                      # [B, K]

        p = top_p if top_p is not None else self.eos_top_p
        p = min(max(float(p), 0.0), 1.0)
        sorted_p, order = torch.sort(probs, dim=1, descending=True)       # [B, K]
        cum = torch.cumsum(sorted_p, dim=1)                               # [B, K]
        eligible_sorted = (cum - sorted_p) < p                            # [B, K]
        eligible_sorted[:, 0] = True                                      # argmax always kept
        n_abs = top_n if top_n is not None else self.eos_top_n
        if n_abs is not None:
            cap = max(1, min(int(n_abs), K))
            eligible_sorted = eligible_sorted & (
                torch.arange(K, device=device).unsqueeze(0) < cap)
        eligible = torch.zeros(B, K, dtype=torch.bool, device=device).scatter_(
            1, order, eligible_sorted)

        mp = min_p if min_p is not None else self.eos_min_p
        if mp > 0.0:
            eligible = eligible & (probs >= mp)
            best = probs.argmax(dim=1, keepdim=True)
            eligible = eligible.scatter(1, best, torch.ones_like(best, dtype=torch.bool))

        relief = relief_score(mu)                                         # [B, K]
        v_ctx = (probs * relief).sum(dim=1)                               # [B]
        elu = torch.where(
            eligible, relief, torch.full_like(relief, float("-inf"))).argmax(dim=1)
        b_idx = torch.arange(B, device=device)
        eos_elu = mu[b_idx, elu].clamp(0.0, 1.0)                          # [B, 6]
        p_elu = probs[b_idx, elu]                                         # [B]
        if eos_observed is not None:
            rel_obs = relief_score(eos_observed)
            use_obs = rel_obs >= (relief[b_idx, elu] - float(obs_margin))
            eos_elu = torch.where(use_obs.unsqueeze(1), eos_observed, eos_elu)
            p_elu = torch.where(use_obs, torch.ones_like(p_elu), p_elu)
            self._last_use_obs = use_obs
        else:
            self._last_use_obs = torch.zeros(B, dtype=torch.bool, device=device)
        if return_value:
            return eos_elu, p_elu, v_ctx
        return eos_elu, p_elu

    @torch.inference_mode()
    def eos_situation_stats(self, sequences=None, chunk: int = 128) -> dict:
        """Diagnostic du goulot de situation sur des séquences enregistrées
        (coreset par défaut) : code de situation de chaque séquence, nombre de
        codes utilisés, perplexité de leur usage, entropie moyenne des poids π
        du MDN, et fraction de séquences où l'élu == µ (mode le plus probable).
        Returns a dict (``codes`` = liste par séquence, -1 sans VQ)."""
        seqs = list(self.buffer._coreset if sequences is None else sequences)
        out = {"n": len(seqs), "codes": [], "codes_used": 0, "perplexity": 0.0,
               "pi_entropy": 0.0, "elu_eq_mu": 0.0}
        if not seqs:
            return out
        codes, ent, same = [], [], []
        for start in range(0, len(seqs), chunk):
            vals = self._batch_tensors(seqs[start:start + chunk])
            ctx = vals[:, :self.ctx_len]
            logits_pi, mu, _ = self._denouement_dist(ctx)
            vq = self.world.last_vq
            codes.append(vq[1].clone() if vq is not None
                         else torch.full((ctx.shape[0],), -1, device=ctx.device))
            p = torch.softmax(logits_pi, dim=-1)
            ent.append(-(p * torch.log(p.clamp(min=1e-12))).sum(dim=-1))
            elu, _ = self.sample_denouements(ctx)
            mu_best = self.predict_denouement(ctx)
            same.append(((elu - mu_best).abs().max(dim=1).values < 1e-6).float())
        self.world.last_vq = None
        c = torch.cat(codes)
        out["codes"] = c.tolist()
        out["pi_entropy"] = float(torch.cat(ent).mean())
        out["elu_eq_mu"] = float(torch.cat(same).mean())
        if self.world.eos_vq_codes > 0:
            hist = torch.bincount(c, minlength=self.world.eos_vq_codes).float()
            q = hist / hist.sum()
            out["codes_used"] = int((hist > 0).sum())
            out["perplexity"] = float(torch.exp(-(q * torch.log(q.clamp(min=1e-12))).sum()))
        return out

    @torch.inference_mode()
    def generate_from_denouement(self, ctx: torch.Tensor,
                                 eos_vec: torch.Tensor) -> dict:
        """Génération auto-régressive depuis ``[contexte, token de dénouement]`` :
        déroule la conséquence rétro-causale (unités de 16 tokens ``s_k``
        inversée + ``a_{k-1}``, de ``k=seq_steps`` à ``k=n_ctx+1``) jusqu'à
        l'action de décision (3 derniers tokens de la séquence). Masque
        strictement causal (KV-cache), positions explicites.

        Returns dict with ``conseq`` [B, conseq_len, 29], ``action`` [B, 3, 29],
        ``full`` [B, seq_len, 29]."""
        B = ctx.shape[0]
        device = ctx.device
        eos_toks = self._eos_tmpl.unsqueeze(0).expand(B, -1).clone()
        eos_toks[:, SIG_OFFSET:SIG_OFFSET + _EOS_NSLOTS] = eos_vec
        n_gen = self.seq_len - 1 - self.ctx_len                  # tokens après le dénouement forcé
        was_training = self.world.training
        self.world.eval()
        try:
            cache = KVCache(len(self.world.transformer.layers), self.world.d_model, B, device)
            prompt = torch.cat([ctx, eos_toks.unsqueeze(1)], dim=1)  # [B, ctx_len+1, 29]
            out = self.world.cache_forward(prompt, cache,
                                           positions=self.salve_positions[:prompt.shape[1]],
                                           eos_cond=eos_vec, eos_pos=self.eos_pos)
            gen = []
            for t in range(n_gen):
                sig = out[:, -1, :].clamp(0.0, 1.0)                  # prédit le token q
                q = self.ctx_len + 1 + t                             # token en cours de génération
                if q < self.action_pos:
                    off = (q - self.conseq_start) % SALVE_TOKENS
                    if off < STATE_TOKENS:
                        tidx = STATE_TOKENS - 1 - off                # s_k inversée
                        tmpl = self.world.state_template[tidx].to(device)
                        vm = self.world.state_valid_mask[tidx].to(device)
                    else:
                        i = off - STATE_TOKENS                       # a_{k-1}
                        tmpl = self.world.action_template[i].to(device)
                        vm = self.world.action_valid_mask[i].to(device)
                else:
                    i = q - self.action_pos
                    tmpl = self.world.action_template[i].to(device)
                    vm = self.world.action_valid_mask[i].to(device)
                sig = sig * vm
                tok = tmpl.unsqueeze(0).expand(B, -1).clone()
                tok[:, SIG_OFFSET:SIG_OFFSET + N_SIGNAL] = sig
                # cohérence entraînement/génération : le token intero des états
                # générés de la conséquence est masqué (à 0) dans les entrées du WM
                if self.conseq_start < q < self.action_pos \
                        and (q - self.conseq_start) % SALVE_TOKENS == 0:
                    tok[:, SIG_OFFSET:] = 0.0
                gen.append(tok)
                out = self.world.cache_forward(tok.unsqueeze(1), cache,
                                               positions=self.salve_positions[q:q + 1],
                                               eos_cond=eos_vec, eos_pos=self.eos_pos)
        finally:
            if was_training:
                self.world.train()
        gen = torch.stack(gen, dim=1)                            # [B, n_gen, 29], tokens ctx_len+1..seq_len-1
        gen_c = torch.cat([eos_toks.unsqueeze(1), gen], dim=1)   # [B, conseq_len+3, 29], depuis eos_pos
        return {
            "conseq": gen_c[:, :self.conseq_len, :],             # inclut le token de dénouement en position 0
            "action": gen_c[:, self.conseq_len:, :],             # a_{n_ctx}, 3 derniers tokens
            "full": torch.cat([ctx, gen_c], dim=1),              # [B, seq_len, 29]
        }

    @torch.inference_mode()
    def rollout_from_sequence(self, seq: list) -> dict:
        """Diagnostic: depuis le contexte d'une séquence enregistrée, élire un
        dénouement MDN (règle de l'observé appliquée), puis générer la
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
            "relief_observed": float(relief_score(eos_obs)[0]),
            "relief_elu": float(relief_score(eos_elu)[0]),
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
            tags = getattr(self.buffer, "last_sample_tags", [])
            seq_tag = tags[0] if tags else ""
            loss = self._policy_step(seqs, step, steps, seq_tag=seq_tag)
            losses.append(loss)
            yield "pol", loss
            if self._skip_phase:
                self._skip_phase = False
                break
        self.last_pol_loss = sum(losses) / len(losses) if losses else float("nan")

    def _policy_step(self, sequences, step_idx: int, total_steps: int,
                     seq_tag: str = "") -> float:
        B = len(sequences)
        device = self.device
        vals = self._batch_tensors(sequences)
        ctx = vals[:, :self.ctx_len, :].clone()
        eos_obs = vals[:, self.eos_pos, SIG_OFFSET:SIG_OFFSET + _EOS_NSLOTS]
        with torch.inference_mode():
            self.world(ctx, attn_mask=build_salve_mask(self.ctx_len, device),
                       is_causal=True,
                       salve_positions=self.salve_positions[:self.ctx_len])
            latent_inf = self._normalize_latent(self.world.last_latent().detach())
            scalars_inf = self._policy_scalars_batch(ctx[:, -STATE_TOKENS:])
            pol_in_inf = torch.cat([scalars_inf, latent_inf], dim=1)
            pol_action = self.policy(pol_in_inf)                 # [B, 5]
            self._last_use_obs = None
            eos_elu_inf, p_elu_inf, v_ctx_inf = self.sample_denouements(
                ctx, eos_observed=eos_obs, return_value=True)
            use_obs = (self._last_use_obs if self._last_use_obs is not None
                       else torch.zeros(B, dtype=torch.bool, device=device))
            a_real_inf = self._decode_action_batch(
                vals[:, self.action_pos:self.action_pos + ACTION_TOKENS])  # [B, 5]
            # Rollout WM uniquement sur les séquences où un meilleur dénouement
            # a été élu (~use_obs), plus l'index 0 pour la vignette Dream Theater.
            need_gen = ~use_obs
            need_gen_ui = need_gen.clone()
            need_gen_ui[0] = True
            gen_idx = need_gen_ui.nonzero(as_tuple=True)[0]
            gen_sub = self.generate_from_denouement(ctx[gen_idx], eos_elu_inf[gen_idx])
            a_wm_sub = self._decode_action_batch(gen_sub["action"])
            a_star_inf = a_real_inf.clone()
            # Remplacer par l'action imaginée WM uniquement là où ~use_obs
            wm_mask_in_sub = need_gen[gen_idx]
            if wm_mask_in_sub.any():
                a_star_inf[gen_idx[wm_mask_in_sub]] = a_wm_sub[wm_mask_in_sub]
            gen_full_0 = gen_sub["full"][0].tolist()
        # clone hors du bloc inference_mode : tensors normaux, utilisables en autograd
        latent = latent_inf.clone()
        scalars = scalars_inf.clone()
        pol_in = torch.cat([scalars, latent], dim=1)
        eos_elu = eos_elu_inf.clone()
        a_star = a_star_inf.clone()                               # [B, 5]
        pred = self.policy(pol_in)
        # MSE en espace [0, 1] uniforme sur les 5 actionneurs : angles ([-1,1]
        # -> [0,1]) et longueurs ([0,1]) ont exactement la même échelle et la
        # même pente de gradient (évite le facteur 8x en défaveur de d_f/d_b).
        pred_u = torch.where(self._act_signed, 0.5 * (pred + 1.0), pred)
        star_u = torch.where(self._act_signed, 0.5 * (a_star + 1.0), a_star)
        per_seq = ((pred_u - star_u) ** 2).mean(dim=1)           # [B]
        # RL offline par régression pondérée par l'avantage (AWR) :
        #   A = relief(élu) - V_MDN(ctx)
        # Permet d'apprendre aussi le maintien debout et l'évitement de chute
        # (élu = 0.0 vs V = -0.6 -> A = +0.6 > 0). max(A, relief_elu) évite
        # d'annuler un contexte où toutes les issues observées sont déjà de
        # bons relèvements.
        relief_elu = relief_score(eos_elu).detach()              # [B]
        adv = relief_elu - v_ctx_inf.clone().detach()            # [B]
        w = p_elu_inf.clone().detach() * torch.relu(torch.maximum(adv, relief_elu))
        loss = (per_seq * w).sum() / w.sum().clamp(min=1e-8)
        self.opt_pol.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt_pol.step()
        n_obs = int(use_obs.sum().item())
        n_active = int((w > 0).sum().item())
        self.last_pol_stats = {"accepted": n_active, "total": B, "obs": n_obs, "wm": B - n_obs}
        self.last_dream_record = self._build_dream_record(
            sequences[0], ctx[:1], pol_action[:1], a_wm_sub[:1],
            eos_obs[:1], eos_elu[:1], gen_full_0, float(p_elu_inf[0]), loss.item(),
            step_idx, total_steps,
            used_obs=bool(use_obs[0].item()), awr_weight=float(w[0].item()),
            seq_tag=seq_tag)
        return loss.item()

    def _build_dream_record(self, seq, ctx, pol_action, a_wm, eos_obs, eos_elu,
                            gen_full_0, p_elu, loss_val, step_idx, total_steps,
                            used_obs: bool = False, awr_weight: float = 1.0,
                            seq_tag: str = "") -> DreamRecord:
        """Dream Theater record for the refonte policy step: lived sequence
        (rebuilt temporal window), context steps, and the WM-generated
        consequence + decision action under the elected dénouement."""
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
        gen_window = sequence_to_window(gen_full_0, n_ctx=n_ctx, seq_steps=self.seq_steps)
        # Même alignement colonne-à-colonne que lived_steps : s0..s{n_ctx-1},
        # s{n_ctx}+a_wm (à l'emplacement n_ctx), puis s{n_ctx+1}..s{seq_steps} (imaginés).
        predicted_steps = (
            context_steps[:n_ctx]
            + [DreamStep(state_tokens=context_steps[-1].state_tokens,
                         action=a_wm[0].tolist(), label=f"s{n_ctx}+a*")]
            + [DreamStep(state_tokens=[list(t) for t in salve[:STATE_TOKENS]],
                         action=None, step_cost=salve_cost(salve), label=f"s{k} (imagined)")
               for k, salve in enumerate(gen_window[n_ctx + 1:], start=n_ctx + 1)]
        )
        is_active = awr_weight > 0.0
        wm_won = is_active and not used_obs
        return DreamRecord(
            step_idx=step_idx,
            total_steps=total_steps,
            loss=loss_val,
            facing=1,
            context_steps=context_steps,
            candidate_actions=[pol_action[0].tolist(), a_wm[0].tolist()],
            real_action=real_action,
            trajectories=[
                DreamTrajectory(candidate_idx=0, regime_idx=0, regime_name="Policy",
                                steps=[], is_best_future=False, is_winner=False,
                                eos_pred=None),
                DreamTrajectory(candidate_idx=1, regime_idx=0,
                                regime_name="Elected outcome",
                                steps=predicted_steps[n_ctx + 1:], is_best_future=wm_won,
                                is_winner=wm_won,
                                eos_pred=eos_elu[0].tolist()),
            ],
            best_candidate_idx=1 if wm_won else -1,
            best_future_indices=[0, 0],
            candidate_names=["0: POLICY", "1: WM HINDSIGHT"],
            eos_forced=eos_elu[0].tolist(),
            lived_steps=lived_steps,
            predicted_steps=predicted_steps,
            eos_realized=eos_obs[0].tolist(),
            used_obs=used_obs,
            awr_weight=awr_weight,
            seq_tag=seq_tag,
        )

    # --- sleep cycle -------------------------------------------------------------
    def sleep_cycle(self, wm_coreset_epoch: int = 5, wm_addendum_epoch: int = 5,
                    wm_batch: int = 64, pol_steps: int = 32, pol_batch: int = 32,
                    wm_target_loss: float | None = 0.004,
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
        wm_converged = (not math.isnan(self.last_wm_loss)) and (
            wm_target_loss is None or self.last_wm_loss <= wm_target_loss)
        dedup_embed = self.encode_sequence_latents if wm_converged else None
        dedup_eps = self.dedup_latent_eps if wm_converged else self.buffer.dedup_eps
        d_intra = self.buffer.deduplicate_addendum(
            eps=dedup_eps, embed_fn=dedup_embed, device=self.device)
        d_core = self.buffer.deduplicate_against_coreset(
            eps=dedup_eps, embed_fn=dedup_embed, device=self.device)
        add_kept = len(self.buffer._addendum)
        total_after = coreset_before + add_kept
        evicted_est = max(0, total_after - self.buffer.pool_capacity)
        coreset_after_est = total_after - evicted_est
        self.sleep_cycle_stats = {
            "coreset_before": coreset_before,
            "wake": n_wake,
            "dedup_mode": "latent" if wm_converged else "geom",
            "dedup_intra": d_intra,
            "dedup_coreset": d_core,
            "addendum_kept": add_kept,
            "evicted": evicted_est,
            "coreset_after": coreset_after_est,
        }
        yield "prune", {
            "pruned": 0,
            "coreset": self.buffer.coreset_size,
            "dedup_mode": self.sleep_cycle_stats["dedup_mode"],
            "dedup_intra": d_intra,
            "dedup_coreset": d_core,
            "addendum_kept": len(self.buffer._addendum),
        }

        has_coreset = self.buffer.coreset_size > 0
        has_addendum = len(self.buffer._addendum) > 0
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

        commit_converged = (not math.isnan(self.last_wm_loss)) and (
            wm_target_loss is None or self.last_wm_loss <= wm_target_loss)
        commit_embed = self.encode_sequence_latents if commit_converged else None
        n_committed = self.buffer.commit_addendum(embed_fn=commit_embed, device=self.device)
        self.sleep_cycle_stats["evicted"] = self.buffer.last_evicted
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
        }

    def load_state_dict(self, sd):
        self.load_world_state(sd["world"])
        self.policy.load_state_dict(sd["policy"])

    def load_world_state(self, sd: dict) -> bool:
        """Charge les poids du WM en tolérant un changement de la tête EOS
        (goulot de situation VQ ajouté/retiré, dimensions MDN) : les clés
        absentes ou de forme différente gardent leur initialisation, le reste
        (tronc transformer, head des tokens) est restauré. Returns True si le
        checkpoint correspond exactement (l'état Adam est alors réutilisable)."""
        own = self.world.state_dict()
        keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        skipped = sorted(set(sd) - set(keep))
        missing = sorted(set(own) - set(keep))
        self.world.load_state_dict(keep, strict=False)
        if skipped or missing:
            print(f"[warning] WM checkpoint: EOS head re-initialized "
                  f"({len(missing)} params missing/reshaped) — retrain the WM")
            return False
        return True

    def save_checkpoint(self, wm_path: str, pol_path: str, buf_path: str | None = None) -> None:
        torch.save({"world": self.world.state_dict(),
                    "opt_wm": self.opt_wm.state_dict(),
                    "last_wm_loss": self.last_wm_loss}, wm_path)
        torch.save({"policy": self.policy.state_dict(),
                    "opt_pol": self.opt_pol.state_dict()}, pol_path)
        if buf_path is not None:
            self.buffer.save(buf_path)

    def load_checkpoint(self, wm_path: str, pol_path: str, buf_path: str | None = None) -> None:
        if not os.path.exists(wm_path) or not os.path.exists(pol_path):
            raise FileNotFoundError("missing WM or policy checkpoint — run with --reset")
        wm = torch.load(wm_path, weights_only=False)
        exact = self.load_world_state(wm["world"])
        if exact and "opt_wm" in wm:
            self.opt_wm.load_state_dict(wm["opt_wm"])
        if exact and "last_wm_loss" in wm:
            self.last_wm_loss = float(wm["last_wm_loss"])
        else:
            self.last_wm_loss = float("nan")
        pol = torch.load(pol_path, weights_only=False)
        self.policy.load_state_dict(pol["policy"])
        if "opt_pol" in pol:
            self.opt_pol.load_state_dict(pol["opt_pol"])
        if buf_path is not None and os.path.exists(buf_path):
            self.buffer.load(buf_path)
