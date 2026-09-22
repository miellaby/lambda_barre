"""Dense token encoder for λ̄'s world model.

Each salve is a list of 16 dense tokens (13 sensory + 3 motor). A token is a
vector of 25 floats concatenating:

    type      (1)   0.0 = sensory, 1.0 = motor
    modality  (4)   hand-picked orthogonal code (intero, somato, visual, motiv)
    canal     (4)   hand-picked code within the modality
    signals   (16)  one normalized [0,1] scalar per slot, zero-padded if < 16

There are no learned embeddings and no separator tokens — the structural prefix
(type + modality + canal) is a fixed hand-encoded label that identifies the
channel, and the 16 signal slots carry the normalized values directly. A salve
is thus ``list[list[float]]`` of length 16, each inner list length 25.

Usage::

    enc = DenseEncoder()
    salve = enc.encode(proprio, touch, cursor, vision, intero, reward, skel)
    # salve: 16 tokens x 25 floats
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from lambda_barre.body import LIMB_MAX, LIMB_MIN


# --- dense token geometry -----------------------------------------------------
TYPE_DIM = 1
MOD_DIM = 4
CANAL_DIM = 4
N_SIGNAL = 16
DENSE_DIM = TYPE_DIM + MOD_DIM + CANAL_DIM + N_SIGNAL  # 25

STATE_TOKENS = 13     # sensory tokens (indices 0..12)
ACTION_TOKENS = 3     # motor tokens (indices 13..15)
SALVE_TOKENS = STATE_TOKENS + ACTION_TOKENS  # 16

# --- quantization (kept for reference / external use) -------------------------
VAL_LEVELS = 9
VAL_MAX = 8


@dataclass
class TokenSpec:
    key: str
    code: str          # 2-3 letter code for display
    max_val: float       # maximum value
    signed: bool       # whether the value can be negative
    min_val: float = 0.0  # optional minimum value


def _build_specs() -> dict[str, TokenSpec]:
    specs: dict[str, TokenSpec] = {}

    def add(key, code, max_val, signed, min_val=None):
        if min_val is None:
            min_val = -max_val if signed else 0.0
        specs[key] = TokenSpec(key, code, max_val, signed, min_val)

    # Proprioception (11)
    add("tronc_angle",              "TA", math.radians(345.0), False, -math.radians(15.0))
    add("membre_angle_avant",       "AA", math.pi,          True)
    add("membre_distance_avant",    "DA", LIMB_MAX,         False)
    add("force_actuateur_avant",    "FA", 2000.0,           True)
    add("membre_angle_arriere",     "AR", math.pi,          True)
    add("membre_distance_arriere",  "DR", LIMB_MAX,         False)
    add("force_actuateur_arriere",  "FR", 2000.0,           True)
    add("queue_angle",             "QA", math.pi,          True)
    add("couple_queue",            "CQ", 200000.0,         True)
    add("accel_tete_avant",        "XA", 2000.0,           True)
    add("accel_tete_haut",         "YA", 2000.0,           True)

    # Vision (16: 16 cells × 1 brightness)
    for i in range(1, 17):
        add(f"vis_c{i}", f"V{i}", 1.0, False)
    # Optical flow (3)
    add("flux_surface", "FS", 5000.0, False)
    add("flux_x",       "FX", 150.0,  True)
    add("flux_y",       "FY", 150.0,  True)

    # Env / cursor (9)
    add("curseur_dir",  "CD", math.pi, True)
    add("curseur_prox", "CP", 1.0,     False)
    add("curseur_vx",   "CV", 1500.0,  True)
    add("curseur_vy",   "CW", 1500.0,  True)
    add("son_0",        "S0", 1.0,     False)
    add("son_1",        "S1", 1.0,     False)
    add("son_2",        "S2", 1.0,     False)
    add("son_3",        "S3", 1.0,     False)
    add("son_4",        "S4", 1.0,     False)

    # Touch (6)
    add("contact_sol_avant",    "SA", 1500.0, False)
    add("contact_sol_arriere",  "SR", 1500.0, False)
    add("collision_tronc_x",    "TX", 1500.0, True)
    add("collision_tronc_y",    "TY", 1500.0, True)
    add("collision_tronc_cx",   "TC", 200.0,  True)
    add("collision_tronc_cy",   "TD", 200.0,  True)

    # Interoception (2) - signed delta in [-1.0, +1.0]
    add("fatigue",     "FT", 1.0, True)
    add("souffrance",  "SF", 1.0, True)

    # Reward (8)
    add("effort",      "EF", 1.0, False)
    add("douleur",     "DO", 1.0, False)
    add("courbature",  "CO", 1.0, False)
    add("instabilite", "IN", 1.0, False)
    add("vertige",     "VE", 1.0, False)
    add("confort",     "CF", 1.0, True)   # signed: reward (negative)
    add("reward_pos",  "RP", 1.0, True)
    add("reward_neg",  "RN", 1.0, False)

    # Actions (5)
    add("membre_avant_theta", "FT", math.pi, True)
    add("membre_avant_d",     "FD", LIMB_MAX,    False, LIMB_MIN)
    add("membre_arriere_theta", "RT", math.pi, True)
    add("membre_arriere_d",     "RD"    , LIMB_MAX,    False, LIMB_MIN)
    add("queue_theta",   "QT", math.pi, True)

    return specs


_SPECS = _build_specs()
_BY_KEY: dict[str, TokenSpec] = _SPECS


def _quantize(raw: float, spec: TokenSpec) -> int:
    if spec.signed:
        t = raw / spec.max_val
        t = max(-1.0, min(1.0, t))
        return int(round((t + 1.0) / 2.0 * VAL_MAX))
    else:
        t = (raw - spec.min_val) / (spec.max_val - spec.min_val)
        t = max(0.0, min(1.0, t))
        return int(round(t * VAL_MAX))


def _dequantize(val: int, spec: TokenSpec) -> float:
    val = max(0, min(VAL_MAX, val))

    if spec.signed:
        t = (val / VAL_MAX) * 2.0 - 1.0
        return t * spec.max_val
    else:
        t = val / VAL_MAX
        return spec.min_val + t * (spec.max_val - spec.min_val)


def dequantize(val: int, key: str) -> float:
    """Dequantize a 0-8 token value back to its physical value."""
    return _dequantize(val, _BY_KEY[key])


def quantize_value(raw: float, key: str) -> int:
    """Quantize a raw physical value to 0-8 for a signal key."""
    return _quantize(raw, _BY_KEY[key])


# --- normalization for the dense (float) format --------------------------------
def _normalize(raw: float, spec: TokenSpec) -> float:
    """Map a raw physical value to a [0,1] float slot value."""
    if spec.signed:
        t = raw / spec.max_val
        t = max(-1.0, min(1.0, t))
        return (t + 1.0) / 2.0
    else:
        t = (raw - spec.min_val) / (spec.max_val - spec.min_val)
        return max(0.0, min(1.0, t))

def _denormalize(t: float, spec: TokenSpec) -> float:
    """Inverse of ``_normalize`` — recover the physical value from [0,1]."""
    t = max(0.0, min(1.0, t))

    if spec.signed:
        return (t * 2.0 - 1.0) * spec.max_val
    else:
        return spec.min_val + t * (spec.max_val - spec.min_val)


# --- dense channel layout -----------------------------------------------------
@dataclass
class ChannelSpec:
    idx: int                       # 0..15
    is_motor: bool                 # type bit
    modality: tuple                 # 4 floats
    canal: tuple                   # 4 floats
    signals: list[str] = field(default_factory=list)  # signal keys, in order
    source: str = ""                # sensor dict: proprio/touch/cursor/vision/
                                    # intero/reward/action


# Modality codes (4d, hand-picked orthogonal-ish)
_M_INTERO = (1.0, 0.0, 0.0, 0.0)
_M_SOMATO = (0.0, 1.0, 0.0, 0.0)
_M_VISUAL = (0.0, 0.0, 1.0, 0.0)
_M_MOTIV  = (1.0, 0.0, 0.0, 1.0)
_M_ACTION = (0.0, 0.0, 0.0, 0.0)


def _build_layout() -> list[ChannelSpec]:
    return [
        # --- sensory (0..12) ---
        ChannelSpec(0, False, _M_SOMATO, (0.0, 0.0, 0.0, 1.0),
                    ["force_actuateur_avant", "force_actuateur_arriere",
                     "couple_queue"], "proprio"),
        ChannelSpec(1, False, _M_SOMATO, (0.0, 0.0, 1.0, 0.0),
                    ["tronc_angle", "queue_angle", "accel_tete_avant",
                     "accel_tete_haut"], "proprio"),
        ChannelSpec(2, False, _M_SOMATO, (0.0, 1.0, 0.0, 0.0),
                    ["membre_angle_avant", "membre_distance_avant"], "proprio"),
        ChannelSpec(3, False, _M_SOMATO, (1.0, 0.0, 0.0, 0.0),
                    ["membre_angle_arriere", "membre_distance_arriere"],
                    "proprio"),
        ChannelSpec(4, False, _M_VISUAL, (1.0, 1.0, 1.0, 1.0),
                    [f"vis_c{i}" for i in range(1, 17)], "vision"),
        ChannelSpec(5, False, _M_VISUAL, (0.0, 0.0, 0.0, 1.0),
                    ["curseur_dir", "curseur_prox", "curseur_vx",
                     "curseur_vy"], "cursor"),
        ChannelSpec(6, False, _M_VISUAL, (0.0, 0.0, 1.0, 0.0),
                    ["flux_surface", "flux_x", "flux_y"], "vision"),
        ChannelSpec(7, False, _M_VISUAL, (0.0, 1.0, 0.0, 0.0),
                    ["son_0", "son_1", "son_2", "son_3", "son_4"], "cursor"),
        ChannelSpec(8, False, _M_SOMATO, (0.0, 0.0, 0.0, 1.0),
                    ["contact_sol_avant", "contact_sol_arriere"], "touch"),
        ChannelSpec(9, False, _M_SOMATO, (0.0, 0.0, 1.0, 0.0),
                    ["collision_tronc_x", "collision_tronc_y",
                     "collision_tronc_cx", "collision_tronc_cy"], "touch"),
        ChannelSpec(10, False, _M_MOTIV, (0.0, 0.0, 0.0, 0.0),
                    ["effort", "douleur", "courbature", "instabilite",
                     "vertige"], "reward"),
        ChannelSpec(11, False, _M_MOTIV, (1.0, 0.0, 0.0, 0.0),
                    ["confort"], "reward"),
        ChannelSpec(12, False, _M_INTERO, (1.0, 1.0, 1.0, 1.0),
                    ["fatigue", "souffrance"], "intero"),
        # --- motor (13..15) ---
        ChannelSpec(13, True, _M_ACTION, (0.0, 0.0, 0.0, 1.0),
                    ["membre_avant_theta", "membre_avant_d"], "action"),
        ChannelSpec(14, True, _M_ACTION, (0.0, 0.0, 1.0, 0.0),
                    ["membre_arriere_theta", "membre_arriere_d"], "action"),
        ChannelSpec(15, True, _M_ACTION, (0.0, 1.0, 0.0, 0.0),
                    ["queue_theta"], "action"),
    ]


_LAYOUT: list[ChannelSpec] = _build_layout()

# Number of valid (non-padded) signal slots per token index.
_N_SIGNALS: list[int] = [len(ch.signals) for ch in _LAYOUT]
print(f"signal slots per token: {_N_SIGNALS}")

# Index of the reward tokens and interoception within the state.
_COST_IDX = 10        # "Coûts"  (effort, douleur, courbature, instabilite, vertige)
_REWARD_IDX = 11      # "Récompense" (confort)
_INTERO_IDX = 12      # "Intéroception" (delta fatigue, delta souffrance)

# Signal-slot offset within a 25-dim token (type + modality + canal = 9).
SIG_OFFSET = TYPE_DIM + MOD_DIM + CANAL_DIM   # 9

# Keys of the 5 innate cost signals (token 10), in slot order.
_COST_KEYS = ("effort", "douleur", "courbature", "instabilite", "vertige")

# Flat signal counts within a salve (for make_salve / construction).
_STATE_FLAT_LEN = sum(_N_SIGNALS[:STATE_TOKENS])    # 53
_ACTION_FLAT_LEN = sum(_N_SIGNALS[STATE_TOKENS:])   # 5

# Number of non-reward sensory scalar slots fed to the policy (tokens 0..9).
N_POLICY_STATE = sum(_N_SIGNALS[:_COST_IDX])         # 45

# Per-token action-key → (skeleton attribute, field) lookup.
_ACTION_LOOKUP = {
    "membre_avant_theta":   ("limb_front", "theta_star"),
    "membre_avant_d":       ("limb_front", "d_star"),
    "membre_arriere_theta": ("limb_back", "theta_star"),
    "membre_arriere_d":     ("limb_back", "d_star"),
    "queue_theta":   ("tail_act", "theta_star"),
}

# Weights for each innate cost signal in salve_cost.
_COST_WEIGHTS = {
    "effort": 1.0,
    "douleur": 4.0,
    "courbature": 2.0,
    "instabilite": 2.0,
    "vertige": 3.0,
    "confort": 1.0,
}


def _prefix(ch: ChannelSpec) -> list[float]:
    """The 9-float structural prefix (type + modality + canal)."""
    return [1.0 if ch.is_motor else 0.0] + list(ch.modality) + list(ch.canal)


def _signal_slot(token: list[float], k: int) -> float:
    """Read the k-th signal slot (0..15) of a 25-dim token."""
    return token[TYPE_DIM + MOD_DIM + CANAL_DIM + k]


def state_tokens(salve: list[list[float]]) -> list[list[float]]:
    """The 13 sensory tokens of a salve."""
    return salve[:STATE_TOKENS]


def action_tokens(salve: list[list[float]]) -> list[list[float]]:
    """The 3 motor tokens of a salve."""
    return salve[STATE_TOKENS:SALVE_TOKENS]


def policy_scalars(salve: list[list[float]]) -> list[float]:
    """The 45 non-reward sensory signal values (tokens 0..9), in order.

    These are the normalized [0,1] floats the policy consumes alongside the
    world-model latent."""
    out: list[float] = []
    for i in range(_COST_IDX):           # tokens 0..9 (exclude reward 10,11 and intero 12)
        ch = _LAYOUT[i]
        token = salve[i]
        for k in range(len(ch.signals)):
            out.append(_signal_slot(token, k))
    return out


def salve_cost(salve: list[list[float]]) -> float:
    """Weighted innate cost from the reward channels (tokens 10 and 11).

    cost = 1·effort + 4·douleur + 2·courbature + 2·instabilite
           + 3·vertige + 1·confort
    (confort is signed: negative = reward; the 5 others are unsigned costs).
    Lower is better. Works on a 16-token salve or a 13-token state."""
    total = 0.0
    cost_ch = _LAYOUT[_COST_IDX]
    for k, key in enumerate(cost_ch.signals):
        t = _signal_slot(salve[_COST_IDX], k)
        total += _COST_WEIGHTS[key] * _denormalize(t, _BY_KEY[key])
    t_confort = _signal_slot(salve[_REWARD_IDX], 0)
    total += _COST_WEIGHTS["confort"] * _denormalize(t_confort,
                                                     _BY_KEY["confort"])
    return total


def encode_action(theta_front: float, d_front: float, theta_back: float, d_back: float,
                  tail_t: float) -> list[list[float]]:
    """Encode the 5 actuator consignes into the 3 motor dense tokens (13..15)."""
    raw = {
        "membre_avant_theta": theta_front, "membre_avant_d": d_front,
        "membre_arriere_theta": theta_back, "membre_arriere_d": d_back,
        "queue_theta": tail_t,
    }
    out: list[list[float]] = []
    for i in range(STATE_TOKENS, SALVE_TOKENS):
        ch = _LAYOUT[i]
        sigs = [0.0] * N_SIGNAL
        for k, key in enumerate(ch.signals):
            sigs[k] = _normalize(raw[key], _BY_KEY[key])
        out.append(_prefix(ch) + sigs)
    return out


def make_salve(state_signals: list[float],
               action_signals: list[float]) -> list[list[float]]:
    """Build a 16-token salve from flat signal lists.

    ``state_signals`` has 53 floats (all sensory slots in token order, tokens
    0..12), ``action_signals`` has 5 floats (motor slots, tokens 13..15).
    Values are placed in valid slots; padded slots stay 0. Structural prefixes
    are filled from the fixed layout. Used by tests."""
    salve: list[list[float]] = []
    si = 0
    for i in range(STATE_TOKENS):
        ch = _LAYOUT[i]
        sigs = [0.0] * N_SIGNAL
        for k in range(len(ch.signals)):
            sigs[k] = state_signals[si]
            si += 1
        salve.append(_prefix(ch) + sigs)
    ai = 0
    for i in range(STATE_TOKENS, SALVE_TOKENS):
        ch = _LAYOUT[i]
        sigs = [0.0] * N_SIGNAL
        for k in range(len(ch.signals)):
            sigs[k] = action_signals[ai]
            ai += 1
        salve.append(_prefix(ch) + sigs)
    return salve


class DenseEncoder:
    """Encodes a full sensor snapshot into a salve of 16 dense tokens.

    A salve is a list of 16 tokens, each a 25-float vector.
    """

    def encode(self, proprio: dict, touch: dict, cursor: dict,
              vision: dict, intero: dict, reward: dict,
              skel) -> list[list[float]]:
        dicts = {"proprio": proprio, "touch": touch, "cursor": cursor,
                 "vision": vision, "intero": intero, "reward": reward}
        salve: list[list[float]] = []
        for ch in _LAYOUT:
            sigs = [0.0] * N_SIGNAL
            if ch.source == "action":
                for k, key in enumerate(ch.signals):
                    attr, fld = _ACTION_LOOKUP[key]
                    raw = getattr(getattr(skel, attr), fld)
                    # Convert anatomical limb coordinates to the
                    # egocentric front/back coordinate frame (angle > 0 = outward rotation).
                    if skel.facing == 1 and key == "membre_avant_theta":
                        raw = -raw
                    elif skel.facing == -1 and key == "membre_arriere_theta":
                        raw = -raw
                    sigs[k] = _normalize(raw, _BY_KEY[key])
            else:
                src = dicts[ch.source]
                for k, key in enumerate(ch.signals):
                    raw = src.get(key, 0.0) if src is not None else 0.0
                    sigs[k] = _normalize(raw, _BY_KEY[key])
            salve.append(_prefix(ch) + sigs)
        return salve

    def decode(self, salve: list[list[float]]) -> list[str]:
        """Decode a salve into a list of short display strings (one per token)."""
        lines: list[str] = []
        for i, token in enumerate(salve):
            ch = _LAYOUT[i] if i < len(_LAYOUT) else None
            kind = "A" if (ch and ch.is_motor) else "S"
            codes = []
            if ch:
                for k, key in enumerate(ch.signals):
                    t = _signal_slot(token, k)
                    codes.append(f"{_BY_KEY[key].code}={t:.2f}")
            lines.append(f"[{kind}{i}] " + " ".join(codes))
        return lines
