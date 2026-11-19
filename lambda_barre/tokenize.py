"""Dense token encoder for λ̄'s world model.

Each salve is a list of 16 dense tokens (13 sensory + 3 motor). A token is a
vector of 29 floats concatenating:

    type      (1)   0.0 = sensory, 1.0 = motor
    modality  (6)   one-hot: one bit per class (intero, proprio, extero,
                    renforcement, action, dénouement/EOS) — exactly one lit
    canal     (6)   one-hot within the modality (ids reused across modalities)
    signals   (16)  one normalized [0,1] scalar per slot, zero-padded if < 16

There are no learned embeddings and no separator tokens — the structural prefix
(type + modality + canal) is a fixed hand-encoded label that identifies the
channel, and the 16 signal slots carry the normalized values directly. A salve
is thus ``list[list[float]]`` of length 16, each inner list length 29.

Usage::

    enc = DenseEncoder()
    salve = enc.encode(proprio, touch, ball, vision, intero, reward, skel)
    # salve: 16 tokens x 29 floats
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from lambda_barre.body import LIMB_MAX, LIMB_MIN


# --- dense token geometry -----------------------------------------------------
TYPE_DIM = 1
MOD_DIM = 6
CANAL_DIM = 6
N_SIGNAL = 16
DENSE_DIM = TYPE_DIM + MOD_DIM + CANAL_DIM + N_SIGNAL  # 29

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
    first_val: float   # physical value mapped to 0.0
    last_val: float    # physical value mapped to 1.0


def _build_specs() -> dict[str, TokenSpec]:
    specs: dict[str, TokenSpec] = {}

    def add(key: str, code: str, first_val: float, last_val: float) -> None:
        specs[key] = TokenSpec(key, code, first_val, last_val)

    # Proprioception (11)
    add("tronc_angle",              "TA", -math.radians(15.0), math.radians(345.0))
    add("membre_angle_avant",       "AA", -math.pi,            math.pi)
    add("membre_distance_avant",    "DA", 0.0,                 LIMB_MAX)
    add("force_actuateur_avant",    "FA", -2000.0,             2000.0)
    add("membre_angle_arriere",     "AR", -math.pi,            math.pi)
    add("membre_distance_arriere",  "DR", 0.0,                 LIMB_MAX)
    add("force_actuateur_arriere",  "FR", -2000.0,             2000.0)
    add("queue_angle",              "QA", -math.pi,            math.pi)
    add("couple_queue",             "CQ", -200000.0,           200000.0)
    add("accel_tete_avant",         "XA", -2000.0,             2000.0)
    add("accel_tete_haut",          "YA", -2000.0,             2000.0)

    # Vision (16: 16 cells × 1 brightness)
    for i in range(1, 17):
        add(f"vis_c{i}", f"V{i}", 0.0, 1.0)
    # Optical flow (3)
    add("flux_surface", "FS", 0.0,    5000.0)
    add("flux_x",       "FX", -150.0, 150.0)
    add("flux_y",       "FY", -150.0, 150.0)

    # Env / ball (9)
    add("ball_dir",  "BD", -math.pi, math.pi)
    add("ball_prox", "BP", 0.0,      1.0)
    add("ball_vr",   "BR", -1500.0,  1500.0)
    add("ball_va",   "BW", -10.0,    10.0)
    add("son_0",     "S0", 0.0,      1.0)
    add("son_1",     "S1", 0.0,      1.0)
    add("son_2",     "S2", 0.0,      1.0)
    add("son_3",     "S3", 0.0,      1.0)
    add("son_4",     "S4", 0.0,      1.0)

    # Touch (6)
    add("contact_sol_avant",   "SA", 0.0,     1500.0)
    add("contact_sol_arriere", "SR", 0.0,     1500.0)
    add("collision_tronc_x",   "TX", -1500.0, 1500.0)
    add("collision_tronc_y",   "TY", -1500.0, 1500.0)
    add("collision_tronc_cx",  "TC", -200.0,  200.0)
    add("collision_tronc_cy",  "TD", -200.0,  200.0)

    # Interoception (2) — absolute levels in [0, 1], masked from WM inputs.
    add("fatigue",    "FT", 0.0, 1.0)
    add("souffrance", "SF", 0.0, 1.0)

    # Reward (8) — costs in [0, 1]; confort & reward_pos are negative costs
    # (0.0 = absent -> 0.0, -1.0 = max reward -> 1.0).
    add("effort",      "EF", 0.0, 1.0)
    add("douleur",     "DO", 0.0, 1.0)
    add("courbature",  "CO", 0.0, 1.0)
    add("instabilite", "IN", 0.0, 1.0)
    add("vertige",     "VE", 0.0, 1.0)
    add("confort",     "CF", 0.0, -1.0)
    add("reward_pos",  "RP", 0.0, -1.0)
    add("reward_neg",  "RN", 0.0, 1.0)

    # Actions (5)
    add("membre_avant_theta",   "FT", -math.pi, math.pi)
    add("membre_avant_d",       "FD", LIMB_MIN, LIMB_MAX)
    add("membre_arriere_theta", "RT", -math.pi, math.pi)
    add("membre_arriere_d",     "RD", LIMB_MIN, LIMB_MAX)
    add("queue_theta",          "QT", -math.pi, math.pi)

    return specs


_SPECS = _build_specs()
_BY_KEY: dict[str, TokenSpec] = _SPECS


def _quantize(raw: float, spec: TokenSpec) -> int:
    t = (raw - spec.first_val) / (spec.last_val - spec.first_val)
    t = max(0.0, min(1.0, t))
    return int(round(t * VAL_MAX))


def _dequantize(val: int, spec: TokenSpec) -> float:
    val = max(0, min(VAL_MAX, val))
    t = val / VAL_MAX
    return spec.first_val + t * (spec.last_val - spec.first_val)


def dequantize(val: int, key: str) -> float:
    """Dequantize a 0-8 token value back to its physical value."""
    return _dequantize(val, _BY_KEY[key])


def quantize_value(raw: float, key: str) -> int:
    """Quantize a raw physical value to 0-8 for a signal key."""
    return _quantize(raw, _BY_KEY[key])


# --- normalization for the dense (float) format --------------------------------
def _normalize(raw: float, spec: TokenSpec) -> float:
    """Map a raw physical value to a [0,1] float slot value."""
    t = (raw - spec.first_val) / (spec.last_val - spec.first_val)
    return max(0.0, min(1.0, t))


def _denormalize(t: float, spec: TokenSpec) -> float:
    """Inverse of ``_normalize`` — recover the physical value from [0,1]."""
    t = max(0.0, min(1.0, t))
    return spec.first_val + t * (spec.last_val - spec.first_val)


# --- dense channel layout -----------------------------------------------------
@dataclass
class ChannelSpec:
    idx: int                       # 0..15
    is_motor: bool                 # type bit
    modality: tuple                 # 4 floats
    canal: tuple                   # 4 floats
    signals: list[str] = field(default_factory=list)  # signal keys, in order
    source: str = ""                # sensor dict: proprio/touch/ball/vision/
                                    # intero/reward/action
    label: str = ""                # short id of the channel (FORCE, VISION, ...)


# Modality codes: one-hot on 6 bits, exactly one lit per token
# (intero, proprio, extero, renforcement, action, dénouement)
_M_INTERO = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_M_PROPRIO = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
_M_EXTERO  = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
_M_RENF    = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
_M_ACTION  = (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
_M_EOS     = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def _canal(i: int) -> tuple:
    """One-hot canal code on CANAL_DIM bits. Ids are per-modality and reused
    across modalities (the one-hot modality already separates them)."""
    return tuple(1.0 if j == i else 0.0 for j in range(CANAL_DIM))


def _build_layout() -> list[ChannelSpec]:
    return [
        # --- sensory (0..12) ---
        ChannelSpec(0, False, _M_PROPRIO, _canal(0),
                    ["force_actuateur_avant", "force_actuateur_arriere",
                     "couple_queue"], "proprio", label="FORCE"),
        ChannelSpec(1, False, _M_PROPRIO, _canal(1),
                    ["tronc_angle", "queue_angle", "accel_tete_avant",
                     "accel_tete_haut"], "proprio", label="POSTURE"),
        ChannelSpec(2, False, _M_PROPRIO, _canal(2),
                    ["membre_angle_avant", "membre_distance_avant"], "proprio",
                    label="LIMB_F"),
        ChannelSpec(3, False, _M_PROPRIO, _canal(3),
                    ["membre_angle_arriere", "membre_distance_arriere"],
                    "proprio", label="LIMB_B"),
        ChannelSpec(4, False, _M_EXTERO, _canal(0),
                    [f"vis_c{i}" for i in range(1, 17)], "vision", label="VISION"),
        ChannelSpec(5, False, _M_EXTERO, _canal(1),
                    ["ball_dir", "ball_prox", "ball_vr",
                     "ball_va"], "ball", label="BALL"),
        ChannelSpec(6, False, _M_EXTERO, _canal(2),
                    ["flux_surface", "flux_x", "flux_y"], "vision", label="FLOW"),
        ChannelSpec(7, False, _M_EXTERO, _canal(3),
                    ["son_0", "son_1", "son_2", "son_3", "son_4"], "ball",
                    label="SOUND"),
        ChannelSpec(8, False, _M_EXTERO, _canal(4),
                    ["contact_sol_avant", "contact_sol_arriere"], "touch",
                    label="TOUCH_FOOT"),
        ChannelSpec(9, False, _M_EXTERO, _canal(5),
                    ["collision_tronc_x", "collision_tronc_y",
                     "collision_tronc_cx", "collision_tronc_cy"], "touch",
                    label="TOUCH_TRUNK"),
        ChannelSpec(10, False, _M_RENF, _canal(0),
                    ["effort", "douleur", "courbature", "instabilite",
                     "vertige"], "reward", label="COSTS"),
        ChannelSpec(11, False, _M_RENF, _canal(1),
                    ["confort"], "reward", label="REWARD"),
        ChannelSpec(12, False, _M_INTERO, _canal(0),
                    ["fatigue", "souffrance"], "intero", label="METAB"),
        # --- motor (13..15) ---
        ChannelSpec(13, True, _M_ACTION, _canal(0),
                    ["membre_avant_theta", "membre_avant_d"], "action",
                    label="ACT_F"),
        ChannelSpec(14, True, _M_ACTION, _canal(1),
                    ["membre_arriere_theta", "membre_arriere_d"], "action",
                    label="ACT_B"),
        ChannelSpec(15, True, _M_ACTION, _canal(2),
                    ["queue_theta"], "action", label="ACT_TAIL"),
    ]


_LAYOUT: list[ChannelSpec] = _build_layout()

# Number of valid (non-padded) signal slots per token index.
_N_SIGNALS: list[int] = [len(ch.signals) for ch in _LAYOUT]
# print(f"signal slots per token: {_N_SIGNALS}")

# Index of the reward tokens and interoception within the state.
_COST_IDX = 10        # "Coûts"  (effort, douleur, courbature, instabilite, vertige)
_REWARD_IDX = 11      # "Récompense" (confort)
_INTERO_IDX = 12      # "Intéroception" (fatigue, souffrance — niveaux absolus)

# Signal-slot offset within a 29-dim token (type + modality + canal = 13).
SIG_OFFSET = TYPE_DIM + MOD_DIM + CANAL_DIM   # 13

# Keys of the 5 innate cost signals (token 10), in slot order.
_COST_KEYS = ("effort", "douleur", "courbature", "instabilite", "vertige")

# Flat signal counts within a salve (for make_salve / construction).
_STATE_FLAT_LEN = sum(_N_SIGNALS[:STATE_TOKENS])    # 53
_ACTION_FLAT_LEN = sum(_N_SIGNALS[STATE_TOKENS:])   # 5

# Number of scalar slots fed to the policy (tokens 0..11 : capteurs + coûts +
# récompense ; hors bilan métabolique, token 12).
N_POLICY_STATE = sum(_N_SIGNALS[:_INTERO_IDX])       # 51

# Per-token action-key → (skeleton attribute, field) lookup.
# Action tokens record the EFFECTIVE actuator values (theta_eff / d_eff) —
# the input the physics actually followed — not the consigne (theta_star /
# d_star), which is only the convergence target of the motor smoothing
# (body.py). This keeps the recorded (state, action) pairs causally exact:
# the world model learns the dynamics of what was really applied. The motor
# smoothing (consigne → effective) is a separate physical layer; recording
# theta_star instead would force the WM to infer the hidden actuator state
# from the consigne history, breaking the causal link between action and
# state transition (the effective value lags the consigne by ~0.5 s).
_ACTION_LOOKUP = {
    "membre_avant_theta":   ("limb_front", "theta_eff"),
    "membre_avant_d":       ("limb_front", "d_eff"),
    "membre_arriere_theta": ("limb_back", "theta_eff"),
    "membre_arriere_d":     ("limb_back", "d_eff"),
    "queue_theta":   ("tail_act", "theta_eff"),
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

# Per-slot channel weights of the EOS aggregate (brain.py): the usual innate
# cost hierarchy — a douleur spike in the denouement weighs more than a
# vertige wobble. Applies wherever the EOS is read as a scalar (relief score,
# saliency anchor, Dream Theater display); the slots themselves stay raw
# per-channel aggregates.
_EOS_CHANNEL_W = ([_COST_WEIGHTS[k] for k in _COST_KEYS]
                  + [_COST_WEIGHTS["confort"]])          # [1, 4, 2, 2, 3, 1]


def _prefix(ch: ChannelSpec) -> list[float]:
    """The 13-float structural prefix (type + modality + canal)."""
    return [1.0 if ch.is_motor else 0.0] + list(ch.modality) + list(ch.canal)


def _signal_slot(token: list[float], k: int) -> float:
    """Read the k-th signal slot (0..15) of a 29-dim token."""
    return token[TYPE_DIM + MOD_DIM + CANAL_DIM + k]


def state_tokens(salve: list[list[float]]) -> list[list[float]]:
    """The 13 sensory tokens of a salve."""
    return salve[:STATE_TOKENS]


def action_tokens(salve: list[list[float]]) -> list[list[float]]:
    """The 3 motor tokens of a salve."""
    return salve[STATE_TOKENS:SALVE_TOKENS]


def policy_scalars(salve: list[list[float]]) -> list[float]:
    """The 51 sensory + cost signal values (tokens 0..11, hors bilan
    métabolique), in order.

    These are the normalized [0,1] floats the policy consumes alongside the
    world-model latent."""
    out: list[float] = []
    for i in range(_INTERO_IDX):         # tokens 0..11 (exclude intero 12)
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

    A salve is a list of 16 tokens, each a 29-float vector.
    """

    def encode(self, proprio: dict, touch: dict, ball: dict,
              vision: dict, intero: dict, reward: dict,
              skel) -> list[list[float]]:
        dicts = {"proprio": proprio, "touch": touch, "ball": ball,
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
