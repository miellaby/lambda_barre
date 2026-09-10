"""Token encoder for λ̄'s world model.

Each token is 2 bytes: 1 byte ID (the token type), 1 byte value (quantized
to 0-255, mapped from [0, 1] after per-signal normalization). Separator tokens
(PROPRIO, VISION, ...) have ID only — their value byte is 0 and ignored.

A salve is the full sequence of tokens produced at each world-model tick
(6 Hz): one group per modality, separated by separator tokens.

    [PROPRIO] tronc_angle membre_angle_avant ... [VISION] vis_c1_h ...
    [ENV] curseur_dir ... [TOUCH] contact_sol_avant ... [INTERO] fatigue ...
    [REWARD] effort ... [ACTION] limb_l_theta ...

Usage:
    enc = TokenEncoder()
    salve = enc.encode(proprio, touch, cursor, vision, intero, reward, skel)
    # salve is a list of (id, value) tuples (0-255 each)
    text = enc.decode(salve)  # list of short strings for display
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# --- separator token IDs ------------------------------------------------------
SEP_PROPRIO = 1
SEP_VISION   = 2
SEP_ENV      = 3
SEP_TOUCH    = 4
SEP_INTERO   = 5
SEP_REWARD   = 6
SEP_ACTION   = 7

_SEP_NAMES = {
    SEP_PROPRIO: "PROPRIO",
    SEP_VISION: "VISION",
    SEP_ENV: "ENV",
    SEP_TOUCH: "TOUCH",
    SEP_INTERO: "INTERO",
    SEP_REWARD: "REWARD",
    SEP_ACTION: "ACTION",
}

# --- scalar token definitions --------------------------------------------------
# Each entry: (id, key, code, scale, signed)
# scale maps the raw value to [-1, +1] if signed, or [0, 1] if unsigned.
# Quantization:  value_byte = clamp(quantize(raw / scale))
#   signed:   (raw / scale + 1) / 2 * 255
#   unsigned: raw / scale * 255

_START = 10  # scalar IDs start at 10

@dataclass
class TokenSpec:
    id: int
    key: str
    code: str          # 2-3 letter code for display
    scale: float       # normalization scale
    signed: bool       # whether the value can be negative


def _build_specs():
    specs: list[TokenSpec] = []
    sid = _START

    def add(key, code, scale, signed):
        nonlocal sid
        specs.append(TokenSpec(sid, key, code, scale, signed))
        sid += 1

    # Proprioception (11)
    add("tronc_angle",              "TA", math.radians(45), True)
    add("membre_angle_avant",       "AA", math.pi,          True)
    add("membre_distance_avant",    "DA", 32.0,             False)
    add("force_actuateur_avant",    "FA", 2000.0,           True)
    add("membre_angle_arriere",     "AR", math.pi,          True)
    add("membre_distance_arriere",  "DR", 32.0,             False)
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

    # Interoception (2)
    add("fatigue",     "FT", 1.0, False)
    add("souffrance",  "SF", 1.0, False)

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
    add("limb_l_theta", "LT", math.pi / 2, True)
    add("limb_l_d",     "LD", 32.0,        False)
    add("limb_r_theta", "RT", math.pi / 2, True)
    add("limb_r_d",     "RD", 32.0,        False)
    add("tail_theta",   "TQ", math.pi / 2, True)

    return specs


_SPECS = _build_specs()
_BY_KEY: dict[str, TokenSpec] = {s.key: s for s in _SPECS}
_BY_ID: dict[int, TokenSpec] = {s.id: s for s in _SPECS}


# --- salve layout (ids in the exact order produced by encode) ------------------
# Each group: (separator_id, [scalar_key, ...]). Replicates encode()'s order so
# the brain and the encoder agree on token positions without hardcoding ids.

_GROUP_LAYOUT = [
    (SEP_PROPRIO, [
        "tronc_angle", "membre_angle_avant", "membre_distance_avant",
        "force_actuateur_avant", "membre_angle_arriere",
        "membre_distance_arriere", "force_actuateur_arriere",
        "queue_angle", "couple_queue", "accel_tete_avant", "accel_tete_haut",
    ]),
    (SEP_VISION, [f"vis_c{i}" for i in range(1, 17)]
                 + ["flux_surface", "flux_x", "flux_y"]),
    (SEP_ENV, ["curseur_dir", "curseur_prox", "curseur_vx", "curseur_vy",
               "son_0", "son_1", "son_2", "son_3", "son_4"]),
    (SEP_TOUCH, ["contact_sol_avant", "contact_sol_arriere",
                 "collision_tronc_x", "collision_tronc_y",
                 "collision_tronc_cx", "collision_tronc_cy"]),
    (SEP_INTERO, ["fatigue", "souffrance"]),
    (SEP_REWARD, ["effort", "douleur", "courbature", "instabilite",
                  "vertige", "confort", "reward_pos", "reward_neg"]),
    (SEP_ACTION, ["limb_l_theta", "limb_l_d", "limb_r_theta",
                  "limb_r_d", "tail_theta"]),
]

# Flat id layout of a full 67-token salve (separators + scalars, in order).
SALVE_IDS: list[int] = []
for sep_id, keys in _GROUP_LAYOUT:
    SALVE_IDS.append(sep_id)
    SALVE_IDS.extend(_BY_KEY[k].id for k in keys)

# Position slices within a salve. The "state" is everything up to and including
# the REWARD group; the "action" is the last group.
STATE_LEN = sum(1 + len(keys) for sep_id, keys in _GROUP_LAYOUT[:-1])   # 61
ACTION_LEN = 1 + len(_GROUP_LAYOUT[-1][1])                              # 6
SALVE_LEN = STATE_LEN + ACTION_LEN                                      # 67

STATE_IDS = SALVE_IDS[:STATE_LEN]
ACTION_IDS = SALVE_IDS[STATE_LEN:]

# Indices (within the state part) of the scalar tokens whose values carry a
# valence, used to read the predicted cost off an imagined next state.
_REWARD_KEYS = _GROUP_LAYOUT[5][1]  # effort..reward_neg
REWARD_SCALAR_OFFSETS = [i for i, tid in enumerate(STATE_IDS)
                        if tid in _BY_ID and _BY_ID[tid].key in _REWARD_KEYS]
# offset, within the state id layout, of the reward_pos / reward_neg tokens
_REWARD_POS_OFF = STATE_IDS.index(_BY_KEY["reward_pos"].id)
_REWARD_NEG_OFF = STATE_IDS.index(_BY_KEY["reward_neg"].id)

# Indices of the scalar (non-separator) tokens within a salve — for the policy
# state vector (the 55 sensory scalar values, in order).
SCALAR_IDS = [tid for tid in SALVE_IDS if tid in _BY_ID]


def dequantize(val: int, key: str) -> float:
    """Dequantize a token value back to its physical value given the signal key."""
    return _dequantize(val, _BY_KEY[key])


def encode_action(theta_l: float, d_l: float, theta_r: float, d_r: float,
                  tail_t: float) -> list[tuple[int, int]]:
    """Encode the 5 actuator consignes into the ACTION group tokens
    (separator + 5 scalars). Used by the brain to tokenize imagined actions
    without going through the skeleton."""
    salve = [(SEP_ACTION, 0)]
    for key, raw in (("limb_l_theta", theta_l), ("limb_l_d", d_l),
                     ("limb_r_theta", theta_r), ("limb_r_d", d_r),
                     ("tail_theta", tail_t)):
        spec = _BY_KEY[key]
        salve.append((spec.id, _quantize(raw, spec)))
    return salve


def state_values(salve: list[tuple[int, int]]) -> list[int]:
    """The quantized values of the 55 sensory scalar tokens (state part,
    separators excluded), in salve order. Feeds the policy as a flat vector."""
    out: list[int] = []
    for tid, val in salve[:STATE_LEN]:
        if tid in _BY_ID:
            out.append(val)
    return out


def action_values(salve: list[tuple[int, int]]) -> list[int]:
    """The quantized values of the 5 action scalar tokens (last 5 of the salve)."""
    return [val for tid, val in salve[STATE_LEN:] if tid in _BY_ID]


def state_token_values(salve: list[tuple[int, int]]) -> list[int]:
    """The 61 token values of the state part (separators included, value 0),
    in state-token order — what the world model consumes."""
    return [val for tid, val in salve[:STATE_LEN]]


def action_token_values(salve: list[tuple[int, int]]) -> list[int]:
    """The 6 token values of the action part (separator included, value 0)."""
    return [val for tid, val in salve[STATE_LEN:]]


# positions (within a 61-token state) that are scalars (not separators)
_STATE_SCALAR_POS = [i for i, tid in enumerate(STATE_IDS) if tid in _BY_ID]


def scalars_from_state_vals(state_vals: list[int]) -> list[int]:
    """Extract the 55 scalar values from a 61-token state-value list
    (dropping separator positions). Inverse of inserting 0s at separators."""
    return [state_vals[i] for i in _STATE_SCALAR_POS]


def state_vals_from_scalars(scalars: list[int]) -> list[int]:
    """Rebuild a 61-token state-value list (separators=0) from 55 scalar values."""
    out = [0] * STATE_LEN
    for pos, v in zip(_STATE_SCALAR_POS, scalars):
        out[pos] = v
    return out


def salve_cost(salve: list[tuple[int, int]]) -> float:
    """Instantaneous cost carried by a salve's REWARD tokens.

    cost = reward_neg (unsigned cost) + reward_pos (signed; negative = reward),
    both dequantized to their physical scale (1.0). Lower is better.
    """
    rp = dequantize(salve[_REWARD_POS_OFF][1], "reward_pos")
    rn = dequantize(salve[_REWARD_NEG_OFF][1], "reward_neg")
    return rn + rp


def _quantize(raw: float, spec: TokenSpec) -> int:
    if spec.signed:
        t = raw / spec.scale
        t = max(-1.0, min(1.0, t))
        return int(round((t + 1.0) / 2.0 * 255))
    else:
        t = max(0.0, min(1.0, raw / spec.scale))
        return int(round(t * 255))


def _dequantize(val: int, spec: TokenSpec) -> float:
    if spec.signed:
        t = (val / 255.0) * 2.0 - 1.0
        return t * spec.scale
    else:
        return (val / 255.0) * spec.scale


class TokenEncoder:
    """Encodes a full sensor snapshot into a salve of 2-byte tokens.

    A salve is a list of (id, value) tuples, each 0-255. Separator tokens
    appear as (sep_id, 0).
    """

    def encode(self, proprio: dict, touch: dict, cursor: dict,
              vision: dict, intero: dict, reward: dict,
              skel) -> list[tuple[int, int]]:
        salve: list[tuple[int, int]] = []

        # PROPRIO
        salve.append((SEP_PROPRIO, 0))
        for key in ("tronc_angle", "membre_angle_avant", "membre_distance_avant",
                     "force_actuateur_avant", "membre_angle_arriere",
                     "membre_distance_arriere", "force_actuateur_arriere",
                     "queue_angle", "couple_queue",
                     "accel_tete_avant", "accel_tete_haut"):
            spec = _BY_KEY[key]
            salve.append((spec.id, _quantize(proprio.get(key, 0.0), spec)))

        # VISION
        salve.append((SEP_VISION, 0))
        for i in range(1, 17):
            key = f"vis_c{i}"
            spec = _BY_KEY[key]
            salve.append((spec.id, _quantize(vision.get(key, 0.0), spec)))
        for key in ("flux_surface", "flux_x", "flux_y"):
            spec = _BY_KEY[key]
            salve.append((spec.id, _quantize(vision.get(key, 0.0), spec)))

        # ENV (cursor + sound)
        salve.append((SEP_ENV, 0))
        for key in ("curseur_dir", "curseur_prox", "curseur_vx", "curseur_vy",
                     "son_0", "son_1", "son_2", "son_3", "son_4"):
            spec = _BY_KEY[key]
            salve.append((spec.id, _quantize(cursor.get(key, 0.0), spec)))

        # TOUCH
        salve.append((SEP_TOUCH, 0))
        for key in ("contact_sol_avant", "contact_sol_arriere",
                     "collision_tronc_x", "collision_tronc_y",
                     "collision_tronc_cx", "collision_tronc_cy"):
            spec = _BY_KEY[key]
            salve.append((spec.id, _quantize(touch.get(key, 0.0), spec)))

        # INTERO
        salve.append((SEP_INTERO, 0))
        for key in ("fatigue", "souffrance"):
            spec = _BY_KEY[key]
            salve.append((spec.id, _quantize(intero.get(key, 0.0), spec)))

        # REWARD
        salve.append((SEP_REWARD, 0))
        for key in ("effort", "douleur", "courbature", "instabilite",
                     "vertige", "confort", "reward_pos", "reward_neg"):
            spec = _BY_KEY[key]
            salve.append((spec.id, _quantize(reward.get(key, 0.0), spec)))

        # ACTION
        salve.append((SEP_ACTION, 0))
        actions = {
            "limb_l_theta": skel.limb_l.theta_star,
            "limb_l_d":     skel.limb_l.d_star,
            "limb_r_theta": skel.limb_r.theta_star,
            "limb_r_d":     skel.limb_r.d_star,
            "tail_theta":  skel.tail_act.theta_star,
        }
        for key in ("limb_l_theta", "limb_l_d", "limb_r_theta",
                     "limb_r_d", "tail_theta"):
            spec = _BY_KEY[key]
            salve.append((spec.id, _quantize(actions[key], spec)))

        return salve

    def decode(self, salve: list[tuple[int, int]]) -> list[str]:
        """Decode a salve into a list of short display strings.
        Separators appear as '[NAME]', scalars as 'CODE=123'."""
        lines: list[str] = []
        for tid, val in salve:
            if tid in _SEP_NAMES:
                lines.append(f"[{_SEP_NAMES[tid]}]")
            else:
                spec = _BY_ID.get(tid)
                if spec:
                    lines.append(f"{spec.code}={val:3d}")
                else:
                    lines.append(f"?{tid}={val:3d}")
        return lines

    def decode_grouped(self, salve: list[tuple[int, int]]) -> list[tuple[str, list[str]]]:
        """Decode into (group_name, [token_strings]) pairs."""
        groups: list[tuple[str, list[str]]] = []
        current_name = ""
        current_tokens: list[str] = []
        for tid, val in salve:
            if tid in _SEP_NAMES:
                if current_name:
                    groups.append((current_name, current_tokens))
                current_name = _SEP_NAMES[tid]
                current_tokens = []
            else:
                spec = _BY_ID.get(tid)
                if spec:
                    current_tokens.append(f"{spec.code}={val:3d}")
                else:
                    current_tokens.append(f"?{tid}={val:3d}")
        if current_name:
            groups.append((current_name, current_tokens))
        return groups
