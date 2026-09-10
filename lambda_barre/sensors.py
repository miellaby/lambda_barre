"""Proprioception and proprioceptive reward circuit.

See proprioception.md for the full spec. The facing sign is an internal mirror
variable — never perceived. "Forward" = toward the head, always positive.

Usage:
    proprio = Proprio(space, skel)
    reward = Reward(skel, proprio)
    for _ in range(3):
        W.step(space, skel, dt)
    signals = proprio.update(skel, frame_dt)
    reward_signals = reward.update(skel, signals, touch_signals, frame_dt)
"""
from __future__ import annotations

import math
from collections import deque

import pymunk

from . import body as B

# --- Reward circuit constants ------------------------------------------------
EFFORT_SCALE = 6000.0             # power at which effort = 0.5
DOULEUR_SEUIL_COLLISION = 800.0   # impulse/dt threshold for trunk collision
DOULEUR_SEUIL_ACCEL = 1500.0      # accel threshold for head
DOULEUR_SCALE = 6000.0            # douleur at which pain = 0.5
COURBATURE_WINDOW = 1.0           # seconds — proprio variance window
COURBATURE_SEUIL_VAR = 5.0        # below this variance, courbature accrues
COURBATURE_ACCRUE = 0.005          # accrual rate when immobile (~4 min to saturate)
COURBATURE_DECAY = 0.80           # per-frame decay when moving
INSTABILITE_K = 0.01              # marge → cost scale
VERTIGE_SCALE = 5.0             # omega at which vertige = 0.5
CONFORT_K = 0.3                   # tronc_angle² → reward scale
CONFORT_DEG = math.radians(10.0)  # comfort zone around upright


def _limb_measure(torso, foot, hip_local):
    """Return (theta_torso, d) measured from the real foot position."""
    ol = torso.world_to_local(foot.position)
    ox = ol.x - hip_local[0]
    oy = ol.y - hip_local[1]
    d = math.hypot(ox, oy)
    theta = math.atan2(-ox, oy)  # same convention as theta_star (torso-local)
    return theta, d


class Proprio:
    """Computes the 11 proprioceptive signals.

    Call ``update`` after the physics substeps to get the snapshot for this
    frame.
    """

    def __init__(self, space: pymunk.Space, skel: "B.Skeleton",
                 substep_dt: float = 1 / 180):
        self.space = space
        self.skel = skel
        self._substep_dt = substep_dt
        # head position history for finite-difference acceleration
        head = B.head_world(skel)
        self._head_prev = (head.x, head.y)
        self._has_prev = False

    def reset(self) -> None:
        """Call after B.reset(skel) to re-sync the head history."""
        head = B.head_world(self.skel)
        self._head_prev = (head.x, head.y)
        self._has_prev = False

    def update(self, skel: "B.Skeleton", dt: float) -> dict:
        """Compute the 11 egocentric signals. Call after the physics substeps."""
        facing = -skel.facing
        torso = skel.torso

        # --- limbs: which is front / back depends on facing ---
        if facing == -1:
            foot_front, hip_front = skel.foot_r, B.HIP_R
            foot_back, hip_back = skel.foot_l, B.HIP_L
            spring_front = skel.spring_r
            spring_back = skel.spring_l
        else:
            foot_front, hip_front = skel.foot_l, B.HIP_L
            foot_back, hip_back = skel.foot_r, B.HIP_R
            spring_front = skel.spring_l
            spring_back = skel.spring_r

        th_f, d_f = _limb_measure(torso, foot_front, hip_front)
        th_b, d_b = _limb_measure(torso, foot_back, hip_back)

        # --- tail ---
        tail_rel = skel.tail.angle - torso.angle
        tail_neutral = -facing * (math.pi / 2)
        queue_angle = tail_rel - tail_neutral

        # --- head acceleration (finite difference, world frame) ---
        head = B.head_world(skel)
        if self._has_prev and dt > 0:
            ax = (head.x - self._head_prev[0]) / dt
            ay = (head.y - self._head_prev[1]) / dt
        else:
            ax = ay = 0.0
        self._head_prev = (head.x, head.y)
        self._has_prev = True

        # --- actuator forces (impulse / substep_dt) ---
        sdt = self._substep_dt
        f_spring_f = spring_front.impulse / sdt if sdt > 0 else 0.0
        f_spring_b = spring_back.impulse / sdt if sdt > 0 else 0.0
        torque_tail = skel.tail_spring.impulse / sdt if sdt > 0 else 0.0

        return {
            "tronc_angle": facing * torso.angle,
            "membre_angle_avant": facing * th_f,
            "membre_distance_avant": d_f,
            "force_actuateur_avant": f_spring_f,
            "membre_angle_arriere": facing * th_b,
            "membre_distance_arriere": d_b,
            "force_actuateur_arriere": f_spring_b,
            "queue_angle": queue_angle,
            "couple_queue": facing * torque_tail,
            "accel_tete_avant": facing * ax,
            "accel_tete_haut": ay,
        }


# --- Reward circuit ----------------------------------------------------------
#
# Computes the 6 instantaneous proprioceptive costs described in
# proprioception.md § "Circuit de renforcement":
#   effort, douleur, courbature, instabilite, vertige  (positive = cost)
#   confort                                          (negative = reward)
#
# The reward is a single signed scalar: positive = penalty, negative = reward.
# reward_pos = confort (reward only)
# reward_neg = effort + douleur + courbature + instabilite + vertige (cost only)

# proprio keys tracked for the courbature variance window
_VARIANCE_KEYS = (
    "tronc_angle", "membre_angle_avant", "membre_angle_arriere",
    "membre_distance_avant", "membre_distance_arriere", "queue_angle",
)


class Reward:
    """Proprioceptive reward circuit.

    Consumes the proprio and touch snapshots (not the physics directly) and
    produces the 6 instantaneous costs + the aggregated positive/negative
    reward. Call after ``proprio.update`` and ``touch.update`` each frame.

    Signals produced:
        effort, douleur, courbature, instabilite, vertige, confort
        reward_pos  — confort (negative = reward, clamped to 0 from below)
        reward_neg  — sum of costs (positive = penalty, clamped to 0 from above)
    """

    def __init__(self, skel: "B.Skeleton", substep_dt: float = 1 / 180):
        self._substep_dt = substep_dt
        # courbature: rolling proprio history for variance
        n = max(2, int(COURBATURE_WINDOW / (1 / 60)))
        self._history: deque[dict] = deque(maxlen=n)
        self._courbature = 0.0
        # vertige: smoothed angular velocity
        self._omega_smooth = 0.0

    def reset(self) -> None:
        self._history.clear()
        self._courbature = 0.0
        self._omega_smooth = 0.0

    def update(self, skel: "B.Skeleton", proprio: dict, touch: dict,
               dt: float) -> dict:
        facing = skel.facing

        # --- effort (power = force * velocity) ---
        # At rest (immobile), velocity = 0 → effort = 0, regardless of the
        # spring tension holding the body up.
        if facing == -1:
            foot_front = skel.foot_r
            foot_back = skel.foot_l
        else:
            foot_front = skel.foot_l
            foot_back = skel.foot_r
        f_av = abs(proprio.get("force_actuateur_avant", 0.0))
        f_ar = abs(proprio.get("force_actuateur_arriere", 0.0))
        cq = abs(proprio.get("couple_queue", 0.0))
        v_av = foot_front.velocity.length
        v_ar = foot_back.velocity.length
        omega_tail = abs(skel.tail.angular_velocity)
        power = f_av * v_av + f_ar * v_ar + cq * omega_tail
        effort = power / (power + EFFORT_SCALE)

        # --- douleur ---
        coll = abs(touch.get("collision_tronc_x", 0.0)) + abs(
            touch.get("collision_tronc_y", 0.0))
        accel = abs(proprio.get("accel_tete_avant", 0.0)) + abs(
            proprio.get("accel_tete_haut", 0.0))
        douleur = (
            max(0.0, coll - DOULEUR_SEUIL_COLLISION)
            + max(0.0, accel - DOULEUR_SEUIL_ACCEL)
        )
        douleur = douleur / (douleur + DOULEUR_SCALE)

        # --- courbature ---
        # Clamped to [0, 1]. Accrues when proprio variance is low (immobile),
        # decays rapidly when variance exceeds the threshold (moving).
        self._history.append({k: proprio.get(k, 0.0) for k in _VARIANCE_KEYS})
        if len(self._history) >= 2:
            vars_sum = 0.0
            for k in _VARIANCE_KEYS:
                vals = [h[k] for h in self._history]
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                vars_sum += var
            var_total = math.sqrt(vars_sum)
        else:
            var_total = 0.0
        if var_total < COURBATURE_SEUIL_VAR:
            self._courbature += COURBATURE_ACCRUE * (
                1.0 - var_total / COURBATURE_SEUIL_VAR) * dt
        else:
            self._courbature *= COURBATURE_DECAY
        self._courbature = max(0.0, min(1.0, self._courbature))
        courbature = self._courbature

        # --- instabilite ---
        fx_front = foot_front.position.x
        fx_back = foot_back.position.x
        com_x = skel.torso.position.x
        left = min(fx_front, fx_back)
        right = max(fx_front, fx_back)
        marge = min(com_x - left, right - com_x)
        instabilite = INSTABILITE_K * max(0.0, -marge)

        # --- vertige (smoothed angular velocity) ---
        omega_raw = abs(skel.torso.angular_velocity)
        self._omega_smooth = self._omega_smooth * 0.85 + omega_raw * 0.15
        vertige = self._omega_smooth / (self._omega_smooth + VERTIGE_SCALE)

        # --- confort ---
        tronc_angle = abs(proprio.get("tronc_angle", 0.0))
        if tronc_angle < CONFORT_DEG:
            confort = -CONFORT_K * (CONFORT_DEG - tronc_angle) / CONFORT_DEG
        else:
            confort = 0.0

        reward_neg = effort + douleur + courbature + instabilite + vertige
        reward_pos = confort  # already negative (reward)

        return {
            "effort": effort,
            "douleur": douleur,
            "courbature": courbature,
            "instabilite": instabilite,
            "vertige": vertige,
            "confort": confort,
            "reward_pos": reward_pos,
            "reward_neg": reward_neg,
        }
