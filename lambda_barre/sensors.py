"""Proprioception: measure the 11 egocentric body signals from the physics.

See proprioception.md for the full spec. The facing sign is an internal mirror
variable — never perceived. "Forward" = toward the head, always positive.

Usage:
    proprio = Proprio(space, skel)
    for _ in range(3):
        W.step(space, skel, dt)
    signals = proprio.update(skel, frame_dt)
"""
from __future__ import annotations

import math

import pymunk

from . import body as B


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
