"""Proprioception: measure the 13 egocentric body signals from the physics.

See proprioception.md for the full spec. The facing sign is an internal mirror
variable — never perceived. "Forward" = toward the head, always positive.

Usage:
    proprio = Proprio(space, skel)
    proprio.reset_contacts()        # before the physics substeps
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
    """Accumulates contact impulses and computes the 13 proprioceptive signals.

    The collision handler is attached once at construction. Call
    ``reset_contacts`` before the physics substeps, then ``update`` after them
    to get the snapshot for this frame.
    """

    def __init__(self, space: pymunk.Space, skel: "B.Skeleton",
                 substep_dt: float = 1 / 180):
        self.space = space
        self.skel = skel
        self._substep_dt = substep_dt
        # accumulated normal impulse per foot body over the frame's substeps
        self._impulse: dict = {skel.foot_l: 0.0, skel.foot_r: 0.0}
        # head position history for finite-difference acceleration
        head = skel.torso.local_to_world(B.HEAD_ANCHOR)
        self._head_prev = (head.x, head.y)
        self._has_prev = False

        space.on_collision(B.FOOT_TYPE, B.GROUND_TYPE,
                           post_solve=self._on_post_solve)

    def _on_post_solve(self, arb: pymunk.Arbiter, space, data):
        ny = arb.total_impulse.y
        for shape in arb.shapes:
            body = shape.body
            if body in self._impulse:
                self._impulse[body] += ny

    def reset_contacts(self) -> None:
        self._impulse[self.skel.foot_l] = 0.0
        self._impulse[self.skel.foot_r] = 0.0

    def reset(self) -> None:
        """Call after B.reset(skel) to re-sync the head history."""
        head = self.skel.torso.local_to_world(B.HEAD_ANCHOR)
        self._head_prev = (head.x, head.y)
        self._has_prev = False
        self.reset_contacts()

    def update(self, skel: "B.Skeleton", dt: float) -> dict:
        """Compute the 13 egocentric signals. Call after the physics substeps."""
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
        head = torso.local_to_world(B.HEAD_ANCHOR)
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

        # --- contact force (impulse / dt) ---
        f_contact_f = self._impulse[foot_front] / dt if dt > 0 else 0.0
        f_contact_b = self._impulse[foot_back] / dt if dt > 0 else 0.0

        return {
            "tronc_angle": facing * torso.angle,
            "membre_angle_avant": facing * th_f,
            "membre_distance_avant": d_f,
            "force_actuateur_avant": f_spring_f,
            "force_contact_sol_avant": f_contact_f,
            "membre_angle_arriere": facing * th_b,
            "membre_distance_arriere": d_b,
            "force_actuateur_arriere": f_spring_b,
            "force_contact_sol_arriere": f_contact_b,
            "queue_angle": queue_angle,
            "couple_queue": facing * torque_tail,
            "accel_tete_avant": facing * ax,
            "accel_tete_haut": ay,
        }
