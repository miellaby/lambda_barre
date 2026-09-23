"""IIR exponential smoother for sensor signals.

Each smoothed sensor signal is low-passed with a first-order IIR (exponential
moving average) with a time constant of ~1/3 s. The smoother is updated every
frame (60 Hz) and read once per world-model tick (3 Hz), producing a salve that
synthesizes the last second of sensory experience into a single token set.

Smoothing is a per-scalar decision: only noisy *derived* quantities (finite
differences, impulse-based forces, thresholded cost spikes) are smoothed.
Direct physical reads (positions, angles), signals already filtered or
integrated inside their sensor (vertige, courbature, sons, fatigue), and
reactivity-critical geometric signals (curseur_dir/prox) pass through raw —
smoothing them would only add lag.

This is the "compression temporelle" described in Lambda barre.md §
"Granularité temporelle du système et compression des actions": the world
model works at its own slow cadence, while the policy can act faster and
reuse the cached latent.
"""
from __future__ import annotations

import math

from .tokenize import _BY_KEY, DenseEncoder

# Time constant of the IIR filter (seconds). At 60 Hz, alpha = dt / (tau + dt).
TAU = 1.0 / 3.0

# Keys of all sensor signals the smoother tracks (everything except actions,
# which are read directly from the skeleton at tick time).
_ACTION_KEYS = {"membre_avant_theta", "membre_avant_d", "membre_arriere_theta", "membre_arriere_d",
                "queue_theta"}
_SENSOR_KEYS = [k for k in _BY_KEY if k not in _ACTION_KEYS]

# Scalars that actually benefit from the IIR: noisy derived quantities where
# the filter acts as a ~1/3 s perceptual integration window.
#   proprio: finite-difference accelerations + impulse-based forces
#   reward: effort (power = force × velocity) and douleur (thresholded spikes)
#   cursor: mouse velocity (finite difference, sampled at 30 Hz)
#   vision: optical flow (platform-rect differencing at 6 Hz)
# Touch signals (contact_sol_*, collision_tronc_*) are deliberately NOT
# smoothed: they must be instantaneous. Brief events falling between two WM
# ticks are lost by design — their effect still reaches the WM through the
# smoothed douleur channel (computed from the raw collisions every frame).
_SMOOTHED_KEYS = {
    "accel_tete_avant", "accel_tete_haut",
    "force_actuateur_avant", "force_actuateur_arriere", "couple_queue",
    "effort", "douleur",
    "curseur_vx", "curseur_vy",
    "flux_surface", "flux_x", "flux_y",
}
# Everything else passes through raw (no lag): direct physics reads
# (tronc_angle, queue_angle, membre_*, contact_sol_*, collision_tronc_*),
# already-slow internal integrators (courbature, vertige, fatigue,
# souffrance), direct geometric functions (instabilite, confort,
# curseur_dir, curseur_prox), spatially-averaged retina cells (vis_c*), and
# internally IIR-filtered sounds (son_*).


class Smoother:
    """One IIR filter per sensor signal. Update at 60 Hz, read at 1 Hz."""

    def __init__(self, tau: float = TAU):
        self._tau = tau
        self._state: dict[str, float] = {k: 0.0 for k in _SENSOR_KEYS}
        self._encoder = DenseEncoder()

    def reset(self) -> None:
        for k in self._state:
            self._state[k] = 0.0

    def reinit(self, proprio: dict, touch: dict, cursor: dict,
               vision: dict, intero: dict, reward: dict) -> None:
        """Immediately overwrite all filter states with the new sensor snapshot
        without EMA blending (used when a discrete reference-frame flip occurs)."""
        for k in _SENSOR_KEYS:
            raw = self._lookup(k, proprio, touch, cursor, vision, intero, reward)
            if raw is not None:
                self._state[k] = raw

    def update(self, proprio: dict, touch: dict, cursor: dict,
               vision: dict, intero: dict, reward: dict, dt: float) -> None:
        """Update the IIR filters with the latest sensor readings.

        Smoothed keys are EMA-blended; all other keys are stored raw
        (pass-through, no lag)."""
        alpha = dt / (self._tau + dt)
        for k in _SENSOR_KEYS:
            raw = self._lookup(k, proprio, touch, cursor, vision, intero, reward)
            if raw is None:
                continue
            if k in _SMOOTHED_KEYS:
                self._state[k] = (1 - alpha) * self._state[k] + alpha * raw
            else:
                self._state[k] = raw

    def _lookup(self, key, proprio, touch, cursor, vision, intero, reward):
        if key in proprio:
            return proprio[key]
        if key in touch:
            return touch[key]
        if key in cursor:
            return cursor[key]
        if key in vision:
            return vision[key]
        if key in intero:
            return intero[key]
        if key in reward:
            return reward[key]
        return None

    def salve(self, skel) -> list[list[float]]:
        """Produce a compressed salve from the smoothed sensor values +
        current actuator consignes (read live from the skeleton)."""
        return self._encoder.encode(
            self._state, self._state, self._state,
            self._state, self._state, self._state, skel)
