"""IIR exponential smoother for sensor signals.

Each sensor signal is smoothed with a first-order IIR (exponential moving
average) with a time constant of ~1 s. The smoother is updated every frame
(60 Hz) and read once per world-model tick (1 Hz), producing a salve that
synthesizes the last second of sensory experience into a single token set.

This is the "compression temporelle" described in Lambda barre.md §
"Granularité temporelle du système et compression des actions": the world
model works at its own slow cadence, while the policy can act faster and
reuse the cached latent.
"""
from __future__ import annotations

import math

from .tokenize import _BY_KEY, DenseEncoder

# Time constant of the IIR filter (seconds). At 60 Hz, alpha = dt / (tau + dt).
TAU = 1.0

# Keys of all sensor signals that get smoothed (everything except actions,
# which are read directly from the skeleton at tick time).
_ACTION_KEYS = {"limb_l_theta", "limb_l_d", "limb_r_theta", "limb_r_d",
                "tail_theta"}
_SENSOR_KEYS = [k for k in _BY_KEY if k not in _ACTION_KEYS]


class Smoother:
    """One IIR filter per sensor signal. Update at 60 Hz, read at 1 Hz."""

    def __init__(self, tau: float = TAU):
        self._tau = tau
        self._state: dict[str, float] = {k: 0.0 for k in _SENSOR_KEYS}
        self._encoder = DenseEncoder()

    def reset(self) -> None:
        for k in self._state:
            self._state[k] = 0.0

    def update(self, proprio: dict, touch: dict, cursor: dict,
               vision: dict, intero: dict, reward: dict, dt: float) -> None:
        """Update all IIR filters with the latest sensor readings."""
        alpha = dt / (self._tau + dt)
        for k in _SENSOR_KEYS:
            raw = self._lookup(k, proprio, touch, cursor, vision, intero, reward)
            if raw is not None:
                self._state[k] = (1 - alpha) * self._state[k] + alpha * raw

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
