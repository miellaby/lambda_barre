"""Intéroception: états internes de λ̄.

See interoception.md for the full spec. L'intéroception consomme les signaux
du circuit de renforcement (effort, douleur) et les accumule en états internes
lents (fatigue, souffrance). Contrairement à la proprioception et à
l'extéroception, l'intéroception n'est pas égocentrée.

Usage:
    intero = Intero()
    intero_signals = intero.update(reward_signals, frame_dt)
"""
from __future__ import annotations

import math

# --- constants ----------------------------------------------------------------
# Asymmetric leaky integrator: fast rise (saturates in ~30s at input=1),
# slow fall (half-life as specified). Rise and fall are decoupled.
FATIGUE_RISE = 1.0 / 60.0            # 1/s — saturates in ~60s at effort=1
FATIGUE_FALL = math.log(2) / 100.0   # 1/s — half-life 100s
SOUFFRANCE_RISE = 1.0 / 30.0         # 1/s — saturates in ~30s at douleur=1
SOUFFRANCE_FALL = math.log(2) / 240.0   # 1/s — half-life 240s


class Intero:
    """Computes the interoceptive signals.

    Fatigue and souffrance are asymmetric leaky integrators: they rise fast
    (saturate in ~30s at full input) and fall slowly (half-life 100s / 1000s).
    Both are clamped to [0, 1].

    Call ``update`` after ``reward.update`` each frame, passing the reward
    signals dict.
    """

    def __init__(self):
        self._fatigue = 0.0
        self._souffrance = 0.0

    def reset(self) -> None:
        self._fatigue = 0.0
        self._souffrance = 0.0

    def update(self, reward: dict, dt: float) -> dict:
        effort = reward.get("effort", 0.0)
        reward_neg = reward.get("reward_neg", 0.0)
        reward_pos = abs(reward.get("reward_pos", 0.0))

        self._fatigue += FATIGUE_RISE * effort * dt
        self._fatigue -= FATIGUE_FALL * self._fatigue * dt
        self._fatigue = max(0.0, min(1.0, self._fatigue))

        # reward_neg fait monter la souffrance, reward_pos et l'évaporation passive la font baisser
        self._souffrance += SOUFFRANCE_RISE * reward_neg * dt
        self._souffrance -= (SOUFFRANCE_FALL * self._souffrance + SOUFFRANCE_RISE * reward_pos) * dt
        self._souffrance = max(0.0, min(1.0, self._souffrance))

        return {
            "fatigue": self._fatigue,
            "souffrance": self._souffrance,
        }
