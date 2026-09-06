"""World setup: gravity, ground, and the simulation step."""
from __future__ import annotations

import math

import pymunk

from . import body as B

# platforms: (centre_x, centre_y, half_width, radius)
PLATFORM_DEFS = [
    (360, -40, 90, 16),
    (-360, 60, 90, 16),
    (45, 78, 20, 12),
]

# moving platform: oscillates horizontally like an old platformer
MOVING_PLATFORM_DEF = (190, 40, 40, 16)   # (cx, cy, half_w, radius)
MOVING_PLATFORM_AMP = 120.0             # px, amplitude of oscillation
MOVING_PLATFORM_PERIOD = 4.0            # seconds, full back-and-forth


def make_space() -> pymunk.Space:
    space = pymunk.Space()
    space.gravity = (0, B.GRAVITY)
    # ground
    ground = pymunk.Body(body_type=pymunk.Body.STATIC)
    ground_shape = pymunk.Segment(ground, (-4000, B.GROUND_Y), (4000, B.GROUND_Y), 4)
    ground_shape.friction = 1.2
    ground_shape.elasticity = 0.0
    ground_shape.collision_type = B.GROUND_TYPE
    space.add(ground, ground_shape)
    # platforms — static so they don't drift, repositioned at runtime by
    # setting body.position + space.reindex_shape()
    platforms: list[tuple[pymunk.Body, pymunk.Segment, float, float]] = []
    for cx, cy, w, h in PLATFORM_DEFS:
        plat = pymunk.Body(body_type=pymunk.Body.STATIC)
        plat.position = (cx, cy)
        s = pymunk.Segment(plat, (-w, 0), (w, 0), h)
        s.friction = 1.0
        s.collision_type = B.GROUND_TYPE
        space.add(plat, s)
        platforms.append((plat, s, w, h))
    # moving platform (also static, repositioned each frame)
    cx, cy, w, h = MOVING_PLATFORM_DEF
    mv_plat = pymunk.Body(body_type=pymunk.Body.STATIC)
    mv_plat.position = (cx, cy)
    mv_seg = pymunk.Segment(mv_plat, (-w, 0), (w, 0), h)
    mv_seg.friction = 1.0
    mv_seg.collision_type = B.GROUND_TYPE
    space.add(mv_plat, mv_seg)
    platforms.append((mv_plat, mv_seg, w, h))
    space._platforms = platforms  # type: ignore[attr-defined]
    space._moving_platform = (mv_plat, mv_seg)  # type: ignore[attr-defined]
    return space


def update_moving_platform(space: pymunk.Space, t: float) -> None:
    """Reposition the moving platform for time t (seconds since start)."""
    mv_plat, mv_seg = space._moving_platform  # type: ignore[attr-defined]
    cx, cy, _, _ = MOVING_PLATFORM_DEF
    offset = MOVING_PLATFORM_AMP * math.sin(2 * math.pi * t / MOVING_PLATFORM_PERIOD)
    mv_plat.position = (cx + offset, cy)
    space.reindex_shape(mv_seg)


def step(space: pymunk.Space, skel: "B.Skeleton", dt: float) -> None:
    """Advance the world by dt, applying the current consignes first."""
    B.apply_consignes(skel)
    space.step(dt)
