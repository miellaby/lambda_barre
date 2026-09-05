"""World setup: gravity, ground, and the simulation step."""
from __future__ import annotations

import pymunk

from . import body as B


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
    # a couple of static platforms to play on (forward / backward)
    for cx, w, h in [((360, -40), 90, 4), ((-360, 60), 90, 4)]:
        plat = pymunk.Body(body_type=pymunk.Body.STATIC)
        s = pymunk.Segment(plat, (cx[0] - w, cx[1]), (cx[0] + w, cx[1]), h)
        s.friction = 1.0
        s.collision_type = B.GROUND_TYPE
        space.add(plat, s)
    return space


def step(space: pymunk.Space, skel: "B.Skeleton", dt: float) -> None:
    """Advance the world by dt, applying the current consignes first."""
    B.apply_consignes(skel)
    space.step(dt)


def foot_contacts(skel: "B.Skeleton") -> tuple[bool, bool]:
    """Which hind feet are touching the ground (for the touch sensor later).

    Robust positional check: a foot touches ground if its lowest point is at or
    below the ground height. A proper contact query (space.bb_query or
    collision callbacks) replaces this in Stage 2.
    """
    out = []
    for foot in (skel.foot_l, skel.foot_r):
        out.append(foot.position.y - B.LIMB_RADIUS <= B.GROUND_Y + 1.5)
    return tuple(out)  # type: ignore[return-value]
