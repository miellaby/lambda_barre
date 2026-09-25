"""World setup: gravity, ground, and the simulation step."""
from __future__ import annotations

import math

import pymunk

from . import body as B

# platforms: (centre_x, centre_y, half_width, radius)
PLATFORM_DEFS = [
    (360, 60, 90, 16),
    (180, 40, 40, 12),
    (-360, 60, 90, 16),
    (-180, 40, 40, 12),
]

BALL_TYPE = 4
BALL_RADIUS = 16.0
BALL_MASS = 0.5
BALL_SPAWN = (-360.0, 92.0)



def create_ball(space: pymunk.Space, pos: tuple[float, float] = BALL_SPAWN,
                radius: float = BALL_RADIUS, mass: float = BALL_MASS) -> tuple[pymunk.Body, pymunk.Circle]:
    """Create a dynamic red ball subject to gravity, with bounce and friction."""
    moment = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, moment)
    body.position = pos
    shape = pymunk.Circle(body, radius)
    shape.friction = 0.7
    shape.elasticity = 0.75
    shape.collision_type = BALL_TYPE
    shape.filter = pymunk.ShapeFilter(group=0)
    space.add(body, shape)
    space.ball = body               # type: ignore[attr-defined]
    space.ball_shape = shape         # type: ignore[attr-defined]
    space.ball_spawn = pos           # type: ignore[attr-defined]

    def pre_solve_ball(arb, space, data):
        arb.restitution = 0.75
        arb.friction = 0.70
        return True

    space.on_collision(BALL_TYPE, B.GROUND_TYPE, pre_solve=pre_solve_ball)
    space.on_collision(BALL_TYPE, B.TORSO_TYPE, pre_solve=pre_solve_ball)
    space.on_collision(BALL_TYPE, B.FOOT_TYPE, pre_solve=pre_solve_ball)

    return body, shape


def reset_ball(space: pymunk.Space, pos: tuple[float, float] | None = None) -> None:
    """Reset the ball position and zero its velocity."""
    if hasattr(space, "ball") and space.ball is not None:
        p = pos if pos is not None else getattr(space, "ball_spawn", BALL_SPAWN)
        space.ball.position = p
        space.ball.velocity = (0.0, 0.0)
        space.ball.angular_velocity = 0.0
        if hasattr(space, "ball_shape") and space.ball_shape is not None:
            space.reindex_shape(space.ball_shape)


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
    space._platforms = platforms  # type: ignore[attr-defined]

    # mobile red balloon
    create_ball(space)
    return space


def step(space: pymunk.Space, skel: "B.Skeleton", dt: float) -> None:
    """Advance the world by dt, applying the current consignes first."""
    B.apply_consignes(skel)
    space.step(dt)
    if hasattr(space, "ball") and space.ball is not None:
        ball = space.ball
        # Rolling friction: Chipmunk circles roll indefinitely without slipping,
        # so damping angular rotation allows surface Coulomb friction to stop the roll.
        ball.angular_velocity *= 0.96
        if abs(ball.velocity.x) < 2.0 and abs(ball.angular_velocity) < 0.2:
            ball.velocity = (0.0, ball.velocity.y)
            ball.angular_velocity = 0.0
        bp = ball.position
        if bp.y < -50.0 or abs(bp.x) > 3000.0:
            reset_ball(space)

