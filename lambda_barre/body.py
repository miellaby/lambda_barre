"""The λ̄ body: a pymunk skeleton and its actuator consignes.

Morphology (from the design doc):
  - torso: a triangular rigid body. Its angle relative to the vertical defines
    the orientation of the whole animat.
  - two hind limbs: each is a single segment positioned relative to the torso
    by an actuator with two continuous commands (theta_i, d_i) — the angle and
    the distance from the hip attachment point. The network later produces
    these consignes; the physics engine turns them into forces/motion.
  - tail: a single segment with one angular degree of freedom, acting as a
    counterweight.
  - head, ears, front legs: aesthetic only, drawn relative to the torso, never
    actuated.

Actuator model (doc, "Mise en oeuvre"):
  The actuator is a *consigne* (setpoint), not an instantaneous constraint. The
  network says "I would like this limb in this configuration"; a PD controller
  turns that into force/torque:
      tau_theta = k_theta * (theta* - theta) - c_theta * theta_dot
      F_d       = k_d       * (d*     - d)     - c_d       * d_dot

Implementation note: each hind limb is modelled as a foot body pulled toward a
target point (hip + d* along direction theta*) in the torso frame by a pymunk
DampedSpring whose rest length and anchor encode (theta*, d*). A positional PD
spring to a moving target point is the stable, faithful realisation of the
two-DOF (angle, length) consigne; the separate k_theta / k_d gains of the doc
are approximated here by the spring's stiffness and damping and can be split
into independent angular/radial gains in a later stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import pymunk

# --- world constants ---------------------------------------------------------
GROUND_Y = 0.0          # ground height (pymunk coords, y up)
GRAVITY = -900.0        # px/s^2

# --- morphology (pymunk units, ~pixels) --------------------------------------
TORSO_MASS = 10.0
TORSO_VERTS = [(-14, -22), (14, -22), (6, 26)]      # triangle, base at hips, apex up
LIMB_MASS = 3.0
LIMB_MAX = 32.0                                    # max segment length
LIMB_MIN = 10.0
LIMB_RADIUS = 6.0                                  # foot collision radius
TAIL_MASS = 10.0
TAIL_LEN = 46.0
TAIL_RADIUS = 3.0

# ears: aesthetic, with rotary springs so they wobble naturally
EAR_MASS = 0.04
EAR_LEN = 16.0
EAR_BASE = 10.0
EAR_ATTACH_L = (2, 32)       # torso-local, high behind the head
EAR_ATTACH_R = (10, 32)
EAR_REST_L = 2.3             # rad relative to torso — backward-up, slightly less reclined
EAR_REST_R = 2.5             # slightly more reclined
EAR_STIFFNESS = 300.0
EAR_DAMPING = 10.0

# front legs: aesthetic circles hanging from torso center via rotary spring
FRONT_MASS = 1.0
FRONT_RADIUS = 5.0
FRONT_LEN = 16.0             # distance from pivot to circle center
FRONT_REST = 0.2             # rad relative to torso — slightly forward
FRONT_STIFFNESS = 500.0
FRONT_DAMPING = 20.0
FRONT_PIVOT = (0, 0)         # center of torso

# hip attachment points on the torso (torso-local coords)
HIP_L = (-7, -20)
HIP_R = (7, -20)
TAIL_ATTACH = (0, -16)
# head anchor (drawn only) — offset to the right of torso center; mirrored
# by facing at every call site so the head follows the orientation.
HEAD_ANCHOR = (7, 24)


def head_world(skel: "Skeleton") -> pymunk.Vec2d:
    """World-space head position, correctly mirrored by facing direction."""
    return skel.torso.local_to_world(
        (HEAD_ANCHOR[0] * skel.facing, HEAD_ANCHOR[1]))

# --- actuator gains (PD) ------------------------------------------------------
# Hind limb: positional spring toward target foot point.
LIMB_STIFFNESS = 5000.0     # ~ k_d
LIMB_DAMPING = 20.0         # ~ c_d
# Tail: rotary spring toward target angle.
TAIL_STIFFNESS = 250000.0      # ~ k_theta — stiff enough to hold against gravity
TAIL_DAMPING = 12000.0         # ~ c_theta — ~0.8 critical damping for crisp response

# All animat body parts share this collision group so they never collide with
# each other (only with the ground / environment). Without this, overlapping
# shapes fight their own joints and the body explodes apart.
ANIMAT_GROUP = 1

# collision types for contact sensing
FOOT_TYPE = 1
GROUND_TYPE = 2
TORSO_TYPE = 3

# facing direction hysteresis (thermostat): the animat only "turns around" when
# its torso tilt exceeds ±15°, so small oscillations near upright don't flip
# the tail/head/ears back and forth.
FACING_DEADZONE = math.radians(15.0)


def _animat_filter() -> pymunk.ShapeFilter:
    return pymunk.ShapeFilter(group=ANIMAT_GROUP)


@dataclass
class LimbActuator:
    """Consignes (target setpoints) for one hind limb, in torso-local frame.

    theta_star: target angle of the limb, radians, measured from torso's
        downward axis (so 0 = foot straight down, pi/2 = foot forward).
    d_star: target foot distance from the hip.
    """
    theta_star: float = 0.0    # straight down
    d_star: float = 20.0

    def clamp(self) -> None:
        if self.d_star < LIMB_MIN:
            self.d_star = LIMB_MIN
        if self.d_star > LIMB_MAX:
            self.d_star = LIMB_MAX


@dataclass
class TailActuator:
    theta_star: float = 0.0   # radians relative to torso


@dataclass
class Skeleton:
    space: pymunk.Space
    torso: pymunk.Body
    foot_l: pymunk.Body
    foot_r: pymunk.Body
    tail: pymunk.Body
    spring_l: pymunk.DampedSpring
    spring_r: pymunk.DampedSpring
    tail_spring: pymunk.DampedRotarySpring = None
    ear_l: pymunk.Body = None
    ear_r: pymunk.Body = None
    ear_spring_l: pymunk.DampedRotarySpring = None
    ear_spring_r: pymunk.DampedRotarySpring = None
    front_l: pymunk.Body = None
    front_r: pymunk.Body = None
    foot_shape_l: pymunk.Shape = None
    foot_shape_r: pymunk.Shape = None
    limb_l: LimbActuator = field(default_factory=LimbActuator)
    limb_r: LimbActuator = field(default_factory=LimbActuator)
    tail_act: TailActuator = field(default_factory=TailActuator)

    # initial spawn pose, used by reset
    spawn: tuple[float, float] = (0.0, 64.0)
    spawn_foot_l: tuple[float, float] = (0.0, 0.0)
    spawn_foot_r: tuple[float, float] = (0.0, 0.0)
    spawn_tail: tuple[float, float] = (0.0, 0.0)
    spawn_tail_angle: float = 0.0
    spawn_ear_l: tuple[float, float] = (0.0, 0.0)
    spawn_ear_l_angle: float = 0.0
    spawn_ear_r: tuple[float, float] = (0.0, 0.0)
    spawn_ear_r_angle: float = 0.0
    spawn_front_l: tuple[float, float] = (0.0, 0.0)
    spawn_front_l_angle: float = 0.0
    spawn_front_r: tuple[float, float] = (0.0, 0.0)
    spawn_front_r_angle: float = 0.0
    spawn_theta_l: float = 0.28
    spawn_theta_r: float = -0.8
    spawn_tail_theta: float = 0.0
    # persistent facing direction: +1 = right, -1 = left. Updated with
    # hysteresis in apply_consignes; never recomputed from the sign of the angle.
    facing: int = 1

    @property
    def limb_front(self: Skeleton) -> LimbActuator:
        return self.limb_r if self.facing == 1 else self.limb_l

    @property
    def limb_back(self: Skeleton) -> LimbActuator:
        return self.limb_l if self.facing == 1 else self.limb_r

def _add_limb(space, torso, hip_local, mass, radius, stiffness, damping):
    """Foot body + DampedSpring to torso. The spring anchor on the torso is
    updated each frame to the target foot point (hip + d* along theta*). The
    foot is a box (square) for stable ground contact; rendered as a circle."""
    half = radius
    box_size = (half * 2, half * 2)
    foot = pymunk.Body(mass, float("inf"))  # infinite moment: rotation locked
    hip_world = torso.local_to_world(hip_local)
    foot.position = hip_world + (0, -40)
    shape = pymunk.Poly.create_box(foot, size=(half * 2, half * 2))
    shape.friction = 3.0
    shape.filter = _animat_filter()
    shape.collision_type = FOOT_TYPE
    space.add(foot, shape)
    spring = pymunk.DampedSpring(
        torso, foot,
        anchor_a=tuple(hip_local), anchor_b=(0, 0),
        rest_length=40.0, stiffness=stiffness, damping=damping,
    )
    space.add(spring)
    return foot, spring, shape


def _add_tail(space, torso, attach_local, mass, length, radius, stiffness, damping):
    """Tail with a rotary spring: finite moment so it swings smoothly when the
    rest angle changes (facing flip). The spring's rest_angle is updated in
    apply_consignes. Non-collidable.

    moment_for_segment returns the moment about (0,0) (the pivot); we shift it
    to the center of mass at (0, -L/2) so pymunk's rotation dynamics are
    consistent with the declared COM and the tail responds crisply.
    """
    i_pivot = pymunk.moment_for_segment(mass, (0, 0), (0, -length), radius)
    i_com = i_pivot - mass * (length / 2) ** 2
    tail = pymunk.Body(mass, i_com)
    tail.center_of_mass = (0, -length / 2)
    tail.position = torso.local_to_world(attach_local)
    tail.angle = torso.angle
    shape = pymunk.Segment(tail, (0, 0), (0, -length), radius)
    shape.sensor = True
    shape.filter = _animat_filter()
    pivot = pymunk.PivotJoint(torso, tail, attach_local, (0, 0))
    spring = pymunk.DampedRotarySpring(
        torso, tail, rest_angle=0.0, stiffness=stiffness, damping=damping,
    )
    space.add(tail, shape, pivot, spring)
    return tail, spring


def _add_ear(space, torso, attach_local, mass, length, base, rest_angle, stiffness, damping):
    """Ear: small aesthetic triangle with a rotary spring so it wobbles
    naturally. Base at the pivot, tip pointing outward. Non-collidable."""
    verts = [(-base / 2, 0), (base / 2, 0), (0, -length)]
    ear = pymunk.Body(mass, pymunk.moment_for_poly(mass, verts))
    ear.center_of_mass = (0, -length / 2)
    ear.position = torso.local_to_world(attach_local)
    ear.angle = torso.angle + rest_angle
    shape = pymunk.Poly(ear, verts)
    shape.sensor = True
    shape.filter = _animat_filter()
    pivot = pymunk.PivotJoint(torso, ear, attach_local, (0, 0))
    spring = pymunk.DampedRotarySpring(
        torso, ear, rest_angle=rest_angle, stiffness=stiffness, damping=damping,
    )
    space.add(ear, shape, pivot, spring)
    return ear, spring


def _add_front_leg(space, torso, pivot_local, length, radius, rest_angle):
    """Front leg: circle attached at pivot via PivotJoint. The body sits at
    the pivot; the circle shape is offset by (0, -length) so its center is
    exactly length from the pivot. Instant angular servo (infinite moment,
    angle imposed in apply_consignes). Non-collidable."""
    leg = pymunk.Body(FRONT_MASS, float("inf"))
    leg.position = torso.local_to_world(pivot_local)
    leg.angle = torso.angle + rest_angle
    shape = pymunk.Circle(leg, radius, (0, -length))
    shape.sensor = True
    shape.filter = _animat_filter()
    pivot = pymunk.PivotJoint(torso, leg, pivot_local, (0, 0))
    space.add(leg, shape, pivot)
    return leg


def build_skeleton(space: pymunk.Space) -> Skeleton:
    torso = pymunk.Body(TORSO_MASS, pymunk.moment_for_poly(TORSO_MASS, TORSO_VERTS))
    # moment_for_poly returns moment about the polygon's centroid, not (0,0).
    # Tell pymunk where the COM actually is so rotation is computed correctly.
    cx = sum(v[0] for v in TORSO_VERTS) / len(TORSO_VERTS)
    cy = sum(v[1] for v in TORSO_VERTS) / len(TORSO_VERTS)
    torso.center_of_mass = (cx, cy)
    torso.position = (0.0, 64.0)
    torso_shape = pymunk.Poly(torso, TORSO_VERTS)
    torso_shape.friction = 0.8
    torso_shape.filter = _animat_filter()
    torso_shape.collision_type = TORSO_TYPE
    space.add(torso, torso_shape)

    foot_l, spring_l, shape_l = _add_limb(space, torso, HIP_L, LIMB_MASS, LIMB_RADIUS,
                                 LIMB_STIFFNESS, LIMB_DAMPING)
    foot_r, spring_r, shape_r = _add_limb(space, torso, HIP_R, LIMB_MASS, LIMB_RADIUS,
                                LIMB_STIFFNESS, LIMB_DAMPING)
    tail, tail_spring = _add_tail(space, torso, TAIL_ATTACH, TAIL_MASS,
                     TAIL_LEN, TAIL_RADIUS, TAIL_STIFFNESS, TAIL_DAMPING)
    ear_l, es_l = _add_ear(space, torso, EAR_ATTACH_L, EAR_MASS, EAR_LEN, EAR_BASE,
                     EAR_REST_L, EAR_STIFFNESS, EAR_DAMPING)
    ear_r, es_r = _add_ear(space, torso, EAR_ATTACH_R, EAR_MASS, EAR_LEN, EAR_BASE,
                     EAR_REST_R, EAR_STIFFNESS, EAR_DAMPING)
    front_l = _add_front_leg(space, torso, FRONT_PIVOT, FRONT_LEN, FRONT_RADIUS,
                             -FRONT_REST)
    front_r = _add_front_leg(space, torso, FRONT_PIVOT, FRONT_LEN, FRONT_RADIUS,
                             FRONT_REST)

    skel = Skeleton(
        space=space, torso=torso, foot_l=foot_l, foot_r=foot_r, tail=tail,
        tail_spring=tail_spring,
        spring_l=spring_l, spring_r=spring_r,
        ear_l=ear_l, ear_r=ear_r,
        ear_spring_l=es_l, ear_spring_r=es_r,
        front_l=front_l, front_r=front_r,
        foot_shape_l=shape_l, foot_shape_r=shape_r,
    )
    # splay feet outward for a wider stance: left foot leans left, right foot
    # leans right. This widens the support base from ~14px to ~27px.
    skel.limb_l.theta_star = 0.28
    skel.limb_r.theta_star = -0.8
    # save spawn positions for reset
    skel.spawn_foot_l = (foot_l.position.x, foot_l.position.y)
    skel.spawn_foot_r = (foot_r.position.x, foot_r.position.y)
    skel.spawn_tail = (tail.position.x, tail.position.y)
    skel.spawn_tail_angle = tail.angle
    skel.spawn_ear_l = (ear_l.position.x, ear_l.position.y)
    skel.spawn_ear_l_angle = ear_l.angle
    skel.spawn_ear_r = (ear_r.position.x, ear_r.position.y)
    skel.spawn_ear_r_angle = ear_r.angle
    skel.spawn_front_l = (front_l.position.x, front_l.position.y)
    skel.spawn_front_l_angle = front_l.angle
    skel.spawn_front_r = (front_r.position.x, front_r.position.y)
    skel.spawn_front_r_angle = front_r.angle
    skel.spawn_theta_l = skel.limb_l.theta_star
    skel.spawn_theta_r = skel.limb_r.theta_star
    skel.spawn_tail_theta = skel.tail_act.theta_star
    return skel


def apply_consignes(skel: Skeleton) -> None:
    """Push the current consignes into the physics.

    For each hind limb we compute the target foot point in the torso frame and
    move the torso-side spring anchor there, with rest_length 0 so the spring
    pulls the foot to exactly that point. For the tail we impose the angle
    directly (servo): ±90° from the torso depending on facing direction, plus
    theta_star as offset. Infinite moment means no torque perturbs it.
    """
    skel.limb_l.clamp()
    skel.limb_r.clamp()
    # lock foot rotation: infinite moment + force angle to 0 each frame
    skel.foot_l.angle = 0.0
    skel.foot_r.angle = 0.0
    skel.foot_l.angular_velocity = 0.0
    skel.foot_r.angular_velocity = 0.0
    for act, shape in (
        (skel.limb_l, skel.foot_shape_l),
        (skel.limb_r, skel.foot_shape_r),
    ):
        # friction proportional to leg extension: a fully extended leg (d_star
        # near LIMB_MAX) grips hard, a retracted leg (d_star near LIMB_MIN)
        # slips. This makes forward locomotion possible.
        t = (act.d_star - LIMB_MIN) / (LIMB_MAX - LIMB_MIN)
        shape.friction = 0.2 + t * 16.0
    for hip_local, act, spring in (
        (HIP_L, skel.limb_l, skel.spring_l),
        (HIP_R, skel.limb_r, skel.spring_r),
    ):
        tx = hip_local[0] + act.d_star * -math.sin(act.theta_star)
        ty = hip_local[1] + act.d_star * -math.cos(act.theta_star)
        spring.anchor_a = (tx, ty)
        spring.rest_length = 0.0
    # facing direction: thermostat hysteresis on ±15°. Only flip when the torso
    # tilt crosses the deadzone in the opposite direction; within ±15° the
    # previous facing persists, preventing chatter near upright.
    if skel.facing == 1 and skel.torso.angle > FACING_DEADZONE:
        skel.facing = -1
    elif skel.facing == -1 and skel.torso.angle < -FACING_DEADZONE:
        skel.facing = 1
    sgn = -skel.facing
    skel.tail_spring.rest_angle = -sgn * (math.pi / 2) + skel.facing * skel.tail_act.theta_star
    # front legs: instant angular servo toward facing direction
    front_base = -sgn * (math.pi / 2 - FRONT_REST)
    skel.front_l.angle = skel.torso.angle + front_base + sgn * 0.12
    skel.front_r.angle = skel.torso.angle + front_base - sgn * 0.12
    skel.front_l.angular_velocity = 0.0
    skel.front_r.angular_velocity = 0.0
    # ears: update rest angle sign with facing direction, but let the spring
    # move them smoothly (not instant) — it's cuter that way
    skel.ear_spring_l.rest_angle = -sgn * EAR_REST_L
    skel.ear_spring_r.rest_angle = -sgn * EAR_REST_R


def reset(skel: Skeleton) -> None:
    """Restore the spawn pose and zero velocities."""
    skel.torso.position = skel.spawn
    skel.torso.angle = 0.0
    skel.torso.velocity = (0, 0)
    skel.torso.angular_velocity = 0.0
    skel.facing = 1
    for foot in (skel.foot_l, skel.foot_r):
        foot.velocity = (0, 0)
        foot.angular_velocity = 0.0
    skel.foot_l.position = skel.spawn_foot_l
    skel.foot_r.position = skel.spawn_foot_r
    skel.tail.position = skel.spawn_tail
    skel.tail.angle = skel.spawn_tail_angle
    skel.tail.velocity = (0, 0)
    skel.tail.angular_velocity = 0.0
    for ear, pos, ang in ((skel.ear_l, skel.spawn_ear_l, skel.spawn_ear_l_angle),
                          (skel.ear_r, skel.spawn_ear_r, skel.spawn_ear_r_angle)):
        ear.position = pos
        ear.angle = ang
        ear.velocity = (0, 0)
        ear.angular_velocity = 0.0
    for front, pos, ang in ((skel.front_l, skel.spawn_front_l, skel.spawn_front_l_angle),
                            (skel.front_r, skel.spawn_front_r, skel.spawn_front_r_angle)):
        front.position = pos
        front.angle = ang
        front.velocity = (0, 0)
        front.angular_velocity = 0.0
    skel.limb_l.theta_star = skel.spawn_theta_l
    skel.limb_r.theta_star = skel.spawn_theta_r
    skel.tail_act.theta_star = skel.spawn_tail_theta
    apply_consignes(skel)
