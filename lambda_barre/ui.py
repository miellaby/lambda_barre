"""Mouse IHM for λ̄: one joystick per hind limb (angle + distance) and a
horizontal slider for the tail angle.

The joystick is a circle with a free handle: the handle's position encodes the
limb consigne directly -- its direction from the centre is theta_star (foot
pointing direction in the torso frame) and its distance from the centre is
d_star (foot distance from the hip). Push the handle out to extend the leg,
swing it around to steer the foot.

The tail slider maps its horizontal travel to theta_star in [-TAIL_RANGE,
+TAIL_RANGE].

These widgets are the single source of truth for the consignes once built:
main reads their state and pushes it into the skeleton each frame.
"""
from __future__ import annotations

import math

import pygame

from . import body as B
from . import render as R

# --- layout ------------------------------------------------------------------
JOY_RADIUS = 64
JOY_L_CENTRE = (96, 300)
JOY_R_CENTRE = (R.WIDTH - 96, 300)

# tail slider: horizontal track below the ground line
TAIL_RANGE = math.pi / 2          # +/- 90° from the facing-default tail angle
TAIL_TRACK_Y = 572
TAIL_TRACK_X1 = 360
TAIL_TRACK_X2 = 600
TAIL_HANDLE_R = 10

# --- colours -----------------------------------------------------------------
JOY_RING_C = (70, 76, 92)
JOY_FILL_C = (40, 44, 58)
JOY_HANDLE_C = (236, 196, 110)
JOY_HANDLE_RING_C = (255, 255, 255)
JOY_CROSS_C = (60, 66, 82)
JOY_LABEL_C = (200, 206, 220)

SLIDER_TRACK_C = (70, 76, 92)
SLIDER_FILL_C = (40, 44, 58)
SLIDER_HANDLE_C = (200, 158, 84)
SLIDER_HANDLE_RING_C = (255, 255, 255)
SLIDER_LABEL_C = (200, 206, 220)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class Joystick:
    """A circular joystick whose handle encodes (theta_star, d_star).

    Screen coords are y-down. The handle offset (dx, dy) from the centre maps
    to the limb consigne via the same convention used in body.apply_consignes:
        tx = hip_x + d * -sin(theta)
        ty = hip_y + d * -cos(theta)
    so handle-down (dy>0) is theta=0 (foot straight down) and handle-left
    (dx<0) is theta=+pi/2 (foot to the -x side).
    """

    def __init__(self, centre: tuple[int, int], radius: int,
                 theta: float, d: float, label: str):
        self.cx, self.cy = centre
        self.radius = radius
        self.theta = theta
        self.d = d
        self.label = label
        self.dragging = False

    # --- consigne <-> handle geometry ----------------------------------------
    def _handle_dist(self) -> float:
        return self.d * self.radius

    def handle_pos(self) -> tuple[int, int]:
        r = self._handle_dist()
        dx = -r * math.sin(self.theta)
        dy = r * math.cos(self.theta)
        return int(self.cx + dx), int(self.cy + dy)

    def _set_from_point(self, mx: int, my: int) -> None:
        dx = mx - self.cx
        dy = my - self.cy
        r = math.hypot(dx, dy)
        if r > self.radius:
            dx *= self.radius / r
            dy *= self.radius / r
            r = self.radius
        if r > 0.5:
            self.theta = math.atan2(-dx, dy)
        # distance: centre = 0, rim = 1
        self.d = _clamp(r / self.radius, 0.0, 1.0)

    # --- hit testing / interaction -------------------------------------------
    def hit(self, mx: int, my: int) -> bool:
        return math.hypot(mx - self.cx, my - self.cy) <= self.radius

    def on_down(self, mx: int, my: int) -> bool:
        if self.hit(mx, my):
            self.dragging = True
            self._set_from_point(mx, my)
            return True
        return False

    def on_motion(self, mx: int, my: int) -> None:
        if self.dragging:
            self._set_from_point(mx, my)

    def on_up(self) -> None:
        self.dragging = False

    # --- drawing -------------------------------------------------------------
    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        pygame.draw.circle(screen, JOY_FILL_C, (self.cx, self.cy), self.radius)
        pygame.draw.circle(screen, JOY_RING_C, (self.cx, self.cy), self.radius, 2)
        # crosshair
        pygame.draw.line(screen, JOY_CROSS_C,
                         (self.cx - self.radius, self.cy),
                         (self.cx + self.radius, self.cy), 1)
        pygame.draw.line(screen, JOY_CROSS_C,
                         (self.cx, self.cy - self.radius),
                         (self.cx, self.cy + self.radius), 1)
        # handle
        hx, hy = self.handle_pos()
        pygame.draw.circle(screen, JOY_HANDLE_C, (hx, hy), 12)
        pygame.draw.circle(screen, JOY_HANDLE_RING_C, (hx, hy), 12, 2)
        # label
        surf = font.render(self.label, True, JOY_LABEL_C)
        screen.blit(surf, (self.cx - surf.get_width() // 2,
                           self.cy - self.radius - 22))
        # readout
        deg = math.degrees(self.theta)
        info = f"{deg:+5.0f}°  d{self.d:4.1f}"
        s2 = font.render(info, True, JOY_LABEL_C)
        screen.blit(s2, (self.cx - s2.get_width() // 2,
                         self.cy + self.radius + 6))


class Slider:
    """Horizontal slider mapping x-position to theta_star in [-range, +range]."""

    def __init__(self, x1: int, x2: int, y: int, theta: float,
                 rng: float, label: str):
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.theta = theta
        self.rng = rng
        self.label = label
        self.dragging = False

    def _handle_x(self) -> int:
        t = (self.theta + self.rng) / (2 * self.rng)
        return int(self.x1 + _clamp(t, 0.0, 1.0) * (self.x2 - self.x1))

    def _set_from_x(self, mx: int) -> None:
        t = (mx - self.x1) / (self.x2 - self.x1)
        t = _clamp(t, 0.0, 1.0)
        self.theta = -self.rng + t * 2 * self.rng

    def hit(self, mx: int, my: int) -> bool:
        hx = self._handle_x()
        # grab the handle or anywhere on the track
        on_handle = math.hypot(mx - hx, my - self.y) <= TAIL_HANDLE_R + 4
        on_track = (self.x1 - 4 <= mx <= self.x2 + 4
                    and abs(my - self.y) <= TAIL_HANDLE_R + 6)
        return on_handle or on_track

    def on_down(self, mx: int, my: int) -> bool:
        if self.hit(mx, my):
            self.dragging = True
            self._set_from_x(mx)
            return True
        return False

    def on_motion(self, mx: int, my: int) -> None:
        if self.dragging:
            self._set_from_x(mx)

    def on_up(self) -> None:
        self.dragging = False

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        # track
        track_h = 6
        rect = pygame.Rect(self.x1, self.y - track_h // 2,
                           self.x2 - self.x1, track_h)
        pygame.draw.rect(screen, SLIDER_FILL_C, rect, border_radius=3)
        pygame.draw.rect(screen, SLIDER_TRACK_C, rect, 1, border_radius=3)
        # centre tick
        cx = (self.x1 + self.x2) // 2
        pygame.draw.line(screen, SLIDER_TRACK_C,
                         (cx, self.y - 8), (cx, self.y + 8), 1)
        # handle
        hx = self._handle_x()
        pygame.draw.circle(screen, SLIDER_HANDLE_C, (hx, self.y), TAIL_HANDLE_R)
        pygame.draw.circle(screen, SLIDER_HANDLE_RING_C,
                           (hx, self.y), TAIL_HANDLE_R, 2)
        # label + readout
        lbl = font.render(self.label, True, SLIDER_LABEL_C)
        screen.blit(lbl, (self.x1, self.y - 28))
        deg = math.degrees(self.theta)
        info = f"{deg:+5.0f}°"
        s2 = font.render(info, True, SLIDER_LABEL_C)
        screen.blit(s2, (self.x2 - s2.get_width(), self.y - 28))


class Controls:
    """Owns the joysticks and the tail slider; routes mouse events."""

    def __init__(self, skel: "B.Skeleton"):
        self.joy_l = Joystick(JOY_L_CENTRE, JOY_RADIUS,
            skel.limb_l.theta_star,
            (skel.limb_l.d_star - B.LIMB_MIN) / (B.LIMB_MAX - B.LIMB_MIN), "L")
        self.joy_r = Joystick(JOY_R_CENTRE, JOY_RADIUS,
            skel.limb_r.theta_star,
            (skel.limb_r.d_star - B.LIMB_MIN) / (B.LIMB_MAX - B.LIMB_MIN), "R")
        self.tail = Slider(TAIL_TRACK_X1, TAIL_TRACK_X2, TAIL_TRACK_Y,
            skel.tail_act.theta_star, TAIL_RANGE, "tail")

    @property
    def _widgets(self):
        return (self.joy_l, self.joy_r, self.tail)

    def on_down(self, mx: int, my: int) -> None:
        for w in self._widgets:
            if w.on_down(mx, my):
                return

    def on_motion(self, mx: int, my: int) -> None:
        for w in self._widgets:
            w.on_motion(mx, my)

    def on_up(self) -> None:
        for w in self._widgets:
            w.on_up()

    def drive(self, skel: "B.Skeleton") -> None:
        """Copy normalized widget state into the skeleton consignes."""
        assert 0.0 <= self.joy_l.d <= 1.0, (
            f"Invalid joy_l.d={self.joy_l.d}, expected [0, 1]"
        )
        assert 0.0 <= self.joy_r.d <= 1.0, (
            f"Invalid joy_r.d={self.joy_r.d}, expected [0, 1]"
        )
        skel.limb_l.theta_star = self.joy_l.theta
        skel.limb_l.d_star = (
            B.LIMB_MIN
            + self.joy_l.d * (B.LIMB_MAX - B.LIMB_MIN)
        )

        skel.limb_r.theta_star = self.joy_r.theta
        skel.limb_r.d_star = (
            B.LIMB_MIN
            + self.joy_r.d * (B.LIMB_MAX - B.LIMB_MIN)
        )

        skel.tail_act.theta_star = self.tail.theta

    def sync(self, skel: "B.Skeleton") -> None:
        """Copy skeleton consignes into the widgets (for display in brain mode)."""
        self.joy_l.theta = skel.limb_l.theta_star
        self.joy_l.d = (
            skel.limb_l.d_star - B.LIMB_MIN
        ) / (B.LIMB_MAX - B.LIMB_MIN)

        self.joy_r.theta = skel.limb_r.theta_star
        self.joy_r.d = (
            skel.limb_r.d_star - B.LIMB_MIN
        ) / (B.LIMB_MAX - B.LIMB_MIN)

        self.tail.theta = skel.tail_act.theta_star

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        self.joy_l.draw(screen, font)
        self.joy_r.draw(screen, font)
        self.tail.draw(screen, font)
