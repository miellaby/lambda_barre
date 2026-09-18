"""Exteroception: perception of the external environment.

See exteroception.md for the full spec. This module implements the cursor
modality (position, velocity, click/keyboard sound), the vision modality (4×4
retina cone + optical flow), and the touch modality (contact forces).

The facing sign is an internal mirror variable — never perceived. "In front
of the head" = always positive, regardless of which way the animat faces.

Usage:
    cursor = extero.Cursor(skel)
    cursor.on_click()                         # on MOUSEBUTTONDOWN (main button)
    cursor.on_key(ev.scancode)               # on KEYDOWN, pass ev.scancode
    cursor_signals = cursor.update(skel, mouse_screen, frame_dt)

    vision = extero.Vision(skel, space)
    vision_signals = vision.update(skel, frame_dt)

    touch = extero.Touch(space, skel)
    touch.reset_contacts()                     # before the physics substeps
    touch_signals = touch.update(skel, frame_dt)
"""
from __future__ import annotations

import math
from collections import deque

import pymunk

from . import body as B
from . import render as R


class Cursor:
    """Mouse cursor position, velocity, and sound in the head's egocentric
    frame.

    Cadenced at 30 Hz: the main loop runs at 60 Hz, but the cursor is sampled
    every other frame. Between samples, the last values are returned.

    The mouse position from pygame is in screen coordinates (y-down). It is
    converted to world coordinates (y-up) before computing egocentric values.

    Sound: 5 frequency cells, each driven by an IIR low-pass filter. On a
    click or keypress, the cells matching the event's frequency mask are
    incremented (cps += 1). At each sample, cps decays exponentially
    (half-life = 1 s). The perceived intensity is ``cps * curseur_prox``,
    normalised to [0, 1]. The sound source is the cursor position — both
    clicks and keyboard keys emit from the cursor.

    Mouse click activates cell 0 only (bit 0 = 1).
    Keyboard keys activate cells based on the key index encoded in binary
    on 5 bits (1-31). Space = bit 0, Return = all bits (31).

    Signals produced:
        curseur_dir: facing * atan2(dy, dx) — angle égocentré (devant = 0)
        curseur_prox: max_r / (max_r + r) — proximité (proche = 1, loin = ~0)
        curseur_vx: facing * mouse_vx — horizontal velocity, egocentered
        curseur_vy: mouse_vy — vertical velocity
        son_0 .. son_4: intensité par cellule fréquentielle (0..1)
    """

    SAMPLE_HZ = 30.0
    CURSOR_MAX_R = 600.0      # px — max distance for proximity normalization
    N_FREQ = 5                # number of frequency cells
    CPS_HALF_LIFE = 1.0       # seconds — IIR decay half-life
    CPS_MAX = 10.0            # normalisation cap for cps

    # Key index: boustrophedon path over the AZERTY keyboard (row 0 L→R,
    # row 1 R→L, row 2 L→R, row 3 R→L) with Gray code indices. Two neighbouring
    # keys differ by exactly 1 bit. 0 = silence (no key).
    # Uses SDL scancodes (physical key positions, layout-independent).
    # 29 keys: 26 letters (AZERTY positions) + comma + space + return.
    _SCANCODE_INDEX: dict = None

    @classmethod
    def _build_key_index(cls) -> dict:
        if cls._SCANCODE_INDEX is not None:
            return cls._SCANCODE_INDEX
        # Boustrophedon order over QWERTY physical layout (scancodes):
        # Row 0 L→R: Q W E R T Y U I O P       (20,26,8,21,23,28,24,12,18,19)
        # Row 1 R→L: ;(=M AZERTY) L K J H G F D S A  (51,15,14,13,11,10,9,7,22,4)
        # Row 2 L→R: Z X C V B N M ,            (29,27,6,25,5,17,16,54)
        # Row 3 R→L: return ... space           (40,57)
        # 30 keys, Gray code: index i → Gray(i) = i ^ (i >> 1), starting at 1
        cls._SCANCODE_INDEX = {
            20: 1,   26: 3,    8: 2,    21: 6,    23: 7,
            28: 5,   24: 4,    12: 12,  18: 13,   19: 15,
            51: 14,  15: 10,   14: 11,  13: 9,    11: 8,
            10: 24,  9: 25,    7: 27,   22: 26,   4: 30,
            29: 31,  27: 29,   6: 28,   25: 20,   5: 21,
            17: 23,  16: 22,   54: 18,  40: 19,   57: 17,
        }
        return cls._SCANCODE_INDEX

    def __init__(self, skel: "B.Skeleton"):
        self._skel = skel
        self._accum = 0.0
        self._t = 0.0
        self._sample_dt = 1.0 / self.SAMPLE_HZ
        self._prev_mouse_world: tuple[float, float] | None = None
        # IIR cps per frequency cell
        self._cps = [0.0] * self.N_FREQ
        # decay factor per sample (half-life = 1s)
        self._decay = 0.5 ** (self._sample_dt / self.CPS_HALF_LIFE)
        self._signals = {
            "curseur_dir": 0.0,
            "curseur_prox": 0.0,
            "curseur_vx": 0.0,
            "curseur_vy": 0.0,
        }
        for i in range(self.N_FREQ):
            self._signals[f"son_{i}"] = 0.0

    def on_click(self) -> None:
        """Register a main-button click. Activates cell 0 only."""
        self._cps[0] += 1.0

    def on_key(self, scancode: int) -> None:
        """Register a keyboard keypress. Activates the cells matching the
        key's 5-bit frequency mask. Call from the event loop on KEYDOWN,
        passing ev.scancode (physical key position, layout-independent)."""
        idx = self._build_key_index()
        mask = idx.get(scancode, 0)
        if mask == 0:
            return
        for bit in range(self.N_FREQ):
            if mask & (1 << bit):
                self._cps[bit] += 1.0

    def reset(self) -> None:
        """Call after B.reset(skel) to clear velocity and sound history."""
        self._accum = 0.0
        self._t = 0.0
        self._prev_mouse_world = None
        self._cps = [0.0] * self.N_FREQ
        self._signals = {k: 0.0 for k in self._signals}

    def update(self, skel: "B.Skeleton",
               mouse_screen: tuple[float, float],
               frame_dt: float) -> dict:
        """Advance the sensor by frame_dt. Recomputes at 30 Hz; returns the
        last computed signals otherwise."""
        self._accum += frame_dt
        self._t += frame_dt
        if self._accum < self._sample_dt:
            return self._signals

        self._accum -= self._sample_dt
        self._compute(skel, mouse_screen)
        return self._signals

    def _compute(self, skel: "B.Skeleton",
                 mouse_screen: tuple[float, float]) -> None:
        facing = skel.facing
        head = B.head_world(skel)

        mouse_world = R.s2w(*mouse_screen)

        dx = facing * (mouse_world[0] - head.x)
        dy = mouse_world[1] - head.y
        r = math.hypot(dx, dy)
        direction = math.atan2(dy, dx)
        prox = self.CURSOR_MAX_R / (self.CURSOR_MAX_R + r) if r > 0 else 1.0

        if self._prev_mouse_world is not None:
            vx = (mouse_world[0] - self._prev_mouse_world[0]) / self._sample_dt
            vy = (mouse_world[1] - self._prev_mouse_world[1]) / self._sample_dt
        else:
            vx = vy = 0.0
        self._prev_mouse_world = mouse_world

        # IIR decay + intensity = cps * prox, normalised
        self._signals = {
            "curseur_dir": direction,
            "curseur_prox": prox,
            "curseur_vx": facing * vx,
            "curseur_vy": vy,
        }
        for i in range(self.N_FREQ):
            self._cps[i] *= self._decay
            intensity = min(1.0, (self._cps[i] / self.CPS_MAX) * prox)
            self._signals[f"son_{i}"] = intensity


# --- Vision: 4×4 retina cone -------------------------------------------------

# Cone geometry
VISION_FOV = math.radians(100)    # total field of view (vertical spread)
VISION_RANGE = 40.0              # px, max depth — about the head length
VISION_ROWS = 4
VISION_COLS = 4
VISION_GRID = 3                  # sample grid per cell (GRID × GRID points)
VISION_HZ = 6.0
VISION_QUERY_RADIUS = 3.0        # max_distance for point_query_nearest
VISION_APEX_OFFSET = 20.0         # px forward from head — cone clears the body

# Shape → grayscale brightness mapping ([0, 1])
BG_GRAY = 0.05                 # background: near-black
GROUND_GRAY = 0.35             # ground: dark
PLATFORM_GRAY = 0.55           # platform: mid-gray
DYNAMIC_GRAY = 0.90            # dynamic object: bright


def _shape_to_gray(shape: pymunk.Shape) -> float:
    """Map a pymunk shape to a grayscale brightness the animat would 'see'."""
    body = shape.body
    if body.body_type == pymunk.Body.STATIC:
        a = body.local_to_world(shape.a)
        if abs(a.y) < 1.0:
            return GROUND_GRAY
        return PLATFORM_GRAY
    return DYNAMIC_GRAY


class Vision:
    """4×4 retina cone: 16 cells, each sampled independently (no occlusion).

    The cone apex is at the head, pointing in the facing direction. It is
    sliced into 4 depth bands (rows) × 4 angular sectors (cols). Each cell is
    a quadrilateral; we sample a VISION_GRID × VISION_GRID grid of points
    inside it via ``space.point_query_nearest`` and compute:

        - mean brightness (grayscale, 1 value in [0, 1])

    Cadenced at 6 Hz: the main loop runs at 60 Hz, so the cone is resampled
    every 10 frames. Between samples, the last values are returned.

    Optical flow (3 signals) is derived from two consecutive vision frames:
    the global movement of static solids (platforms) is tracked, not the cone
    cells. For each platform that moved between frames, we compute:
        - flux_surface: sum of their surface area (no mirror)
        - flux_x: facing * dx of the movement barycentre, egocentered
        - flux_y: dy of the movement barycentre, egocentered

    Signals produced (16 + 3 values, flat dict):
        vis_c{i}  for i in 1..16  — mean brightness [0, 1]
        flux_surface, flux_x, flux_y
    Cell order is row-major: row 0 (nearest) cols 0-3, then row 1, etc.
    Col 0 = lowest angle (downward-most), col 3 = highest (upward-most).
    """

    SAMPLE_HZ = VISION_HZ

    def __init__(self, skel: "B.Skeleton", space: pymunk.Space):
        self._space = space
        self._filter = pymunk.ShapeFilter(group=B.ANIMAT_GROUP)
        self._accum = 0.0
        self._sample_dt = 1.0 / self.SAMPLE_HZ
        self._signals: dict[str, float] = {}
        for i in range(1, 17):
            self._signals[f"vis_c{i}"] = 0.0
        self._signals["flux_surface"] = 0.0
        self._signals["flux_x"] = 0.0
        self._signals["flux_y"] = 0.0
        # raw per-cell brightness for debug rendering
        self.cells: list[float] = [BG_GRAY] * 16
        # previous platform rects: id(body) -> (x0, y0, x1, y1)
        self._prev_plat_rect: dict[int, tuple[float, float, float, float]] = {}

    def reset(self) -> None:
        self._accum = 0.0
        self.cells = [BG_GRAY] * 16
        self._prev_plat_rect = {}
        self._signals = {k: 0.0 for k in self._signals}

    def update(self, skel: "B.Skeleton", frame_dt: float) -> dict:
        self._accum += frame_dt
        if self._accum < self._sample_dt:
            return self._signals
        self._accum -= self._sample_dt
        self._compute(skel)
        return self._signals

    def _compute(self, skel: "B.Skeleton") -> None:
        head = B.head_world(skel)
        # forward direction in world: +1 = right (angle 0), -1 = left (angle π)
        forward_angle = 0.0 if skel.facing > 0 else math.pi
        # apex pushed forward past the head so the cone clears the body
        apex_x = head.x + VISION_APEX_OFFSET * math.cos(forward_angle)
        apex_y = head.y + VISION_APEX_OFFSET * math.sin(forward_angle)
        half_fov = VISION_FOV / 2
        ang_step = VISION_FOV / VISION_COLS
        depth_step = VISION_RANGE / VISION_ROWS

        for row in range(VISION_ROWS):
            inner = row * depth_step
            outer = (row + 1) * depth_step
            for col in range(VISION_COLS):
                # angular sector boundaries (col 0 = down, col 3 = up)
                a_lo = forward_angle - half_fov + col * ang_step
                a_hi = forward_angle - half_fov + (col + 1) * ang_step

                # 2D sample grid: GRID depths × GRID angles within the cell
                grays: list[float] = []
                for ga in range(VISION_GRID):
                    ta = (ga + 0.5) / VISION_GRID
                    a = a_lo + ta * (a_hi - a_lo)
                    ca = math.cos(a)
                    sa = math.sin(a)
                    for gr in range(VISION_GRID):
                        tr = (gr + 0.5) / VISION_GRID
                        r = inner + tr * (outer - inner)
                        px = apex_x + r * ca
                        py = apex_y + r * sa
                        info = self._space.point_query_nearest(
                            (px, py), VISION_QUERY_RADIUS, self._filter)
                        if info is not None:
                            grays.append(_shape_to_gray(info.shape))
                        else:
                            grays.append(BG_GRAY)

                # mean brightness
                n = len(grays)
                mv = sum(grays) / n

                idx = row * VISION_COLS + col
                self._signals[f"vis_c{idx + 1}"] = mv
                self.cells[idx] = mv

        # --- optical flow: intersection of platform rectangles t-1 vs t ---
        # surface that changed = symmetric difference of old and new rects.
        # barycentre of the changed area, egocentered.
        platforms = getattr(self._space, "_platforms", [])
        cur_rect: dict[int, tuple[float, float, float, float]] = {}
        for plat, seg, w, h in platforms:
            cx, cy = plat.position.x, plat.position.y
            cur_rect[id(plat)] = (cx - w, cy - h, cx + w, cy + h)

        if self._prev_plat_rect:
            facing = skel.facing
            total_changed = 0.0
            wbx = 0.0   # weighted barycentre x
            wby = 0.0   # weighted barycentre y
            for pid, (nx0, ny0, nx1, ny1) in cur_rect.items():
                if pid not in self._prev_plat_rect:
                    continue
                ox0, oy0, ox1, oy1 = self._prev_plat_rect[pid]
                # intersection
                ix0 = max(ox0, nx0)
                iy0 = max(oy0, ny0)
                ix1 = min(ox1, nx1)
                iy1 = min(oy1, ny1)
                inter = 0.0
                if ix1 > ix0 and iy1 > iy0:
                    inter = (ix1 - ix0) * (iy1 - iy0)
                old_area = (ox1 - ox0) * (oy1 - oy0)
                new_area = (nx1 - nx0) * (ny1 - ny0)
                changed = old_area + new_area - 2 * inter
                if changed < 1e-6:
                    continue
                # barycentre of the symmetric difference:
                # (old*old_c + new*new_c - 2*inter*inter_c) / changed
                ocx = (ox0 + ox1) / 2
                ocy = (oy0 + oy1) / 2
                ncx = (nx0 + nx1) / 2
                ncy = (ny0 + ny1) / 2
                icx = (ix0 + ix1) / 2 if inter > 0 else 0.0
                icy = (iy0 + iy1) / 2 if inter > 0 else 0.0
                bx = (old_area * ocx + new_area * ncx
                      - 2 * inter * icx) / changed
                by = (old_area * ocy + new_area * ncy
                      - 2 * inter * icy) / changed
                total_changed += changed
                wbx += changed * bx
                wby += changed * by
            if total_changed > 0:
                bx = wbx / total_changed
                by = wby / total_changed
                self._signals["flux_surface"] = total_changed
                self._signals["flux_x"] = facing * (bx - head.x)
                self._signals["flux_y"] = by - head.y
            else:
                self._signals["flux_surface"] = 0.0
                self._signals["flux_x"] = 0.0
                self._signals["flux_y"] = 0.0
        self._prev_plat_rect = cur_rect


# --- Touch: contact forces on feet and torso ---------------------------------

class Touch:
    """Contact forces on the feet and torso, egocentered.

    Two ``CollisionHandler`` ``post_solve`` callbacks accumulate impulses:

    - **Feet** (FOOT_TYPE vs GROUND_TYPE): normal impulse (y component only),
      accumulated per foot, swapped by facing (front/back). Produces 2 scalars.
    - **Trunk** (TORSO_TYPE vs GROUND_TYPE): full impulse vector (x, y) plus
      the contact point barycentre, accumulated over the frame. Produces 4
      scalars (force x/y + contact point cx/cy).

    Cadenced at 60 Hz (same as the physics frame).

    Signals produced:
        contact_sol_avant: force normale du sol sur la patte avant (scalaire)
        contact_sol_arriere: idem patte arriere
        collision_tronc_x: facing * impulse_x / dt (avant = +)
        collision_tronc_y: impulse_y / dt (vers le haut = +)
        collision_tronc_cx: facing * (contact_x - head_x) (avant = +)
        collision_tronc_cy: contact_y - head_y (au-dessus = +)
    """

    def __init__(self, space: pymunk.Space, skel: "B.Skeleton"):
        self._skel = skel
        self._space = space
        # foot impulses (y component only)
        self._impulse: dict = {skel.foot_l: 0.0, skel.foot_r: 0.0}
        # trunk collision accumulator
        self._trunk_impulse_x = 0.0
        self._trunk_impulse_y = 0.0
        self._trunk_contact_x = 0.0
        self._trunk_contact_y = 0.0
        self._trunk_weight = 0.0   # total impulse magnitude for weighting
        self._signals = {
            "contact_sol_avant": 0.0,
            "contact_sol_arriere": 0.0,
            "collision_tronc_x": 0.0,
            "collision_tronc_y": 0.0,
            "collision_tronc_cx": 0.0,
            "collision_tronc_cy": 0.0,
        }

        space.on_collision(B.FOOT_TYPE, B.GROUND_TYPE,
                           post_solve=self._on_foot_post_solve)
        space.on_collision(B.TORSO_TYPE, B.GROUND_TYPE,
                           post_solve=self._on_trunk_post_solve)

    def _on_foot_post_solve(self, arb: pymunk.Arbiter, space, data):
        ny = arb.total_impulse.y
        for shape in arb.shapes:
            body = shape.body
            if body in self._impulse:
                self._impulse[body] += ny

    def _on_trunk_post_solve(self, arb: pymunk.Arbiter, space, data):
        ti = arb.total_impulse
        self._trunk_impulse_x += ti.x
        self._trunk_impulse_y += ti.y
        # contact point barycentre, weighted by impulse magnitude
        w = abs(ti.x) + abs(ti.y)
        if w > 0:
            cps = arb.contact_point_set
            for cp in cps.points:
                # point_a is on the trunk, point_b on the ground; use point_a
                self._trunk_contact_x += w * cp.point_a.x
                self._trunk_contact_y += w * cp.point_a.y
                break  # use first contact point
            self._trunk_weight += w

    def reset_contacts(self) -> None:
        self._impulse[self._skel.foot_l] = 0.0
        self._impulse[self._skel.foot_r] = 0.0
        self._trunk_impulse_x = 0.0
        self._trunk_impulse_y = 0.0
        self._trunk_contact_x = 0.0
        self._trunk_contact_y = 0.0
        self._trunk_weight = 0.0

    def reset(self) -> None:
        self.reset_contacts()
        self._signals = {k: 0.0 for k in self._signals}

    def update(self, skel: "B.Skeleton", dt: float) -> dict:
        facing = skel.facing
        if facing == 1:
            foot_front = skel.foot_r
            foot_back = skel.foot_l
        else:
            foot_front = skel.foot_l
            foot_back = skel.foot_r

        f_front = self._impulse[foot_front] / dt if dt > 0 else 0.0
        f_back = self._impulse[foot_back] / dt if dt > 0 else 0.0

        facing_ego = skel.facing

        tx = self._trunk_impulse_x / dt if dt > 0 else 0.0
        ty = self._trunk_impulse_y / dt if dt > 0 else 0.0
        if self._trunk_weight > 0:
            cx = self._trunk_contact_x / self._trunk_weight
            cy = self._trunk_contact_y / self._trunk_weight
            # contact point in trunk-local frame, egocentered by facing
            lc = skel.torso.world_to_local((cx, cy))
            ccx = facing_ego * lc.x
            ccy = lc.y
        else:
            ccx = ccy = 0.0

        self._signals = {
            "contact_sol_avant": f_front,
            "contact_sol_arriere": f_back,
            "collision_tronc_x": facing_ego * tx,
            "collision_tronc_y": ty,
            "collision_tronc_cx": ccx,
            "collision_tronc_cy": ccy,
        }
        return self._signals
