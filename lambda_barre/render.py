"""Pygame renderer for λ̄. Physics is y-up (ground at y=0); screen is y-down."""
from __future__ import annotations

import math

import pygame

from . import body as B

# screen geometry
WIDTH, HEIGHT = 960, 600
GROUND_SCREEN_Y = 540           # world y=0 maps here
ORIGIN_X = WIDTH // 2           # world x=0 maps here

BG = (24, 26, 34)
GROUND_C = (90, 96, 110)
TORSO_C = (236, 196, 110)
LIMB_C = (210, 168, 92)
TAIL_C = (200, 158, 84)
HEAD_C = (246, 210, 120)
EAR_C = (236, 196, 110)
HUD_C = (200, 206, 220)
TARGET_C = (110, 200, 255)

# proprio panel
PROPRIO_LABEL_C = (160, 166, 180)
PROPRIO_BAR_BG = (40, 44, 58)
PROPRIO_BAR_POS = (110, 200, 140)
PROPRIO_BAR_NEG = (200, 110, 110)


def w2s(x: float, y: float) -> tuple[int, int]:
    """World -> screen."""
    return int(ORIGIN_X + x), int(GROUND_SCREEN_Y - y)


def s2w(sx: float, sy: float) -> tuple[float, float]:
    """Screen -> world."""
    return sx - ORIGIN_X, GROUND_SCREEN_Y - sy


def draw(screen, skel: "B.Skeleton", font, show_targets: bool = True) -> None:
    screen.fill(BG)
    # ground line
    pygame.draw.line(screen, GROUND_C, w2s(-WIDTH, 0), w2s(WIDTH, 0), 4)
    # platforms (static segments, skip ground which is at y≈0)
    for shape in skel.space.shapes:
        body = getattr(shape, "body", None)
        if body is None or body.body_type != B.pymunk.Body.STATIC:
            continue
        if not isinstance(shape, B.pymunk.Segment):
            continue
        a = body.local_to_world(shape.a)
        b = body.local_to_world(shape.b)
        if abs(a.y) < 1e-6 and abs(b.y) < 1e-6:
            continue  # ground
        r = int(shape.radius)
        sa, sb = w2s(a.x, a.y), w2s(b.x, b.y)
        # capsule = rect (straight part) + 2 discs (caps), pixel-aligned
        x0, x1 = min(sa[0], sb[0]), max(sa[0], sb[0])
        cy = sa[1]
        rect = pygame.Rect(x0, cy - r, x1 - x0, r * 2)
        pygame.draw.rect(screen, (70, 74, 88), rect)
        pygame.draw.circle(screen, (70, 74, 88), (x0, cy), r)
        pygame.draw.circle(screen, (70, 74, 88), (x1, cy), r)

    torso = skel.torso
    # torso triangle — mirror x when facing left so the asymmetric apex
    # follows the orientation
    sgn = skel.facing
    verts = [torso.local_to_world((v[0] * sgn, v[1])) for v in B.TORSO_VERTS]
    pts = [w2s(v.x, v.y) for v in verts]
    pygame.draw.polygon(screen, TORSO_C, pts)
    pygame.draw.polygon(screen, (255, 255, 255), pts, 2)

    # tail: isoceles triangle in the tail body's own frame (base at pivot,
    # apex at the tip), drawn like the torso.
    tail_w = 16
    t_bl = skel.tail.local_to_world((-tail_w / 2, 0))
    t_br = skel.tail.local_to_world(( tail_w / 2, 0))
    t_tip = skel.tail.local_to_world((0, -B.TAIL_LEN))
    tpts = [w2s(t_bl.x, t_bl.y), w2s(t_br.x, t_br.y), w2s(t_tip.x, t_tip.y)]
    pygame.draw.polygon(screen, TAIL_C, tpts)
    pygame.draw.polygon(screen, (255, 255, 255), tpts, 2)

    # hind limbs: hip -> foot. Foot rendered as a half-disc (flat side down,
    # diameter = square side length). Foot angle is locked to world, so the
    # flat side is always horizontal.
    for hip_local, foot in ((B.HIP_L, skel.foot_l), (B.HIP_R, skel.foot_r)):
        hip = torso.local_to_world(hip_local)
        ft = foot.position
        # pygame.draw.line(screen, LIMB_C, w2s(hip.x, hip.y), w2s(ft.x, ft.y), 7)
        half = B.LIMB_RADIUS
        cx, cy = w2s(ft.x, ft.y)
        # half-disc: base aligned with square bottom. Square spans cy-half..cy+half
        # (world y = ft.y +/- half → screen y = cy +/- half). The flat base is
        # at the bottom of the square (screen y = cy + half). The semicircle
        # bulges upward from there.
        base_y = cy + half
        pts = []
        n = 16
        for i in range(n + 1):
            a = math.pi - i * math.pi / n   # pi -> 0 (left to right, top half)
            pts.append((cx + half * math.cos(a), base_y - half * math.sin(a)))
        pygame.draw.polygon(screen, LIMB_C, pts)
        pygame.draw.line(screen, (255, 255, 255), (cx - half, base_y), (cx + half, base_y), 2)

    # ears: triangles with rotary springs, drawn from each ear body's own frame
    for ear in (skel.ear_l, skel.ear_r):
        bl = ear.local_to_world((-B.EAR_BASE / 2, 0))
        br = ear.local_to_world(( B.EAR_BASE / 2, 0))
        tip = ear.local_to_world((0, -B.EAR_LEN))
        epts = [w2s(bl.x, bl.y), w2s(br.x, br.y), w2s(tip.x, tip.y)]
        pygame.draw.polygon(screen, EAR_C, epts)
        pygame.draw.polygon(screen, (255, 255, 255), epts, 2)

    # aesthetic head: isoceles triangle in world space (not rotated with torso).
    # -20° when facing right, 200° when facing left — horizontal, slightly down.
    sgn = skel.facing
    head_angle = math.radians(-20 if sgn > 0 else 200)
    ha = torso.local_to_world((B.HEAD_ANCHOR[0] * sgn, B.HEAD_ANCHOR[1]))
    dx, dy = math.cos(head_angle), math.sin(head_angle)   # pointing direction
    px, py = -math.sin(head_angle), math.cos(head_angle)  # perpendicular (base)
    base_w = 14
    tip_len = 22
    h_base_l = (ha.x - px * base_w / 2, ha.y - py * base_w / 2)
    h_base_r = (ha.x + px * base_w / 2, ha.y + py * base_w / 2)
    h_tip    = (ha.x + dx * tip_len,   ha.y + dy * tip_len)
    hpts = [w2s(*h_base_l), w2s(*h_base_r), w2s(*h_tip)]
    pygame.draw.polygon(screen, HEAD_C, hpts)
    pygame.draw.polygon(screen, (255, 255, 255), hpts, 2)
    # front legs: circles attached at torso center, offset by FRONT_LEN
    for front in (skel.front_l, skel.front_r):
        fc = front.local_to_world((0, -B.FRONT_LEN))
        pygame.draw.circle(screen, LIMB_C, w2s(fc.x, fc.y), int(B.FRONT_RADIUS))
        pygame.draw.circle(screen, (255, 255, 255), w2s(fc.x, fc.y), int(B.FRONT_RADIUS), 2)

    # target foot points (consignes)
    if show_targets:
        for hip_local, act in ((B.HIP_L, skel.limb_l), (B.HIP_R, skel.limb_r)):
            tx = hip_local[0] + act.d_star * -math.sin(act.theta_star)
            ty = hip_local[1] + act.d_star * -math.cos(act.theta_star)
            tp = torso.local_to_world((tx, ty))
            pygame.draw.circle(screen, TARGET_C, w2s(tp.x, tp.y), 5, 2)
        # tail target angle marker: drawn from torso attach point so it
        # follows the torso rotation visibly
        tail_target = skel.torso.angle - skel.tail_spring.rest_angle
        ta = torso.local_to_world(B.TAIL_ATTACH)
        tip = (ta.x + 30 * math.sin(tail_target), ta.y - 30 * math.cos(tail_target))
        pygame.draw.line(screen, TARGET_C, w2s(ta.x, ta.y), w2s(*tip), 1)


# --- proprioception panel ----------------------------------------------------

# (2-letter code, dict key, scale, description) — scale maps value to [-1, +1]
_PROPRIO_SPECS = [
    ("TA", "tronc_angle",              math.radians(45), "tronc angle (inclinaison, + = penche avant)"),
    ("AA", "membre_angle_avant",       math.pi,          "angle patte avant (+ = pied vers avant)"),
    ("DA", "membre_distance_avant",    32.0,             "distance patte avant (longueur de patte)"),
    ("FA", "force_actuateur_avant",    2000.0,           "force actuateur patte avant (effort musculaire)"),
    ("AR", "membre_angle_arriere",     math.pi,          "angle patte arriere (+ = pied vers avant)"),
    ("DR", "membre_distance_arriere",  32.0,             "distance patte arriere (longueur de patte)"),
    ("FR", "force_actuateur_arriere",  2000.0,           "force actuateur patte arriere (effort musculaire)"),
    ("QA", "queue_angle",              math.pi,          "angle queue (+ = vers avant, rel. neutrale)"),
    ("CQ", "couple_queue",             200000.0,         "couple queue (+ = vers avant)"),
    ("XA", "accel_tete_avant",         2000.0,           "acceleration tete avant (+ = projete en avant)"),
    ("YA", "accel_tete_haut",          2000.0,           "acceleration tete haut (+ = vers le haut)"),
]

PROPRIO_BLOCK = 40
PROPRIO_GAP = 4
PROPRIO_TOP = 70


def _lerp_color(c1, c2, t):
    return (int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t))


def proprio_block_rect(index: int) -> pygame.Rect:
    """Return the screen rect of the block at the given index."""
    n = len(_PROPRIO_SPECS)
    total_w = n * PROPRIO_BLOCK + (n - 1) * PROPRIO_GAP
    x0 = (WIDTH - total_w) // 2
    bx = x0 + index * (PROPRIO_BLOCK + PROPRIO_GAP)
    return pygame.Rect(bx, PROPRIO_TOP, PROPRIO_BLOCK, PROPRIO_BLOCK)


def draw_proprio(screen, font, signals: dict,
                 mouse_pos: tuple[int, int] | None = None) -> str | None:
    """Draw the proprioceptive signals as a row of coloured blocks.
    Returns the hover description text if a block is hovered, else None."""
    n = len(_PROPRIO_SPECS)
    total_w = n * PROPRIO_BLOCK + (n - 1) * PROPRIO_GAP
    x0 = (WIDTH - total_w) // 2
    y = PROPRIO_TOP
    bg = PROPRIO_BAR_BG
    pos_c = PROPRIO_BAR_POS
    neg_c = PROPRIO_BAR_NEG

    hover = -1
    for i, (code, key, scale, desc) in enumerate(_PROPRIO_SPECS):
        bx = x0 + i * (PROPRIO_BLOCK + PROPRIO_GAP)
        rect = pygame.Rect(bx, y, PROPRIO_BLOCK, PROPRIO_BLOCK)
        if mouse_pos and rect.collidepoint(mouse_pos):
            hover = i
        val = signals.get(key, 0.0)
        t = max(-1.0, min(1.0, val / scale))
        if t >= 0:
            fill = _lerp_color(bg, pos_c, t)
        else:
            fill = _lerp_color(bg, neg_c, -t)
        pygame.draw.rect(screen, fill, rect, border_radius=4)
        border_c = (140, 150, 170) if hover == i else (60, 66, 82)
        pygame.draw.rect(screen, border_c, rect, 2, border_radius=4)
        # 2-letter label centred
        s = font.render(code, True, PROPRIO_LABEL_C)
        screen.blit(s, (bx + (PROPRIO_BLOCK - s.get_width()) // 2,
                        y + (PROPRIO_BLOCK - s.get_height()) // 2))

    if hover >= 0:
        code, key, scale, desc = _PROPRIO_SPECS[hover]
        val = signals.get(key, 0.0)
        return f"{key}  =  {val:+.1f}"
    return None


# --- extéroception: curseur ---------------------------------------------------

# (2-letter code, dict key, scale, description) — scale maps value to [-1, +1]
_CURSOR_SPECS = [
    ("CD", "curseur_dir",  math.pi, "curseur direction (devant = 0)"),
    ("CP", "curseur_prox", 1.0,    "curseur proximite (proche = 1)"),
    ("CV", "curseur_vx",  1500.0, "curseur vx (mouvement vers devant = +)"),
    ("CW", "curseur_vy",  1500.0, "curseur vy (mouvement vers le haut = +)"),
    ("S0", "son_0",       1.0,    "son cellule 0 (clic souris / espace)"),
    ("S1", "son_1",       1.0,    "son cellule 1"),
    ("S2", "son_2",       1.0,    "son cellule 2"),
    ("S3", "son_3",       1.0,    "son cellule 3"),
    ("S4", "son_4",       1.0,    "son cellule 4"),
]

CURSOR_BLOCK = 40
CURSOR_GAP = 4
CURSOR_TOP = 115


def draw_cursor(screen, font, signals: dict,
                mouse_pos: tuple[int, int] | None = None) -> str | None:
    """Draw the cursor extéroceptive signals as a row of coloured blocks.
    Returns the hover description text if a block is hovered, else None."""
    n = len(_CURSOR_SPECS)
    total_w = n * CURSOR_BLOCK + (n - 1) * CURSOR_GAP
    x0 = (WIDTH - total_w) // 2
    y = CURSOR_TOP
    bg = PROPRIO_BAR_BG
    pos_c = PROPRIO_BAR_POS
    neg_c = PROPRIO_BAR_NEG

    hover = -1
    for i, (code, key, scale, desc) in enumerate(_CURSOR_SPECS):
        bx = x0 + i * (CURSOR_BLOCK + CURSOR_GAP)
        rect = pygame.Rect(bx, y, CURSOR_BLOCK, CURSOR_BLOCK)
        if mouse_pos and rect.collidepoint(mouse_pos):
            hover = i
        val = signals.get(key, 0.0)
        t = max(-1.0, min(1.0, val / scale))
        if t >= 0:
            fill = _lerp_color(bg, pos_c, t)
        else:
            fill = _lerp_color(bg, neg_c, -t)
        pygame.draw.rect(screen, fill, rect, border_radius=4)
        border_c = (140, 150, 170) if hover == i else (60, 66, 82)
        pygame.draw.rect(screen, border_c, rect, 2, border_radius=4)
        s = font.render(code, True, PROPRIO_LABEL_C)
        screen.blit(s, (bx + (CURSOR_BLOCK - s.get_width()) // 2,
                        y + (CURSOR_BLOCK - s.get_height()) // 2))

    if hover >= 0:
        code, key, scale, desc = _CURSOR_SPECS[hover]
        val = signals.get(key, 0.0)
        return f"{key}  =  {val:+.1f}"
    return None


# --- extéroception: vision cone -----------------------------------------------

VISION_C = (90, 100, 120)
VISION_CELL_BORDER = (60, 66, 82)


def draw_vision(screen, font, vision, skel: "B.Skeleton") -> None:
    """Draw the 4×4 retina cone overlaid on the scene. Each cell is a
    semi-transparent quadrilateral filled with its perceived brightness."""
    from . import extero as E

    head = B.head_world(skel)
    forward_angle = 0.0 if skel.facing > 0 else math.pi
    # apex pushed forward past the head (same offset as the sensor)
    apex_x = head.x + E.VISION_APEX_OFFSET * math.cos(forward_angle)
    apex_y = head.y + E.VISION_APEX_OFFSET * math.sin(forward_angle)
    half_fov = E.VISION_FOV / 2
    ang_step = E.VISION_FOV / E.VISION_COLS
    depth_step = E.VISION_RANGE / E.VISION_ROWS

    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    for row in range(E.VISION_ROWS):
        inner = row * depth_step
        outer = (row + 1) * depth_step
        for col in range(E.VISION_COLS):
            a_lo = forward_angle - half_fov + col * ang_step
            a_hi = forward_angle - half_fov + (col + 1) * ang_step

            # 4 corners of the cell (world coords)
            il = (apex_x + inner * math.cos(a_lo),
                  apex_y + inner * math.sin(a_lo))
            ih = (apex_x + inner * math.cos(a_hi),
                  apex_y + inner * math.sin(a_hi))
            ol = (apex_x + outer * math.cos(a_lo),
                  apex_y + outer * math.sin(a_lo))
            oh = (apex_x + outer * math.cos(a_hi),
                  apex_y + outer * math.sin(a_hi))

            pts = [w2s(*il), w2s(*ih), w2s(*oh), w2s(*ol)]

            idx = row * E.VISION_COLS + col
            g = vision.cells[idx]
            gray = int(g * 255)
            pygame.draw.polygon(overlay, (gray, gray, gray, 90), pts)
            pygame.draw.polygon(overlay, (*VISION_CELL_BORDER, 140), pts, 1)

    screen.blit(overlay, (0, 0))


# --- extéroception: touch (contact forces) ------------------------------------

_TOUCH_SPECS = [
    ("SA", "contact_sol_avant",   1500.0, "contact sol patte avant"),
    ("SR", "contact_sol_arriere", 1500.0, "contact sol patte arriere"),
    ("TX", "collision_tronc_x", 1500.0, "collision tronc"),
    ("TY", "collision_tronc_y", 1500.0, "collision tronc"),
    ("TC", "collision_tronc_cx", 200.0, "collision tronc"),
    ("TD", "collision_tronc_cy", 200.0, "collision tronc"),
]

TOUCH_BLOCK = 40
TOUCH_GAP = 4
TOUCH_TOP = 160


def draw_touch(screen, font, signals: dict,
               mouse_pos: tuple[int, int] | None = None) -> str | None:
    """Draw the touch (contact force) signals as coloured blocks.
    Returns the hover description text if a block is hovered, else None."""
    n = len(_TOUCH_SPECS)
    total_w = n * TOUCH_BLOCK + (n - 1) * TOUCH_GAP
    x0 = (WIDTH - total_w) // 2
    y = TOUCH_TOP
    bg = PROPRIO_BAR_BG
    pos_c = PROPRIO_BAR_POS
    neg_c = PROPRIO_BAR_NEG

    hover = -1
    for i, (code, key, scale, desc) in enumerate(_TOUCH_SPECS):
        bx = x0 + i * (TOUCH_BLOCK + TOUCH_GAP)
        rect = pygame.Rect(bx, y, TOUCH_BLOCK, TOUCH_BLOCK)
        if mouse_pos and rect.collidepoint(mouse_pos):
            hover = i
        val = signals.get(key, 0.0)
        t = max(-1.0, min(1.0, val / scale))
        if t >= 0:
            fill = _lerp_color(bg, pos_c, t)
        else:
            fill = _lerp_color(bg, neg_c, -t)
        pygame.draw.rect(screen, fill, rect, border_radius=4)
        border_c = (140, 150, 170) if hover == i else (60, 66, 82)
        pygame.draw.rect(screen, border_c, rect, 2, border_radius=4)
        s = font.render(code, True, PROPRIO_LABEL_C)
        screen.blit(s, (bx + (TOUCH_BLOCK - s.get_width()) // 2,
                        y + (TOUCH_BLOCK - s.get_height()) // 2))

    if hover >= 0:
        code, key, scale, desc = _TOUCH_SPECS[hover]
        val = signals.get(key, 0.0)
        return f"{key}  =  {val:+.1f}"
    return None


# --- extéroception: optical flow ----------------------------------------------

_FLUX_SPECS = [
    ("FS", "flux_surface", 5000.0, "flux surface (aire totale en mouvement)"),
    ("FX", "flux_x",       150.0,  "flux x (deplacement devant = +)"),
    ("FY", "flux_y",       150.0,  "flux y (deplacement vers le haut = +)"),
]

FLUX_BLOCK = 40
FLUX_GAP = 4
FLUX_TOP = 205


def draw_flux(screen, font, signals: dict,
              mouse_pos: tuple[int, int] | None = None) -> str | None:
    """Draw the 3 optical flow signals as coloured blocks.
    Returns the hover description text if a block is hovered, else None."""
    n = len(_FLUX_SPECS)
    total_w = n * FLUX_BLOCK + (n - 1) * FLUX_GAP
    x0 = (WIDTH - total_w) // 2
    y = FLUX_TOP
    bg = PROPRIO_BAR_BG
    pos_c = PROPRIO_BAR_POS
    neg_c = PROPRIO_BAR_NEG

    hover = -1
    for i, (code, key, scale, desc) in enumerate(_FLUX_SPECS):
        bx = x0 + i * (FLUX_BLOCK + FLUX_GAP)
        rect = pygame.Rect(bx, y, FLUX_BLOCK, FLUX_BLOCK)
        if mouse_pos and rect.collidepoint(mouse_pos):
            hover = i
        val = signals.get(key, 0.0)
        t = max(-1.0, min(1.0, val / scale))
        if t >= 0:
            fill = _lerp_color(bg, pos_c, t)
        else:
            fill = _lerp_color(bg, neg_c, -t)
        pygame.draw.rect(screen, fill, rect, border_radius=4)
        border_c = (140, 150, 170) if hover == i else (60, 66, 82)
        pygame.draw.rect(screen, border_c, rect, 2, border_radius=4)
        s = font.render(code, True, PROPRIO_LABEL_C)
        screen.blit(s, (bx + (FLUX_BLOCK - s.get_width()) // 2,
                        y + (FLUX_BLOCK - s.get_height()) // 2))

    if hover >= 0:
        code, key, scale, desc = _FLUX_SPECS[hover]
        val = signals.get(key, 0.0)
        return f"{key}  =  {val:+.1f}"
    return None


# --- reward + intéroception ---------------------------------------------------

REWARD_BLOCK = 40
REWARD_GAP = 4
REWARD_TOP = 250

REWARD_POS_C = (110, 200, 140)   # green = comfort (reward)
REWARD_NEG_C = (200, 110, 110)   # red = costs (penalty)
INTERO_C = (180, 160, 100)      # amber = internal states

_INTERO_SPECS = [
    ("FT", "fatigue",     "fatigue"),
    ("SF", "souffrance",  "souffrance"),
]


def draw_reward(screen, font, signals: dict, intero_signals: dict,
                mouse_pos: tuple[int, int] | None = None) -> str | None:
    """Draw 4 boxes in one row: + (comfort), - (costs), FT, SF.
    On hover of the negative box, detail the 5 cost components (e,d,c,i,v).
    Returns the hover description text, else None."""
    bg = PROPRIO_BAR_BG
    n = 2 + len(_INTERO_SPECS)
    total_w = n * REWARD_BLOCK + (n - 1) * REWARD_GAP
    x0 = (WIDTH - total_w) // 2
    y = REWARD_TOP

    # positive box (confort)
    r_pos = pygame.Rect(x0, y, REWARD_BLOCK, REWARD_BLOCK)
    hover_pos = bool(mouse_pos and r_pos.collidepoint(mouse_pos))
    pos_val = signals.get("reward_pos", 0.0)
    pos_t = max(0.0, min(1.0, -pos_val))
    fill_pos = _lerp_color(bg, REWARD_POS_C, pos_t)
    pygame.draw.rect(screen, fill_pos, r_pos, border_radius=4)
    pygame.draw.rect(screen, (140, 150, 170) if hover_pos else (60, 66, 82),
                     r_pos, 2, border_radius=4)
    s = font.render("+", True, PROPRIO_LABEL_C)
    screen.blit(s, (r_pos.x + (REWARD_BLOCK - s.get_width()) // 2,
                    r_pos.y + (REWARD_BLOCK - s.get_height()) // 2))

    # negative box (costs)
    r_neg = pygame.Rect(x0 + REWARD_BLOCK + REWARD_GAP, y,
                        REWARD_BLOCK, REWARD_BLOCK)
    hover_neg = bool(mouse_pos and r_neg.collidepoint(mouse_pos))
    neg_val = signals.get("reward_neg", 0.0)
    neg_t = max(0.0, min(1.0, neg_val))
    fill_neg = _lerp_color(bg, REWARD_NEG_C, neg_t)
    pygame.draw.rect(screen, fill_neg, r_neg, border_radius=4)
    pygame.draw.rect(screen, (140, 150, 170) if hover_neg else (60, 66, 82),
                     r_neg, 2, border_radius=4)
    s = font.render("-", True, PROPRIO_LABEL_C)
    screen.blit(s, (r_neg.x + (REWARD_BLOCK - s.get_width()) // 2,
                    r_neg.y + (REWARD_BLOCK - s.get_height()) // 2))

    # intero boxes
    hover_intero = -1
    for i, (code, key, desc) in enumerate(_INTERO_SPECS):
        bx = x0 + (2 + i) * (REWARD_BLOCK + REWARD_GAP)
        rect = pygame.Rect(bx, y, REWARD_BLOCK, REWARD_BLOCK)
        if mouse_pos and rect.collidepoint(mouse_pos):
            hover_intero = i
        val = intero_signals.get(key, 0.0)
        t = max(0.0, min(1.0, val))
        fill = _lerp_color(bg, INTERO_C, t)
        pygame.draw.rect(screen, fill, rect, border_radius=4)
        pygame.draw.rect(screen, (140, 150, 170) if hover_intero == i
                        else (60, 66, 82), rect, 2, border_radius=4)
        s = font.render(code, True, PROPRIO_LABEL_C)
        screen.blit(s, (bx + (REWARD_BLOCK - s.get_width()) // 2,
                        y + (REWARD_BLOCK - s.get_height()) // 2))

    if hover_pos:
        return f"confort = {pos_val:+.2f}"
    if hover_neg:
        e = signals.get("effort", 0.0)
        d = signals.get("douleur", 0.0)
        c = signals.get("courbature", 0.0)
        i = signals.get("instabilite", 0.0)
        v = signals.get("vertige", 0.0)
        return (f"couts: e={e:.2f} d={d:.2f} c={c:.2f} "
                f"i={i:.2f} v={v:.2f}  |  total={neg_val:.2f}")
    if hover_intero >= 0:
        code, key, desc = _INTERO_SPECS[hover_intero]
        val = intero_signals.get(key, 0.0)
        return f"{desc}  =  {val:.2f}"
    return None


# --- token stream display -----------------------------------------------------

TOKEN_PANEL_X = WIDTH - 190
TOKEN_PANEL_W = 180
TOKEN_PANEL_Y = 10
TOKEN_PANEL_H = HEIGHT - 20
TOKEN_COL_W = TOKEN_PANEL_W // 2 - 4
TOKEN_LINE_H = 11
TOKEN_SEP_C = (90, 100, 120)
TOKEN_TEXT_C = (140, 140, 150)
TOKEN_SALVE_SEP_C = (50, 54, 66)


def draw_tokens(screen, font_small, salves: list,
                 encoder) -> None:
    """Draw the last N salves as decoded text in 2 columns on the right.
    Most recent salves at the bottom. Auto-scrolls to show the latest."""
    # flatten all salves into lines, skipping vision tokens (too many)
    all_lines: list[tuple[str, int]] = []
    for si, salve in enumerate(salves):
        if si > 0:
            all_lines.append(("---", 2))
        # token index 4 is the VISION/Luminosité channel — too many signals
        for tidx, line in enumerate(encoder.decode(salve)):
            if tidx == 4:
                continue
            all_lines.append((line, 0))

    # how many lines fit per column?
    max_per_col = TOKEN_PANEL_H // TOKEN_LINE_H
    total_visible = 2 * max_per_col
    n = len(all_lines)
    start = max(0, n - total_visible)
    visible = all_lines[start:]

    # fill column 0 top-to-bottom, then column 1
    col_n = min(len(visible), max_per_col)

    for idx, (text, ltype) in enumerate(visible):
        col = 0 if idx < col_n else 1
        row = idx if col == 0 else idx - col_n
        x = TOKEN_PANEL_X + col * (TOKEN_COL_W + 4)
        y = TOKEN_PANEL_Y + row * TOKEN_LINE_H
        if ltype == 2:
            s = font_small.render(text, True, TOKEN_SALVE_SEP_C)
        elif ltype == 1:
            s = font_small.render(text, True, TOKEN_SEP_C)
        else:
            s = font_small.render(text, True, TOKEN_TEXT_C)
        screen.blit(s, (x, y))
