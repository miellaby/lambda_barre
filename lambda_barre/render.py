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


def draw(screen, skel: "B.Skeleton", font, show_targets: bool = True) -> None:
    screen.fill(BG)
    # ground line
    pygame.draw.line(screen, GROUND_C, w2s(-WIDTH, 0), w2s(WIDTH, 0), 4)
    # static platforms (skip ground, already drawn)
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
        pygame.draw.line(screen, (70, 74, 88), w2s(a.x, a.y), w2s(b.x, b.y), 4)

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
    ("CA", "force_contact_sol_avant",  1500.0,           "force contact sol patte avant"),
    ("AR", "membre_angle_arriere",     math.pi,          "angle patte arriere (+ = pied vers avant)"),
    ("DR", "membre_distance_arriere",  32.0,             "distance patte arriere (longueur de patte)"),
    ("FR", "force_actuateur_arriere",  2000.0,           "force actuateur patte arriere (effort musculaire)"),
    ("CR", "force_contact_sol_arriere",1500.0,           "force contact sol patte arriere"),
    ("QA", "queue_angle",              math.pi,          "angle queue (+ = vers avant, rel. neutrale)"),
    ("CQ", "couple_queue",             200000.0,         "couple queue (+ = vers avant)"),
    ("XA", "accel_tete_avant",         2000.0,           "acceleration tete avant (+ = projete en avant)"),
    ("YA", "accel_tete_haut",          2000.0,           "acceleration tete haut (+ = vers le haut)"),
]

PROPRIO_BLOCK = 56
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
                 mouse_pos: tuple[int, int] | None = None) -> None:
    """Draw the 13 proprioceptive signals as a row of coloured blocks with
    2-letter labels overlaid. When the mouse hovers a block, a descriptive
    label is shown below the grid."""
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

    # description label below the grid
    label_y = y + PROPRIO_BLOCK + 8
    if hover >= 0:
        code, key, scale, desc = _PROPRIO_SPECS[hover]
        val = signals.get(key, 0.0)
        text = f"{desc}  =  {val:+.1f}"
    else:
        text = ""
    if text:
        s = font.render(text, True, PROPRIO_LABEL_C)
        screen.blit(s, ((WIDTH - s.get_width()) // 2, label_y))
