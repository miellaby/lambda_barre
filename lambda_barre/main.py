"""Main entry point: a pygame window where you drive λ̄'s consignes by mouse.

Controls (the "manual control" mode of Stage 1 — no brain yet):

  Left hind limb:  drag the left joystick  — direction = foot angle (theta*),
                                            distance from centre = leg length (d*)
  Right hind limb: drag the right joystick — same convention
  Tail:           drag the tail slider     — horizontal travel = tail angle

  BACKSPACE   reset to spawn pose
  G           toggle target markers
  ESC          quit

Joysticks are circles with a free handle: push the handle out to extend the
leg, swing it around to steer the foot. Centre = retracted leg straight down.
"""
from __future__ import annotations

import argparse
import math

import pygame

from . import body as B
from . import world as W
from . import render as R
from . import ui as UI
from . import sensors as S
from . import extero as E


def hud(font, skel, fps):
    lines = [
        f"fps {fps:4.0f}   facing {'R' if skel.facing == 1 else 'L'}",
        "drag joysticks (legs) / slider (tail) · RIGHT-DRAG platforms · "
        "BACK reset · G targets · ESC quit",
    ]
    return [font.render(t, True, R.HUD_C) for t in lines]


def _platform_hit(space, world_pos):
    """Return the (body, shape) of the platform under world_pos, or None."""
    for plat, seg, w, h in space._platforms:
        a = plat.local_to_world(seg.a)
        b = plat.local_to_world(seg.b)
        # distance from point to segment
        abx, aby = b.x - a.x, b.y - a.y
        apx, apy = world_pos[0] - a.x, world_pos[1] - a.y
        ab2 = abx * abx + aby * aby
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2)) if ab2 > 0 else 0.0
        cx, cy = a.x + t * abx, a.y + t * aby
        d = math.hypot(world_pos[0] - cx, world_pos[1] - cy)
        if d <= h + 4:
            return plat, seg
    return None


def run(headless: bool = False, steps: int = 0) -> None:
    pygame.init()
    flags = pygame.SCALED
    screen = pygame.display.set_mode((R.WIDTH, R.HEIGHT), flags) if not headless else None
    font = pygame.font.SysFont("monospace", 16) if screen else None
    clock = pygame.time.Clock()
    pygame.display.set_caption("lambda barre — Stage 1")

    space = W.make_space()
    skel = B.build_skeleton(space)
    controls = UI.Controls(skel)
    controls.push(skel)
    B.apply_consignes(skel)
    proprio = S.Proprio(space, skel)
    cursor = E.Cursor(skel)
    vision = E.Vision(skel, space)
    touch = E.Touch(space, skel)
    show_targets = True
    drag_plat = None       # (body, shape) of platform being right-dragged
    drag_offset = (0, 0)  # world-space offset from platform centre to mouse
    sim_t = 0.0            # simulation time for moving platform

    n = 0
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_BACKSPACE:
                    B.reset(skel)
                    controls = UI.Controls(skel)
                    proprio.reset()
                    cursor.reset()
                    vision.reset()
                    touch.reset()
                elif ev.key == pygame.K_g:
                    show_targets = not show_targets
                else:
                    cursor.on_key(ev.scancode)
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                controls.on_down(*ev.pos)
                cursor.on_click()
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
                wpos = R.s2w(*ev.pos)
                hit = _platform_hit(space, wpos)
                if hit is not None:
                    drag_plat = hit
                    drag_offset = (wpos[0] - hit[0].position.x,
                                   wpos[1] - hit[0].position.y)
            elif ev.type == pygame.MOUSEMOTION:
                controls.on_motion(*ev.pos)
                if drag_plat is not None:
                    wpos = R.s2w(*ev.pos)
                    drag_plat[0].position = (wpos[0] - drag_offset[0],
                                             wpos[1] - drag_offset[1])
                    space.reindex_shape(drag_plat[1])
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                controls.on_up()
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 3:
                drag_plat = None

        # the widgets are the single source of truth for the consignes
        controls.push(skel)

        # animate the moving platform
        sim_t += 1 / 60
        W.update_moving_platform(space, sim_t)

        # fixed timestep physics, several substeps for stability
        touch.reset_contacts()
        for _ in range(3):
            W.step(space, skel, 1 / 180)
        signals = proprio.update(skel, 1 / 60)
        touch_signals = touch.update(skel, 1 / 60)
        cursor_signals = cursor.update(skel, pygame.mouse.get_pos(), 1 / 60)
        vision_signals = vision.update(skel, 1 / 60)

        if screen is not None:
            mouse = pygame.mouse.get_pos()
            R.draw(screen, skel, font, show_targets)
            R.draw_vision(screen, font, vision, skel)
            h_proprio = R.draw_proprio(screen, font, signals, mouse)
            h_touch = R.draw_touch(screen, font, touch_signals, mouse)
            h_flux = R.draw_flux(screen, font, vision_signals, mouse)
            h_cursor = R.draw_cursor(screen, font, cursor_signals, mouse)
            hover_text = h_proprio or h_touch or h_flux or h_cursor
            if hover_text:
                s = font.render(hover_text, True, R.PROPRIO_LABEL_C)
                screen.blit(s, ((R.WIDTH - s.get_width()) // 2, R.HEIGHT - 24))
            controls.draw(screen, font)
            for i, surf in enumerate(hud(font, skel, clock.get_fps())):
                screen.blit(surf, (12, 10 + i * 20))
            pygame.display.flip()
            clock.tick(60)

        n += 1
        if steps and n >= steps:
            running = False

    pygame.quit()


def main() -> None:
    p = argparse.ArgumentParser(description="lambda barre — Stage 1")
    p.add_argument("--headless", action="store_true",
                   help="run without a window (smoke test); requires --steps")
    p.add_argument("--steps", type=int, default=0,
                   help="in headless mode, stop after this many frames")
    args = p.parse_args()
    if args.headless and not args.steps:
        p.error("--headless requires --steps")
    try:
        run(headless=args.headless, steps=args.steps)
    except KeyboardInterrupt:
        pygame.quit()


if __name__ == "__main__":
    main()
