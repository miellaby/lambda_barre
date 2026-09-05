"""Main entry point: a pygame window where you drive λ̄'s consignes by mouse.

Controls (the "manual control" mode of Stage 1 — no brain yet):

  Left hind limb:  drag the left joystick  — direction = foot angle (theta*),
                                            distance from centre = leg length (d*)
  Right hind limb: drag the right joystick — same convention
  Tail:           drag the tail slider     — horizontal travel = tail angle

  SPACE        reset to spawn pose
  G           toggle target markers
  ESC          quit

Joysticks are circles with a free handle: push the handle out to extend the
leg, swing it around to steer the foot. Centre = retracted leg straight down.
"""
from __future__ import annotations

import argparse

import pygame

from . import body as B
from . import world as W
from . import render as R
from . import ui as UI
from . import sensors as S


def hud(font, skel, fps):
    lines = [
        f"fps {fps:4.0f}   facing {'R' if skel.facing == 1 else 'L'}   "
        f"feet {W.foot_contacts(skel)}",
        "drag joysticks (legs) / slider (tail) · SPACE reset · G targets · ESC quit",
    ]
    return [font.render(t, True, R.HUD_C) for t in lines]


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
    show_targets = True

    n = 0
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    B.reset(skel)
                    controls = UI.Controls(skel)
                    proprio.reset()
                elif ev.key == pygame.K_g:
                    show_targets = not show_targets
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                controls.on_down(*ev.pos)
            elif ev.type == pygame.MOUSEMOTION:
                controls.on_motion(*ev.pos)
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                controls.on_up()

        # the widgets are the single source of truth for the consignes
        controls.push(skel)

        # fixed timestep physics, several substeps for stability
        proprio.reset_contacts()
        for _ in range(3):
            W.step(space, skel, 1 / 180)
        signals = proprio.update(skel, 1 / 60)

        if screen is not None:
            R.draw(screen, skel, font, show_targets)
            R.draw_proprio(screen, font, signals, pygame.mouse.get_pos())
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
