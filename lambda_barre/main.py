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
from . import intero as I
from .tokenize import DenseEncoder
from .brain import Brain
from .smoother import Smoother

import os

# Brain parameter checkpoints — saved after each sleep cycle, loaded at startup
# unless --reset is passed. World model and policy are stored separately so a
# change to one does not invalidate the other.
_CKPT_DIR = os.path.dirname(os.path.dirname(__file__))
_WM_CKPT = os.path.join(_CKPT_DIR, "wm_ckpt.pt")
_POL_CKPT = os.path.join(_CKPT_DIR, "pol_ckpt.pt")


def hud(font, skel, fps, brain=None, auto=False, status=None):
    lines = [
        f"fps {fps:2.0f} facing {'R' if skel.facing == 1 else 'L'} "
        f"{'[BRAIN]' if auto else '[manual]'}",
    ]
    if brain is not None:
        wm = brain.last_wm_loss
        pol = brain.last_pol_loss
        lines.append(
            f"buf {len(brain.buffer)} wm {wm:.2f} pol {pol:.2f} "
            f"ret {brain.last_return:.2f}")
        lines.append(
            f"inf wm {brain._wm_time:.1f}ms pol {brain._pol_time:.1f}ms")
    if status:
        lines.append(status)
    return [font.render(t, True, R.HUD_C) for t in lines]


def _keys_hint(font):
    text = "R reset · G targets · B brain · S sleep"
    return font.render(text, True, R.HUD_C)


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


def run(headless: bool = False, steps: int = 0, reset: bool = False) -> None:
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    flags = pygame.SCALED
    screen = pygame.display.set_mode((R.WIDTH, R.HEIGHT), flags) if not headless else None
    font = pygame.font.SysFont("monospace", 16) if screen else None
    font_small = pygame.font.SysFont("monospace", 10) if screen else None
    clock = pygame.time.Clock()
    pygame.display.set_caption("lambda barre — Stage 1")

    space = W.make_space()
    skel = B.build_skeleton(space)
    controls = UI.Controls(skel)
    controls.push(skel)
    B.apply_consignes(skel)
    proprio = S.Proprio(space, skel)
    reward = S.Reward(skel)
    cursor = E.Cursor(skel)
    vision = E.Vision(skel, space)
    touch = E.Touch(space, skel)
    intero = I.Intero()
    encoder = DenseEncoder()
    brain = Brain()
    auto = False                 # brain drives the consignes when True
    if reset:
        for p in (_WM_CKPT, _POL_CKPT):
            if os.path.exists(p):
                os.remove(p)
        status = "reset (fresh brain)"
    else:
        try:
            brain.load_checkpoint(_WM_CKPT, _POL_CKPT)
            status = "brain loaded from checkpoint"
        except Exception:
            for p in (_WM_CKPT, _POL_CKPT):
                if os.path.exists(p):
                    os.remove(p)
            status = "checkpoint incompatible, fresh brain"
    prev_salve = None           # last salve, for (s_t, a_t, s_{t+1}) logging
    smoother = Smoother(tau=1.0)
    # salves: list = []       # buffer of last 10 salves (for display, disabled)
    wm_accum = 0.0          # world model tick accumulator (1 Hz)
    pol_accum = 0.0         # policy tick accumulator (6 Hz)
    WM_DT = 0.5             # world model cadence — 2 Hz
    POL_DT = 1.0 / 6.0      # policy cadence — 6 Hz
    show_targets = True
    drag_plat = None       # (body, shape) of platform being right-dragged
    drag_offset = (0, 0)  # world-space offset from platform centre to mouse
    sleep_gen = None       # active sleep generator (None when not sleeping)

    n = 0
    running = True
    while running:
        dt = (clock.tick(60) / 1000.0) if screen is not None else 1.0 / 60.0
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    B.reset(skel)
                    controls = UI.Controls(skel)
                    proprio.reset()
                    reward.reset()
                    cursor.reset()
                    vision.reset()
                    touch.reset()
                    intero.reset()
                    smoother.reset()
                    brain.clear_history()
                    # salves.clear()
                    wm_accum = 0.0
                    pol_accum = 0.0
                    prev_salve = None
                    status = "reset"
                elif ev.key == pygame.K_g:
                    show_targets = not show_targets
                elif ev.key == pygame.K_b:
                    auto = not auto
                    prev_salve = None
                    brain.clear_history()
                    status = "BRAIN on" if auto else "BRAIN off (manual)"
                elif ev.key == pygame.K_s:
                    if len(brain.buffer) >= 1:
                        sleep_gen = brain.sleep()
                        status = "sleeping..."
                    else:
                        status = "need a complete sequence to sleep"
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

        # the widgets are the single source of truth for the consignes in
        # consume one sleep step per frame to keep the UI responsive
        if sleep_gen is not None:
            try:
                label, value = next(sleep_gen)
                if label == "wm":
                    status = f"sleeping: wm {value:.2f}"
                elif label == "pol":
                    status = f"sleeping: pol {value:.2f}"
                elif label == "done":
                    stats = value
                    sleep_gen = None
                    brain.save_checkpoint(_WM_CKPT, _POL_CKPT)
                    brain.clear_history()
                    print("[sleep]", stats)
                    status = (f"slept: wm {stats['wm_loss']:.2f} "
                              f"pol {stats['pol_loss']:.2f} "
                              f"ret {stats['return']:.2f}")
            except StopIteration:
                sleep_gen = None

        # manual mode; in brain (auto) mode the policy drives them instead.
        if not auto:
            controls.push(skel)
        elif sleep_gen is None:
            controls.sync(skel)

        # fixed timestep physics, several substeps for stability
        touch.reset_contacts()
        for _ in range(3):
            W.step(space, skel, 1 / 180)
        signals = proprio.update(skel, 1 / 60)
        touch_signals = touch.update(skel, 1 / 60)
        reward_signals = reward.update(skel, signals, touch_signals, 1 / 60)
        intero_signals = intero.update(reward_signals, 1 / 60)
        cursor_signals = cursor.update(skel, pygame.mouse.get_pos(), 1 / 60)
        vision_signals = vision.update(skel, 1 / 60)

        # update IIR smoother every frame (60 Hz)
        smoother.update(signals, touch_signals, cursor_signals,
                        vision_signals, intero_signals, reward_signals, 1 / 60)

        # world model tick at 2 Hz: produce smoothed salve + fresh latent
        wm_accum += dt
        if wm_accum >= WM_DT:
            wm_accum -= WM_DT
            salve = smoother.salve(skel)
            # salves.append(salve)
            # if len(salves) > 10:
            #     salves.pop(0)
            if auto:
                if prev_salve is not None:
                    brain.record(prev_salve, salve)
                brain.wake_tick(salve)
                prev_salve = salve

        # policy tick at 6 Hz: reuse cached latent, produce new consignes
        # (frozen during sleep — the policy is offline)
        pol_accum += dt
        if auto and sleep_gen is None and pol_accum >= POL_DT:
            pol_accum -= POL_DT
            tl, dl, tr, dr, tq = brain.act(prev_salve if prev_salve is not None
                                           else smoother.salve(skel))
            skel.limb_l.theta_star = tl
            skel.limb_l.d_star = dl
            skel.limb_r.theta_star = tr
            skel.limb_r.d_star = dr
            skel.tail_act.theta_star = tq

        if screen is not None:
            mouse = pygame.mouse.get_pos()
            R.draw(screen, skel, font, show_targets)
            # R.draw_tokens(screen, font_small, salves, encoder)
            R.draw_vision(screen, font, vision, skel)
            h_proprio = R.draw_proprio(screen, font, signals, mouse)
            h_touch = R.draw_touch(screen, font, touch_signals, mouse)
            h_flux = R.draw_flux(screen, font, vision_signals, mouse)
            h_cursor = R.draw_cursor(screen, font, cursor_signals, mouse)
            h_reward = R.draw_reward(screen, font, reward_signals,
                                     intero_signals, mouse)
            hover_text = (h_proprio or h_touch or h_flux or h_cursor
                         or h_reward)
            if hover_text:
                s = font.render(hover_text, True, R.PROPRIO_LABEL_C)
                screen.blit(s, ((R.WIDTH - s.get_width()) // 2, R.HEIGHT - 24))
            controls.draw(screen, font)
            for i, surf in enumerate(hud(font, skel, clock.get_fps(),
                                         brain, auto, status)):
                screen.blit(surf, (12, 10 + i * 20))
            hint = _keys_hint(font)
            screen.blit(hint, (R.WIDTH - hint.get_width() - 12, 10))
            pygame.display.flip()

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
    p.add_argument("--reset", action="store_true",
                   help="discard saved brain parameters and start fresh")
    args = p.parse_args()
    if args.headless and not args.steps:
        p.error("--headless requires --steps")
    try:
        run(headless=args.headless, steps=args.steps, reset=args.reset)
    except KeyboardInterrupt:
        pygame.quit()


if __name__ == "__main__":
    main()
