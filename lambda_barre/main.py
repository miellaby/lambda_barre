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

from lambda_barre.models import THETA_RANGE

from . import body as B
from . import world as W
from . import render as R
from . import ui as UI
from . import proprio as S
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
        dev = getattr(brain, "device_desc", str(brain.device))
        lines.append(f"dev {dev}")
        lines.append(
            f"buf {len(brain.buffer)} (pool {brain.buffer.pool_size}, {brain.buffer.num_segments}s) wm {wm:.2f} pol {pol:.2f} ")
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


def run(headless: bool = False, steps: int = 0, reset: bool = False,
        device: str | None = None) -> None:
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
    controls.drive(skel)
    B.apply_consignes(skel)
    proprio = S.Proprio(space, skel)
    reward = S.Reward(skel)
    cursor = E.Cursor(skel)
    vision = E.Vision(skel, space)
    touch = E.Touch(space, skel)
    intero = I.Intero()
    encoder = DenseEncoder()
    brain = Brain(device=device)
    auto = False                 # brain drives the consignes when True
    if reset:
        for p in (_WM_CKPT, _POL_CKPT):
            if os.path.exists(p):
                os.remove(p)
        status = "brain CLEARED"
    else:
        try:
            brain.load_checkpoint(_WM_CKPT, _POL_CKPT)
            status = "brain restored"
        except Exception:
            for p in (_WM_CKPT, _POL_CKPT):
                if os.path.exists(p):
                    os.remove(p)
            status = "brain can't be restored"
    smoother = Smoother(tau=1.0)
    salves: list = []       # buffer of last 10 salves (for display, disabled)
    WM_DT = 1.0 / 3.0       # world model cadence — 3 Hz
    POL_DT = 1.0 / 6.0      # policy cadence — 6 Hz
    wm_accum = WM_DT        # world model tick accumulator (1 Hz)   - starts full to produce a first salve on the first frame
    pol_accum = 0.0         # policy tick accumulator (6 Hz)
    show_targets = True
    drag_plat = None       # (body, shape) of platform being right-dragged
    drag_offset = (0, 0)  # world-space offset from platform centre to mouse
    sleep_gen = None       # active sleep generator (None when not sleeping)

    prev_facing = skel.facing

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
                    salves.clear()
                    wm_accum = WM_DT
                    pol_accum = 0.0
                    prev_facing = skel.facing
                    status = "reset"
                elif ev.key == pygame.K_g:
                    show_targets = not show_targets
                elif ev.key == pygame.K_b:
                    auto = not auto
                    # brain.clear_history() # toggling brain on/off doesn't justify clearing history
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
                    print("[sleep]", stats)
                    status = (f"slept: wm {stats['wm_loss']:.2f} "
                              f"pol {stats['pol_loss']:.2f} ")
            except StopIteration:
                sleep_gen = None

        # manual mode; in brain (auto) mode the policy drives them instead.
        if auto:
            controls.sync(skel)
        else:
            controls.drive(skel)

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

        # detect instantaneous facing direction flip (frame of reference change)
        if skel.facing != prev_facing:
            prev_facing = skel.facing
            # 1. Re-initialize smoother without EMA blending to prevent cross-facing signal corruption
            smoother.reinit(signals, touch_signals, cursor_signals,
                           vision_signals, intero_signals, reward_signals)
            if sleep_gen is None:
                # 2. Clear pre-flip wake context history
                brain.clear_history()
                # 3. Produce clean salve in new frame, record it, and update WM latent immediately
                salve = smoother.salve(skel)
                salves.append(salve)
                if len(salves) > 3:
                    salves.pop(0)
                brain.record(salve)
                if auto:
                    brain.wake_tick(salve)
                    # 4. Immediately re-evaluate policy in the new reference frame
                    theta_front, d_front, theta_back, d_back, tail_t = brain.act(salve)
                    facing = skel.facing
                    theta_front_phys = theta_front * THETA_RANGE
                    theta_back_phys = theta_back * THETA_RANGE
                    d_front_phys = B.LIMB_MIN + d_front * (B.LIMB_MAX - B.LIMB_MIN)
                    d_back_phys = B.LIMB_MIN + d_back * (B.LIMB_MAX - B.LIMB_MIN)
                    tail_t_phys = tail_t * THETA_RANGE
                    if facing == 1:
                        skel.limb_r.theta_star = -theta_front_phys
                        skel.limb_r.d_star = d_front_phys
                        skel.limb_l.theta_star = +theta_back_phys
                        skel.limb_l.d_star = d_back_phys
                    else:
                        skel.limb_l.theta_star = +theta_front_phys
                        skel.limb_l.d_star = d_front_phys
                        skel.limb_r.theta_star = -theta_back_phys
                        skel.limb_r.d_star = d_back_phys
                    skel.tail_act.theta_star = tail_t_phys
                    B.apply_consignes(skel)
                wm_accum = 0.0
                pol_accum = 0.0
        else:
            # update IIR smoother normally every frame (60 Hz)
            smoother.update(signals, touch_signals, cursor_signals,
                            vision_signals, intero_signals, reward_signals, 1 / 60)

        if sleep_gen is None: # brain is offline when not sleeping

            # world model tick at 3 Hz: produce smoothed salve
            wm_accum += dt
            if wm_accum >= WM_DT:
                wm_accum -= WM_DT
                salve = smoother.salve(skel)
                salves.append(salve)
                if len(salves) > 3:
                    salves.pop(0)
                # even with the brain offline, we record the salve for the next sleep cycle
                brain.record(salve)
                if auto: # produces fresh latent for policy when brain online
                    brain.wake_tick(salve)


            # policy tick at 6 Hz: reuse cached latent, produce new consignes
            # (frozen during sleep — the policy is offline)
            pol_accum += dt
            if pol_accum >= POL_DT:
                pol_accum -= POL_DT
                theta_front, d_front, theta_back, d_back, tail_t = brain.act(smoother.salve(skel))
                if auto:
                    # put the consignes back into the skeleton for the next physics step
                    facing = skel.facing
                    theta_front_phys = theta_front * THETA_RANGE
                    theta_back_phys = theta_back * THETA_RANGE
                    d_front_phys = B.LIMB_MIN + d_front * (B.LIMB_MAX - B.LIMB_MIN)
                    d_back_phys = B.LIMB_MIN + d_back * (B.LIMB_MAX - B.LIMB_MIN)
                    tail_t_phys = tail_t * THETA_RANGE

                    if facing == 1:
                        # limb_front = limb_r (right side: outward is right (+x), so theta_r = -theta_front)
                        # limb_back = limb_l (left side: outward is left (-x), so theta_l = +theta_back)
                        skel.limb_r.theta_star = -theta_front_phys
                        skel.limb_r.d_star = d_front_phys
                        skel.limb_l.theta_star = +theta_back_phys
                        skel.limb_l.d_star = d_back_phys
                    else:
                        # limb_front = limb_l (left side: outward is left (-x), so theta_l = +theta_front)
                        # limb_back = limb_r (right side: outward is right (+x), so theta_r = -theta_back)
                        skel.limb_l.theta_star = +theta_front_phys
                        skel.limb_l.d_star = d_front_phys
                        skel.limb_r.theta_star = -theta_back_phys
                        skel.limb_r.d_star = d_back_phys

                    skel.tail_act.theta_star = tail_t_phys

        if screen is not None:
            mouse = pygame.mouse.get_pos()
            R.draw(screen, skel, font, show_targets)
            R.draw_tokens(screen, font_small, salves, encoder)
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
    p.add_argument("--device", type=str, default=os.environ.get("LAMBDA_DEVICE", "cpu"),
                   choices=["cpu", "cuda", "auto", "cpu-fast"],
                   help="computation device: 'cpu' (default safe), 'cuda', 'auto', or 'cpu-fast'")
    p.add_argument("--accel", action="store_true", default=bool(os.environ.get("LAMBDA_ACCEL")),
                   help="enable maximum hardware acceleration (CUDA if available, else CPU)")
    args = p.parse_args()
    if args.headless and not args.steps:
        p.error("--headless requires --steps")
    dev = "auto" if args.accel else args.device
    try:
        run(headless=args.headless, steps=args.steps, reset=args.reset, device=dev)
    except KeyboardInterrupt:
        pygame.quit()


if __name__ == "__main__":
    main()
