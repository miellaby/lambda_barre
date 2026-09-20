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
from .dream import DreamTheater

import os

# Brain parameter checkpoints — saved after each sleep cycle, loaded at startup
# unless --reset is passed. World model and policy are stored separately so a
# change to one does not invalidate the other.
_CKPT_DIR = os.path.dirname(os.path.dirname(__file__))
_WM_CKPT = os.path.join(_CKPT_DIR, "wm_ckpt.pt")
_POL_CKPT = os.path.join(_CKPT_DIR, "pol_ckpt.pt")
_BUF_CKPT = os.path.join(_CKPT_DIR, "buf_ckpt.pt")


SPEED_PRESETS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def hud(font, skel, fps, brain=None, auto=False, status=None, speed=1.0,
        freeze_physics: bool = False, no_smooth: bool = False):
    tags = ["[BRAIN]" if auto else "[manual]"]
    if freeze_physics:
        tags.append("[PHYS FROZEN]")
    if no_smooth:
        tags.append("[NO SMOOTH]")
    tag_str = " ".join(tags)
    lines = [
        f"fps {fps:2.0f} speed {speed:.2f}x facing {'R' if skel.facing == 1 else 'L'} {tag_str}",
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
    text = "R reset · G targets · B brain · S sleep · D dream · [-/+] speed"
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
        device: str | None = None, bootstrap_dataset: int = 0,
        pol_steps: int = 64, wm_epochs: int = 512,
        freeze_physics: bool = False, no_smooth: bool = False) -> None:
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
        for p in (_WM_CKPT, _POL_CKPT, _BUF_CKPT):
            if os.path.exists(p):
                os.remove(p)
        status = "brain CLEARED"
    else:
        try:
            brain.load_checkpoint(_WM_CKPT, _POL_CKPT, _BUF_CKPT)
            status = f"brain restored (pool {brain.buffer.pool_size})"
        except Exception as e:
            print(f"[warning] Checkpoint load: {e}")
            status = "brain restored partially"

    if bootstrap_dataset > 0:
        from .dataset import generate_balance_dataset
        print(f"[bootstrap] Generating {bootstrap_dataset} transitions of physical experience...")
        brain.buffer = generate_balance_dataset(target_transitions=bootstrap_dataset)
        brain.buffer.save(_BUF_CKPT)
        status = f"dataset bootstrapped ({len(brain.buffer)} seqs)"
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
    dream_theater = DreamTheater(R.WIDTH, R.HEIGHT)
    show_dream = False     # manual inspection of last dream trajectory

    prev_facing = skel.facing
    speed_idx = 2  # 1.00x in SPEED_PRESETS
    sim_accum = 0.0

    # Initial sensor read and smoother initialization
    signals = proprio.update(skel, 1.0 / 60.0)
    touch_signals = touch.update(skel, 1.0 / 60.0)
    reward_signals = reward.update(skel, signals, touch_signals, 1.0 / 60.0)
    intero_signals = intero.update(reward_signals, 1.0 / 60.0)
    cursor_signals = cursor.update(skel, (400, 300), 1.0 / 60.0)
    vision_signals = vision.update(skel, 1.0 / 60.0)
    smoother.reinit(signals, touch_signals, cursor_signals, vision_signals, intero_signals, reward_signals)

    n = 0
    running = True
    while running:
        if screen is not None:
            clock.tick(60)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                if show_dream and sleep_gen is None:
                    show_dream = False
                else:
                    running = False
            elif sleep_gen is not None or show_dream:
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_d and sleep_gen is None:
                    show_dream = False
                    continue
                if dream_theater.handle_event(ev):
                    continue
                continue
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_r:
                    B.reset(skel)
                    controls = UI.Controls(skel)
                    proprio.reset()
                    reward.reset()
                    cursor.reset()
                    vision.reset()
                    touch.reset()
                    intero.reset()
                    smoother.reset()
                    sig_r = proprio.update(skel, 1.0 / 60.0)
                    ts_r = touch.update(skel, 1.0 / 60.0)
                    rs_r = reward.update(skel, sig_r, ts_r, 1.0 / 60.0)
                    isg_r = intero.update(rs_r, 1.0 / 60.0)
                    cs_r = cursor.update(skel, (400, 300), 1.0 / 60.0)
                    vs_r = vision.update(skel, 1.0 / 60.0)
                    smoother.reinit(sig_r, ts_r, cs_r, vs_r, isg_r, rs_r)
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
                elif ev.key == pygame.K_d:
                    if brain.last_dream_record is not None:
                        show_dream = True
                        dream_theater.update(brain.last_dream_record)
                        status = "dream viewer"
                    else:
                        status = "no dream record yet (run sleep first)"
                elif ev.key == pygame.K_s:
                    if len(brain.buffer) >= 1:
                        show_dream = False
                        dream_theater.clear()
                        brain.last_dream_record = None
                        dream_theater.is_paused = False
                        sleep_gen = brain.sleep(wm_epochs=wm_epochs, pol_steps=pol_steps)
                        status = "sleeping..."
                    else:
                        status = "need a complete sequence to sleep"
                elif ev.key in (pygame.K_LEFTBRACKET, pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed_idx = max(0, speed_idx - 1)
                    status = f"speed {SPEED_PRESETS[speed_idx]:.2f}x"
                elif ev.key in (pygame.K_RIGHTBRACKET, pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    speed_idx = min(len(SPEED_PRESETS) - 1, speed_idx + 1)
                    status = f"speed {SPEED_PRESETS[speed_idx]:.2f}x"
                elif ev.key in (pygame.K_0, pygame.K_KP0):
                    speed_idx = 2
                    status = "speed 1.00x"
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

        # consume one sleep step per frame to keep the UI responsive
        if sleep_gen is not None and not dream_theater.is_paused:
            try:
                label, value = next(sleep_gen)
                if label == "wm":
                    wm_step = getattr(brain, "sleep_wm_step", 0)
                    wm_total = getattr(brain, "sleep_wm_epochs", 0)
                    if wm_total > 0:
                        status = f"sleeping: wm {wm_step}/{wm_total} (loss {value:.2f})"
                    else:
                        status = f"sleeping: wm {value:.2f}"
                elif label == "pol":
                    status = f"sleeping: pol {value:.2f}"
                    if brain.last_dream_record is not None:
                        dream_theater.update(brain.last_dream_record)
                elif label == "done":
                    stats = value
                    sleep_gen = None
                    brain.save_checkpoint(_WM_CKPT, _POL_CKPT, _BUF_CKPT)
                    print("[sleep]", stats)
                    status = (f"slept: wm {stats['wm_loss']:.2f} "
                              f"pol {stats['pol_loss']:.2f} ")
            except StopIteration:
                sleep_gen = None

        speed = SPEED_PRESETS[speed_idx] if not headless else 1.0

        def sim_step():
            nonlocal prev_facing, wm_accum, pol_accum
            # manual mode; in brain (auto) mode the policy drives them instead.
            if auto:
                controls.sync(skel)
            else:
                controls.drive(skel)

            # fixed timestep physics (3 substeps of 1/180s = 1/60s per sim step)
            touch.reset_contacts()
            if not freeze_physics:
                for _ in range(3):
                    W.step(space, skel, 1.0 / 180.0)
            sig = proprio.update(skel, 1.0 / 60.0)
            ts = touch.update(skel, 1.0 / 60.0)
            rs = reward.update(skel, sig, ts, 1.0 / 60.0)
            isg = intero.update(rs, 1.0 / 60.0)
            mouse_pos = pygame.mouse.get_pos() if screen else (400, 300)
            cs = cursor.update(skel, mouse_pos, 1.0 / 60.0)
            vs = vision.update(skel, 1.0 / 60.0)

            # detect instantaneous facing direction flip (frame of reference change)
            if skel.facing != prev_facing:
                prev_facing = skel.facing
                # 1. Re-initialize smoother without EMA blending to prevent cross-facing signal corruption
                smoother.reinit(sig, ts, cs, vs, isg, rs)
                if sleep_gen is None:
                    # 2. Clear pre-flip wake context history
                    brain.clear_history()
                    # 3. Produce clean salve in new frame, record it, and update WM latent immediately
                    salve = encoder.encode(sig, ts, cs, vs, isg, rs, skel) if no_smooth else smoother.salve(skel)
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
                # update IIR smoother normally every sim step (60 Hz)
                smoother.update(sig, ts, cs, vs, isg, rs, 1.0 / 60.0)

            if sleep_gen is None:  # brain is offline when not sleeping
                # world model tick at 3 Hz (every 20 physics frames, WM_DT = 1/3 s)
                wm_accum += 1.0 / 60.0
                if wm_accum >= WM_DT:
                    wm_accum -= WM_DT
                    salve = encoder.encode(sig, ts, cs, vs, isg, rs, skel) if no_smooth else smoother.salve(skel)
                    salves.append(salve)
                    if len(salves) > 3:
                        salves.pop(0)
                    # even with the brain offline, we record the salve for the next sleep cycle
                    brain.record(salve)
                    if auto:  # produces fresh latent for policy when brain online
                        brain.wake_tick(salve)

                # policy tick at 6 Hz (every 10 physics frames, POL_DT = 1/6 s)
                pol_accum += 1.0 / 60.0
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

            return sig, ts, rs, isg, cs, vs

        if sleep_gen is None and not show_dream:
            sim_accum += speed
            if sim_accum > 16.0:
                sim_accum = speed
            while sim_accum >= 1.0:
                sim_accum -= 1.0
                signals, touch_signals, reward_signals, intero_signals, cursor_signals, vision_signals = sim_step()
                n += 1
                if steps and n >= steps:
                    running = False
                    break
        else:
            sim_accum = 0.0

        if screen is not None:
            if sleep_gen is not None or show_dream:
                dream_theater.draw(screen, font, font_small, status=status)
                pygame.display.flip()
            else:
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
                                             brain, auto, status, speed,
                                             freeze_physics, no_smooth)):
                    screen.blit(surf, (12, 10 + i * 20))
                hint = _keys_hint(font)
                screen.blit(hint, (R.WIDTH - hint.get_width() - 12, 10))
                pygame.display.flip()

        if headless and (steps and n >= steps):
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
    p.add_argument("--bootstrap-dataset", type=int, default=0, nargs="?", const=3000,
                   help="generate a large physical experience dataset (default: 3000 transitions, 100x bigger) before running")
    p.add_argument("--pol-steps", type=int, default=32,
                   help="policy training steps per sleep cycle (default: 32)")
    p.add_argument("--wm-epochs", type=int, default=512,
                   help="world model training epochs per sleep cycle (default: 512)")
    p.add_argument("--freeze-physics", action="store_true",
                   help="freeze physics simulation (for debugging token recordings)")
    p.add_argument("--no-smooth", action="store_true",
                   help="bypass IIR sensor smoother (use raw snapshots)")
    args = p.parse_args()
    if args.headless and not args.steps:
        p.error("--headless requires --steps")
    dev = "auto" if args.accel else args.device
    try:
        run(headless=args.headless, steps=args.steps, reset=args.reset, device=dev,
            bootstrap_dataset=args.bootstrap_dataset, pol_steps=args.pol_steps,
            wm_epochs=args.wm_epochs,
            freeze_physics=args.freeze_physics, no_smooth=args.no_smooth)
    except KeyboardInterrupt:
        pygame.quit()


if __name__ == "__main__":
    main()
