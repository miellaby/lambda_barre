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
import os

import pygame
import torch

from lambda_barre.models import THETA_RANGE

from . import body as B
from . import world as W
from . import render as R
from . import ui as UI
from . import proprio as S
from . import extero as E
from . import intero as I
from .tokenize import DenseEncoder, salve_cost
from . import config as CFG
from .brain import Brain
from .smoother import Smoother
from .dream import CAND_COLORS, DreamStep, DreamTheater, draw_thumbnail

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
    if brain is not None:
        tags.append("[REC]" if brain.is_recording else "[REC PAUSED]")
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
            f"core {brain.buffer.coreset_size} · add {brain.buffer.addendum_size} ({brain.buffer.num_segments}s) wm {wm:.2f} pol {pol:.2f}")
        lines.append(
            f"inf wm {brain._wm_time:.1f}ms pol {brain._pol_time:.1f}ms")
    if status:
        lines.append(status)
    return [font.render(t, False, R.HUD_C) for t in lines]


def _keys_hint(font):
    text = "R reset · G overlays · B brain · S sleep · D dream · [-/+] speed"
    return font.render(text, False, R.HUD_C)


def _wm_view_step(brain: Brain, salve: list) -> DreamStep:
    """Egocentric subjective view of one WM-tick salve: the 13 reception
    (state) tokens plus the 3 action tokens decoded back to the 5 consignes.
    Same decoding path as the WM/Dream theaters, so the thumbnail shows
    exactly what was recorded into the buffer."""
    a_toks = torch.tensor(salve[13:16], dtype=torch.float32,
                          device=brain.device).unsqueeze(0)
    a_vals = brain._decode_action_batch(a_toks)[0].tolist()
    return DreamStep(state_tokens=salve[:13], action=a_vals,
                     step_cost=salve_cost(salve))


def _apply_consignes(skel, theta_front, d_front, theta_back, d_back, tail_t):
    """Push the policy's normalized consignes into the skeleton as physical
    targets (theta_star / d_star) in the anatomical frame."""
    theta_front_phys = theta_front * THETA_RANGE
    theta_back_phys = theta_back * THETA_RANGE
    d_front_phys = B.LIMB_MIN + d_front * (B.LIMB_MAX - B.LIMB_MIN)
    d_back_phys = B.LIMB_MIN + d_back * (B.LIMB_MAX - B.LIMB_MIN)
    tail_t_phys = tail_t * THETA_RANGE
    if skel.facing == 1:
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


def _synthesize_consigne(samples: list[float], threshold: float = 0.0) -> float:
    """Synthesize one representative consigne from the series of samples
    captured between two WM ticks.

    Returns the raw sample with the maximum absolute value (the extreme
    excursion). Ties prefer the last occurrence.

    - Monotonic progression → the extreme endpoint
    - Peak / trough (any position) → the peak / trough sample
    - Noise around a mean → the mean (if |extremum − mean| < threshold)

    The returned value is always a raw sample — never an extrapolated
    curve value. The recorded value is the *intent* (extreme), not the
    *effect* (smoothed) — the motor smoothing (theta_eff) is a separate
    physical layer that the WM must learn to bridge.
    """
    if not samples:
        return 0.0
    if len(samples) == 1:
        return samples[0]

    extremum = max(enumerate(samples),
                   key=lambda iv: (abs(iv[1]), iv[0]))[1]

    if threshold > 0.0:
        mean = sum(samples) / len(samples)
        if abs(extremum - mean) < threshold:
            return mean

    return extremum


def _sample_raw_consignes(skel) -> tuple:
    """Extract raw consigne values from the skeleton (facing-independent).
    Returns (theta_l, d_l, theta_r, d_r, tail_theta)."""
    return (
        skel.limb_l.theta_star, skel.limb_l.d_star,
        skel.limb_r.theta_star, skel.limb_r.d_star,
        skel.tail_act.theta_star,
    )


def _override_consignes(skel, record: tuple) -> tuple:
    """Override the skeleton's consignes with synthesized record values.
    Returns the original values for restoration."""
    orig = _sample_raw_consignes(skel)
    skel.limb_l.theta_star, skel.limb_l.d_star = record[0], record[1]
    skel.limb_r.theta_star, skel.limb_r.d_star = record[2], record[3]
    skel.tail_act.theta_star = record[4]
    return orig


def _restore_consignes(skel, orig: tuple) -> None:
    """Restore the skeleton's consignes after an override."""
    skel.limb_l.theta_star, skel.limb_l.d_star = orig[0], orig[1]
    skel.limb_r.theta_star, skel.limb_r.d_star = orig[2], orig[3]
    skel.tail_act.theta_star = orig[4]


def _synthesize_record(buffer: list) -> tuple | None:
    """Synthesize a single representative consigne tuple from the buffer of
    raw samples captured between two WM ticks. Returns None if the buffer
    is empty (no override needed).

    Thresholds (10% of actuator range) distinguish noise (→ mean) from
    real excursions (→ extremum)."""
    if not buffer:
        return None

    theta_thresh = 0.1 * THETA_RANGE
    d_thresh = 0.1 * (B.LIMB_MAX - B.LIMB_MIN)
    thresholds = (theta_thresh, d_thresh, theta_thresh, d_thresh, theta_thresh)

    n = len(buffer[0])
    return tuple(
        _synthesize_consigne([sample[i] for sample in buffer], thresholds[i])
        for i in range(n)
    )


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


def run(headless: bool = False, steps: int = 0,
        device: str | None = None, bootstrap_dataset: int = 0,
        pol_steps: int = 64, wm_coreset_epoch: int = 512, wm_addendum_epoch: int = 128,
        eos_force_eps_lo: float = 0.15, eos_force_eps_hi: float = 0.40,
        eos_force_amplify: float = 1.0,
        wm_impact_floor: float = 0.25,
        freeze_physics: bool = False, no_smooth: bool = False,
        wm_target_loss: float = 0.01,
        use_kv_cache: bool = True,
        pol_eos_lookahead: bool = True,
        compile_wm: bool = False,
        seq_steps: int = 10,
        n_ctx: int = 4,
        wm_hz: float = 3.0,
        pol_hz: float = 6.0,
        # Transformer architecture (from config.toml or defaults)
        d_model: int = 64, nhead: int = 4, wm_layers: int = 3,
        dim_ff: int = 384, wm_dropout: float = 0.0,
        pol_hidden: int = 256,
        # Training hyperparameters
        lr_wm: float = 3e-4, lr_pol: float = 2e-4,
        act_std: float = 0.05,
        wm_batch: int = 64,
        seed: int = 0,
        # Buffer capacities
        buffer_capacity: int = 1000, buffer_pool_capacity: int = 1000) -> None:
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_HINT_RENDER_SCALE_QUALITY"] = "1"
    try:
        import ctypes
        ctypes.CDLL("libSDL2-2.0.so.0").SDL_SetHint(b"SDL_HINT_RENDER_SCALE_QUALITY", b"1")
    except Exception:
        pass
    pygame.init()
    flags = pygame.SCALED | pygame.RESIZABLE
    screen = pygame.display.set_mode((R.WIDTH, R.HEIGHT), flags) if not headless else None
    # if screen is not None:
    #     try:
    #         import warnings
    #         with warnings.catch_warnings():
    #             warnings.simplefilter("ignore", DeprecationWarning)
    #             win = pygame.Window.from_display_module()
    #             win.size = (R.WIDTH, R.HEIGHT)
    #             win.position = pygame.WINDOWPOS_CENTERED
    #     except Exception:
    #         pass
    font = pygame.font.SysFont(R.FONT_NAME, 13) if screen else None
    font_small = pygame.font.SysFont(R.FONT_NAME, 11) if screen else None
    clock = pygame.time.Clock()
    pygame.display.set_caption("lambda barre — Stage 1")

    space = W.make_space()
    skel = B.build_skeleton(space)
    controls = UI.Controls(skel)
    controls.drive(skel)
    B.apply_consignes(skel)
    proprio = S.Proprio(space, skel)
    reward = S.Reward(skel)
    ball_sensor = E.Ball(skel)
    vision = E.Vision(skel, space)
    touch = E.Touch(space, skel)
    intero = I.Intero()
    encoder = DenseEncoder()
    brain = Brain(device=device, compile_wm=compile_wm, seq_steps=seq_steps, n_ctx=n_ctx,
                  d_model=d_model, nhead=nhead, layers=wm_layers,
                  dim_ff=dim_ff, dropout=wm_dropout,
                  pol_hidden=pol_hidden,
                  lr_wm=lr_wm, lr_pol=lr_pol,
                  act_std=act_std,
                  seed=seed,
                  buffer_capacity=buffer_capacity,
                  buffer_pool_capacity=buffer_pool_capacity,
                  eos_force_eps_lo=eos_force_eps_lo,
                  eos_force_eps_hi=eos_force_eps_hi,
                  eos_force_amplify=eos_force_amplify,
                  wm_impact_floor=wm_impact_floor)
    auto = False                 # brain drives the consignes when True
    try:
        brain.load_checkpoint(_WM_CKPT, _POL_CKPT, _BUF_CKPT)
        status = f"brain restored (coreset {brain.buffer.coreset_size})"
    except Exception as e:
        print(f"[warning] Checkpoint load: {e}")
        status = "brain restored partially"

    if bootstrap_dataset > 0:
        from .dataset import generate_balance_dataset
        print(f"[bootstrap] Generating {bootstrap_dataset} transitions of physical experience...")
        brain.buffer = generate_balance_dataset(target_transitions=bootstrap_dataset,
                                               seq_steps=seq_steps, n_ctx=n_ctx)
        brain.buffer.save(_BUF_CKPT)
        status = f"dataset bootstrapped ({len(brain.buffer)} seqs)"
    smoother = Smoother()
    salves: list = []       # buffer of last 10 salves (for display, disabled)
    wm_view: DreamStep | None = None  # egocentric thumbnail of the last WM tick (wake)
    WM_DT = 1.0 / wm_hz       # world model cadence
    POL_DT = 1.0 / pol_hz     # policy cadence
    wm_accum = 0.0            # WM record accumulator (fires at wm_hz, every other brain tick)
    pol_accum = 0.0           # brain tick accumulator (fires at pol_hz)
    consigne_buffer: list = []  # raw consigne samples between WM ticks (for synthesis)
    show_overlays = True   # G: toggle targets, egocentric view, and token panel
    drag_plat = None       # (body, shape) of platform being right-dragged
    drag_offset = (0, 0)  # world-space offset from platform centre to mouse
    drag_ball = False      # whether red balloon is being left-dragged
    drag_ball_offset = (0.0, 0.0)
    drag_ball_prev_pos = None
    drag_ball_vel = (0.0, 0.0)
    sleep_gen = None       # active sleep generator (None when not sleeping)
    dream_theater = DreamTheater(R.WIDTH, R.HEIGHT)
    dream_theater.coreset_size = brain.buffer.coreset_size
    dream_theater.addendum_size = len(brain.buffer) - brain.buffer.coreset_size
    show_dream = False     # manual inspection of last dream trajectory

    prev_facing = skel.facing
    speed_idx = 2  # 1.00x in SPEED_PRESETS
    sim_accum = 0.0

    # Initial sensor read and smoother initialization
    signals = proprio.update(skel, 1.0 / 60.0)
    touch_signals = touch.update(skel, 1.0 / 60.0)
    reward_signals = reward.update(skel, signals, touch_signals, 1.0 / 60.0)
    intero_signals = intero.update(reward_signals, 1.0 / 60.0)
    ball_signals = ball_sensor.update(skel, getattr(space, "ball", (400, 300)), 1.0 / 60.0)
    vision_signals = vision.update(skel, 1.0 / 60.0)
    smoother.reinit(signals, touch_signals, ball_signals, vision_signals, intero_signals, reward_signals)

    n = 0
    running = True
    while running:
        if screen is not None:
            clock.tick(60)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and (ev.mod & pygame.KMOD_ALT or pygame.key.get_mods() & pygame.KMOD_ALT):
                pygame.display.toggle_fullscreen()
                status = "fullscreen on" if pygame.display.is_fullscreen() else "fullscreen off"
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                if show_dream and sleep_gen is None:
                    show_dream = False
                else:
                    running = False
            elif sleep_gen is not None or show_dream:
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_d and sleep_gen is None:
                    show_dream = False
                    continue
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_s and sleep_gen is None:
                    if len(brain.buffer) >= 1:
                        show_dream = False
                        dream_theater.clear()
                        brain.last_dream_record = None
                        dream_theater.is_paused = False
                        dream_theater.sleep_active = True
                        dream_theater.set_sleep_progress("prune", 0, 0)
                        sleep_gen = brain.sleep(wm_coreset_epoch=wm_coreset_epoch,
                                                wm_addendum_epoch=wm_addendum_epoch,
                                                wm_batch=wm_batch,
                                                pol_steps=pol_steps,
                                                wm_target_loss=wm_target_loss,
                                                use_kv_cache=use_kv_cache,
                                                pol_eos_lookahead=pol_eos_lookahead)
                        status = "sleeping..."
                    else:
                        status = "need a complete sequence to sleep"
                    continue
                if dream_theater.handle_event(ev):
                    continue
                continue
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_r:
                    B.reset(skel)
                    W.reset_ball(space)
                    controls = UI.Controls(skel)
                    proprio.reset()
                    reward.reset()
                    ball_sensor.reset()
                    vision.reset()
                    touch.reset()
                    intero.reset()
                    smoother.reset()
                    sig_r = proprio.update(skel, 1.0 / 60.0)
                    ts_r = touch.update(skel, 1.0 / 60.0)
                    rs_r = reward.update(skel, sig_r, ts_r, 1.0 / 60.0)
                    isg_r = intero.update(rs_r, 1.0 / 60.0)
                    cs_r = ball_sensor.update(skel, getattr(space, "ball", (400, 300)), 1.0 / 60.0)
                    vs_r = vision.update(skel, 1.0 / 60.0)
                    smoother.reinit(sig_r, ts_r, cs_r, vs_r, isg_r, rs_r)
                    brain.clear_history()
                    salves.clear()
                    consigne_buffer.clear()
                    wm_view = None
                    wm_accum = 0.0
                    pol_accum = 0.0
                    prev_facing = skel.facing
                    status = "reset"
                elif ev.key == pygame.K_g:
                    show_overlays = not show_overlays
                    status = "overlays on" if show_overlays else "overlays off"
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
                        dream_theater.sleep_active = True
                        dream_theater.set_sleep_progress("prune", 0, 0)
                        sleep_gen = brain.sleep(wm_coreset_epoch=wm_coreset_epoch,
                                                wm_addendum_epoch=wm_addendum_epoch,
                                                wm_batch=wm_batch,
                                                pol_steps=pol_steps,
                                                wm_target_loss=wm_target_loss,
                                                use_kv_cache=use_kv_cache,
                                                pol_eos_lookahead=pol_eos_lookahead)
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
                    ball_sensor.on_key(ev.scancode)
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                controls.on_down(*ev.pos)
                wpos = R.s2w(*ev.pos)
                ball = getattr(space, "ball", None)
                if ball is not None:
                    dist_to_ball = math.hypot(wpos[0] - ball.position.x, wpos[1] - ball.position.y)
                    ball_r = getattr(space, "ball_shape", None).radius if hasattr(space, "ball_shape") else 16.0
                    if dist_to_ball <= ball_r + 14.0:
                        drag_ball = True
                        drag_ball_offset = (wpos[0] - ball.position.x, wpos[1] - ball.position.y)
                        drag_ball_prev_pos = (ball.position.x, ball.position.y)
                        drag_ball_vel = (0.0, 0.0)
                        ball.velocity = (0.0, 0.0)
                ball_sensor.on_click()
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
                wpos = R.s2w(*ev.pos)
                hit = _platform_hit(space, wpos)
                if hit is not None:
                    drag_plat = hit
                    drag_offset = (wpos[0] - hit[0].position.x,
                                   wpos[1] - hit[0].position.y)
            elif ev.type == pygame.MOUSEMOTION:
                controls.on_motion(*ev.pos)
                wpos = R.s2w(*ev.pos)
                if drag_plat is not None:
                    drag_plat[0].position = (wpos[0] - drag_offset[0],
                                             wpos[1] - drag_offset[1])
                    space.reindex_shape(drag_plat[1])
                if drag_ball and hasattr(space, "ball") and space.ball is not None:
                    target_x = wpos[0] - drag_ball_offset[0]
                    target_y = wpos[1] - drag_ball_offset[1]
                    if drag_ball_prev_pos is not None:
                        dt = 1.0 / 60.0
                        vx = (target_x - drag_ball_prev_pos[0]) / dt
                        vy = (target_y - drag_ball_prev_pos[1]) / dt
                        drag_ball_vel = (vx * 0.9, vy * 0.9)
                    space.ball.position = (target_x, target_y)
                    space.ball.velocity = drag_ball_vel
                    drag_ball_prev_pos = (target_x, target_y)
                    if hasattr(space, "ball_shape"):
                        space.reindex_shape(space.ball_shape)
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                controls.on_up()
                if drag_ball and hasattr(space, "ball") and space.ball is not None:
                    space.ball.velocity = drag_ball_vel
                drag_ball = False
                drag_ball_prev_pos = None
                drag_ball_vel = (0.0, 0.0)
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 3:
                drag_plat = None

        # consume one sleep step per frame to keep the UI responsive
        should_step = (sleep_gen is not None) and (not dream_theater.is_paused or dream_theater.step_once)
        if should_step:
            dream_theater.step_once = False
            try:
                label, value = next(sleep_gen)
                if label == "prune":
                    d_intra = value.get('dedup_intra', 0)
                    d_core = value.get('dedup_coreset', 0)
                    status = f"pruned {value['pruned']} (coreset {value['coreset']} | duplicates: -{d_intra} intra, -{d_core} coreset)"
                    dream_theater.set_sleep_progress("prune", 1, 1)
                elif label in ("wm", "wm_coreset", "wm_addendum"):
                    wm_step = getattr(brain, "sleep_wm_step", 0)
                    wm_total = getattr(brain, "sleep_wm_epochs", 0)
                    if label == "wm_addendum":
                        status = f"learning addendum {wm_step}/{wm_total} (loss {value:.2f})"
                    elif label == "wm_coreset":
                        status = f"learning coreset {wm_step}/{wm_total} (loss {value:.2f})"
                    else:
                        status = f"learning wm {wm_step}/{wm_total} (loss {value:.2f})"
                    dream_theater.set_sleep_progress("wm", wm_step, wm_total)
                elif label == "filter":
                    status = f"surprise filter: -{value['dropped']} known, {value['kept']} kept"
                    dream_theater.set_sleep_progress("filter", 1, 1)
                elif label == "pol":
                    pol_stats = getattr(brain, "last_pol_stats", None)
                    pol_step = getattr(brain, "sleep_pol_step", 0)
                    pol_total = getattr(brain, "sleep_pol_steps", 0)
                    if pol_stats is not None:
                        status = (f"sleeping: pol {value:.2f} "
                                  f"(hindsight {pol_stats['accepted']}/{pol_stats['total']} accepted)")
                    else:
                        status = f"sleeping: pol {value:.2f}"
                    dream_theater.set_sleep_progress("pol", pol_step, pol_total)
                    if brain.last_dream_record is not None:
                        dream_theater.update(brain.last_dream_record)
                elif label == "done":
                    stats = value
                    sleep_gen = None
                    dream_theater.sleep_active = False
                    dream_theater.set_sleep_progress("done", 1, 1)
                    brain.save_checkpoint(_WM_CKPT, _POL_CKPT, _BUF_CKPT)
                    print("[sleep]", stats)
                    f_stats = getattr(brain, "last_filter_stats", None)
                    drop_info = f" [dropped: {f_stats['dropped']}]" if f_stats and f_stats.get("dropped", 0) > 0 else ""
                    status = (f"slept: coreset {stats.get('coreset_size', brain.buffer.coreset_size)}{drop_info} "
                              f"wm {stats['wm_loss']:.2f} pol {stats['pol_loss']:.2f}")
            except StopIteration:
                sleep_gen = None
                dream_theater.sleep_active = False

        speed = SPEED_PRESETS[speed_idx] if not headless else 1.0

        def sim_step():
            nonlocal prev_facing, wm_accum, pol_accum, wm_view, drag_ball_vel

            def encode_salve():
                return encoder.encode(sig, ts, cs, vs, isg, rs, skel) if no_smooth else smoother.salve(skel)

            # 1. Controls (manual joystick or brain sync)
            if auto:
                controls.sync(skel)
            else:
                controls.drive(skel)

            # 2. Physics (60 Hz, 3 substeps of 1/180s)
            touch.reset_contacts()
            if not freeze_physics:
                for _ in range(3):
                    W.step(space, skel, 1.0 / 180.0)
                if drag_ball and hasattr(space, "ball") and space.ball is not None:
                    if drag_ball_prev_pos is not None:
                        space.ball.position = drag_ball_prev_pos
                    space.ball.velocity = drag_ball_vel
                    drag_ball_vel = (drag_ball_vel[0] * 0.5, drag_ball_vel[1] * 0.5)

            # 3. Sensors
            sig = proprio.update(skel, 1.0 / 60.0)
            ts = touch.update(skel, 1.0 / 60.0)
            rs = reward.update(skel, sig, ts, 1.0 / 60.0)
            isg = intero.update(rs, 1.0 / 60.0)
            ball_target = getattr(space, "ball", None)
            cs = ball_sensor.update(skel, ball_target if ball_target is not None else (400, 300), 1.0 / 60.0)
            vs = vision.update(skel, 1.0 / 60.0)

            # 4. Smoother + facing flip
            if skel.facing != prev_facing:
                prev_facing = skel.facing
                smoother.reinit(sig, ts, cs, vs, isg, rs)
                if sleep_gen is None:
                    brain.clear_history()
                    consigne_buffer.clear()
                    wm_accum = 0.0
                    pol_accum = 0.0
            else:
                smoother.update(sig, ts, cs, vs, isg, rs, 1.0 / 60.0)

            # 5. Brain — offline during sleep
            if sleep_gen is not None:
                return sig, ts, rs, isg, cs, vs

            # Sample raw consignes for synthesis (before brain tick may change them)
            consigne_buffer.append(_sample_raw_consignes(skel))

            if auto:
                # Brain tick at policy cadence (6 Hz): WM forward + policy action.
                # The latent is always fresh — never stale.
                pol_accum += 1.0 / 60.0
                if pol_accum >= POL_DT:
                    pol_accum -= POL_DT
                    salve = encode_salve()
                    theta_front, d_front, theta_back, d_back, tail_t = brain.brain_tick(salve)
                    _apply_consignes(skel, theta_front, d_front, theta_back, d_back, tail_t)
                    # Re-encode so the action tokens carry the fresh consigne
                    salve = encode_salve()

                    # Record at WM cadence (3 Hz = every other brain tick)
                    wm_accum += POL_DT
                    if wm_accum >= WM_DT:
                        wm_accum -= WM_DT
                        consigne_buffer.clear()
                        brain.record_tick(salve)
                        salves.append(salve)
                        if len(salves) > 3:
                            salves.pop(0)
                        wm_view = _wm_view_step(brain, salve)
            else:
                # Manual mode: record tick at WM cadence — synthesize the
                # consigne from the buffer (the user may have changed it many
                # times between two records; the snapshot alone is not
                # representative). No WM inference here: the latent is only
                # consumed in auto mode, where brain_tick refreshes it.
                wm_accum += 1.0 / 60.0
                if wm_accum >= WM_DT:
                    wm_accum -= WM_DT
                    record = _synthesize_record(consigne_buffer)
                    if record is not None:
                        orig = _override_consignes(skel, record)
                        salve = encode_salve()
                        _restore_consignes(skel, orig)
                    else:
                        salve = encode_salve()
                    consigne_buffer.clear()
                    brain.record_tick(salve)
                    salves.append(salve)
                    if len(salves) > 3:
                        salves.pop(0)
                    wm_view = _wm_view_step(brain, salve)

            return sig, ts, rs, isg, cs, vs

        if sleep_gen is None and not show_dream:
            sim_accum += speed
            if sim_accum > 16.0:
                sim_accum = speed
            while sim_accum >= 1.0:
                sim_accum -= 1.0
                signals, touch_signals, reward_signals, intero_signals, ball_signals, vision_signals = sim_step()
                n += 1
                if steps and n >= steps:
                    running = False
                    break
        else:
            sim_accum = 0.0

        if screen is not None:
            if sleep_gen is not None or show_dream:
                dream_theater.draw(screen, font, font_small, status=status, brain=brain)
                pygame.display.flip()
            else:
                mouse = pygame.mouse.get_pos()
                R.draw(screen, skel, font, show_overlays, mouse_pos=mouse)
                if show_overlays:
                    R.draw_tokens(screen, font_small, salves, encoder)
                R.draw_vision(screen, font, vision, skel)
                h_proprio = R.draw_proprio(screen, font, signals, mouse)
                h_touch = R.draw_touch(screen, font, touch_signals, mouse)
                h_flux = R.draw_flux(screen, font, vision_signals, mouse)
                h_ball = R.draw_ball(screen, font, ball_signals, mouse)
                h_reward = R.draw_reward(screen, font, reward_signals,
                                         intero_signals, mouse)
                hover_text = (h_proprio or h_touch or h_flux or h_ball
                             or h_reward)
                if hover_text:
                    s = font.render(hover_text, False, R.PROPRIO_LABEL_C)
                    screen.blit(s, ((R.WIDTH - s.get_width()) // 2, R.HEIGHT - 24))
                controls.draw(screen, font)
                hud_surfs = hud(font, skel, clock.get_fps(), brain, auto,
                                status, speed, freeze_physics, no_smooth)
                for i, surf in enumerate(hud_surfs):
                    screen.blit(surf, (12, 10 + i * 20))
                if show_overlays and wm_view is not None:
                    # egocentric subjective view (reception + action tokens) of
                    # the last WM tick, refreshed at WM cadence (wm_hz)
                    th_rect = pygame.Rect(12, 10 + len(hud_surfs) * 20 + 6, 92, 68)
                    draw_thumbnail(screen, th_rect, wm_view, font_small,
                                   facing=skel.facing, cand_color=CAND_COLORS[0],
                                   highlight_label="s+a (wm)")
                hint = _keys_hint(font)
                screen.blit(hint, (R.WIDTH - hint.get_width() - 12, 10))
                pygame.display.flip()

        if headless and (steps and n >= steps):
            running = False

    pygame.quit()


def main() -> None:
    cfg = CFG.load_config()

    # Defaults matching run() signature
    params: dict = {
        "headless": False, "steps": 0, "reset": False,
        "device": os.environ.get("LAMBDA_DEVICE", "cpu"),
        "accel": bool(os.environ.get("LAMBDA_ACCEL")),
        "bootstrap_dataset": 0,
        "pol_steps": 32, "wm_coreset_epoch": 512, "wm_addendum_epoch": 128, "wm_batch": 64,
        "eos_force_eps_lo": 0.15, "eos_force_eps_hi": 0.40,
        "eos_force_amplify": 1.0,
        "wm_impact_floor": 0.25,
        "freeze_physics": False, "no_smooth": False,
        "wm_target_loss": 0.01,
        "use_kv_cache": True, "pol_eos_lookahead": True,
        "compile_wm": False,
        "seq_steps": 10, "n_ctx": 4,
        "wm_hz": 3.0, "pol_hz": 6.0,
        "d_model": 64, "nhead": 4, "wm_layers": 3,
        "dim_ff": 384, "wm_dropout": 0.0,
        "pol_hidden": 256,
        "lr_wm": 3e-4, "lr_pol": 2e-4,
        "act_std": 0.05,
        "seed": 0,
        "buffer_capacity": 1000, "buffer_pool_capacity": 1000,
    }

    # Override with config.toml
    CFG.apply_toml(params, cfg)

    # CLI arguments — SUPPRESS default so we can detect which were explicitly set
    p = argparse.ArgumentParser(description="lambda barre — Stage 1")
    p.add_argument("--headless", action="store_true", default=argparse.SUPPRESS,
                   help="run without a window (smoke test); requires --steps")
    p.add_argument("--steps", type=int, default=argparse.SUPPRESS,
                   help="in headless mode, stop after this many frames")
    p.add_argument("--reset", action="store_true", default=argparse.SUPPRESS,
                   help="delete brain checkpoints and exit (does not start the app)")
    p.add_argument("--device", type=str, default=argparse.SUPPRESS,
                   choices=["cpu", "cuda", "auto", "cpu-fast"],
                   help="computation device: 'cpu' (default safe), 'cuda', 'auto', or 'cpu-fast'")
    p.add_argument("--accel", action="store_true", default=argparse.SUPPRESS,
                   help="enable maximum hardware acceleration (CUDA if available, else CPU)")
    p.add_argument("--bootstrap-dataset", type=int, default=argparse.SUPPRESS,
                   nargs="?", const=3000,
                   help="generate a large physical experience dataset before running")
    p.add_argument("--pol-steps", type=int, default=argparse.SUPPRESS,
                   help="policy training steps per sleep cycle")
    p.add_argument("--wm-coreset-epoch", type=int, default=argparse.SUPPRESS,
                   help="world model training epochs per sleep cycle (Coreset phase)")
    p.add_argument("--wm-addendum-epoch", type=int, default=argparse.SUPPRESS,
                   help="world model training epochs per sleep cycle (Addendum phase)")
    p.add_argument("--wm-batch", type=int, default=argparse.SUPPRESS,
                   help="world model training batch size")
    p.add_argument("--wm-target-loss", type=float, default=argparse.SUPPRESS,
                   help="target loss for World Model coreset training")
    p.add_argument("--freeze-physics", action="store_true", default=argparse.SUPPRESS,
                   help="freeze physics simulation (for debugging token recordings)")
    p.add_argument("--no-smooth", action="store_true", default=argparse.SUPPRESS,
                   help="bypass IIR sensor smoother (use raw snapshots)")
    p.add_argument("--no-kv-cache", action="store_true", default=argparse.SUPPRESS,
                   help="disable the persistent KV-cache for imagined policy rollouts")
    p.add_argument("--no-eos-lookahead", action="store_true", default=argparse.SUPPRESS,
                   help="disable EOS-lookahead policy supervision (fall back to stochastic candidates)")
    p.add_argument("--compile", action="store_true", default=argparse.SUPPRESS,
                   dest="compile_wm",
                   help="torch.compile the World Model training step")
    # NOTE: checkpoint-structure options (--seq-steps, --n-ctx, --wm-hz,
    # --pol-hz, --d-model, --nhead, --wm-layers, --dim-ff, --wm-dropout,
    # --pol-hidden, --buffer-capacity, --buffer-pool-capacity) are
    # intentionally NOT exposed on the CLI.  They define the shape of
    # wm_ckpt.pt, pol_ckpt.pt and buf_ckpt.pt — changing any of them
    # invalidates existing checkpoints.  Set them in config.toml instead;
    # after a change, run with --reset to discard old checkpoints and
    # start fresh.
    # Training hyperparameters
    p.add_argument("--lr-wm", type=float, default=argparse.SUPPRESS,
                   help="World Model learning rate")
    p.add_argument("--lr-pol", type=float, default=argparse.SUPPRESS,
                   help="Policy learning rate")
    p.add_argument("--act-std", type=float, default=argparse.SUPPRESS,
                   help="action sampling std during wake")
    p.add_argument("--seed", type=int, default=argparse.SUPPRESS,
                   help="random seed")

    args = p.parse_args()
    cli = vars(args)

    # CLI overrides config
    CFG.merge_cli(params, cli)

    # Resolve device: --accel overrides --device
    accel = params.pop("accel")
    if accel:
        params["device"] = "auto"

    # --reset: delete checkpoints and exit without starting the app
    if params.pop("reset", False):
        for ckpt in (_WM_CKPT, _POL_CKPT, _BUF_CKPT):
            if os.path.exists(ckpt):
                os.remove(ckpt)
                print(f"removed {os.path.basename(ckpt)}")
        print("brain checkpoints cleared")
        return

    if params["headless"] and not params["steps"]:
        p.error("--headless requires --steps")

    try:
        run(**params)
    except KeyboardInterrupt:
        pygame.quit()


if __name__ == "__main__":
    main()
