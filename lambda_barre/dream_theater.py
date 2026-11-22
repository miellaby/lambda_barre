"""Standalone Dream Theater: run sleep cycles offline, straight from the
brain checkpoints, without the wake-side physics simulation.

Equivalent to launching the main app and pressing S: the World Model, Policy,
and Experience Buffer are restored from the checkpoints, one sleep cycle runs
as a generator (one step per UI frame), and the Dream Theater renders every
policy-training step. Unless --no-save is passed, the trained checkpoints are
written back at the end of each cycle, exactly like the main app.

Usage:
    .venv/bin/python -m lambda_barre.dream_theater
    .venv/bin/python -m lambda_barre.dream_theater --accel
    dream-theater   (console script)
"""
from __future__ import annotations

import argparse
import os

import pygame

from . import config as CFG
from . import render as R
from .brain import Brain
from .dream import DreamTheater

# Brain parameter checkpoints — same defaults as the main app (repo root).
_CKPT_DIR = os.path.dirname(os.path.dirname(__file__))
_WM_CKPT = os.path.join(_CKPT_DIR, "wm_ckpt.pt")
_POL_CKPT = os.path.join(_CKPT_DIR, "pol_ckpt.pt")
_BUF_CKPT = os.path.join(_CKPT_DIR, "buf_ckpt.pt")


def run(pol_steps: int = 32,
        wm_coreset_epoch: int = 512, wm_addendum_epoch: int = 128,
        wm_batch: int = 64, wm_target_loss: float = 0.004,
        save: bool = True,
        wm_path: str = _WM_CKPT,
        pol_path: str = _POL_CKPT,
        buf_path: str = _BUF_CKPT,
        **params) -> None:
    os.environ["SDL_HINT_RENDER_SCALE_QUALITY"] = "1"
    try:
        import ctypes
        ctypes.CDLL("libSDL2-2.0.so.0").SDL_SetHint(b"SDL_HINT_RENDER_SCALE_QUALITY", b"1")
    except Exception:
        pass
    pygame.init()
    flags = pygame.SCALED | pygame.RESIZABLE
    screen = pygame.display.set_mode((R.WIDTH, R.HEIGHT), flags)
    clock = pygame.time.Clock()
    pygame.display.set_caption("lambda barre — Dream Theater")
    font = pygame.font.SysFont(R.FONT_NAME, 13)
    font_small = pygame.font.SysFont(R.FONT_NAME, 11)

    brain = Brain(**CFG.brain_kwargs(params))
    try:
        brain.load_checkpoint(wm_path, pol_path, buf_path)
        status = f"brain restored (coreset {brain.buffer.coreset_size})"
    except Exception as e:
        print(f"[warning] Checkpoint load: {e}")
        status = "brain restored partially"
    if len(brain.buffer) < 1:
        print("Empty buffer: run the main app (or its dataset bootstrap) before dreaming.")
        pygame.quit()
        return

    dream_theater = DreamTheater(R.WIDTH, R.HEIGHT, max_history=pol_steps)
    dream_theater.coreset_size = brain.buffer.coreset_size
    dream_theater.addendum_size = len(brain.buffer) - brain.buffer.coreset_size
    sleep_gen = None       # active sleep generator (None when not sleeping)

    running = True
    while running:
        clock.tick(60)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and (ev.mod & pygame.KMOD_ALT or pygame.key.get_mods() & pygame.KMOD_ALT):
                pygame.display.toggle_fullscreen()
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False
            elif dream_theater.handle_event(ev):
                continue
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_s and sleep_gen is None:
                dream_theater.clear()
                brain.last_dream_record = None
                dream_theater.is_paused = False
                dream_theater.sleep_active = True
                dream_theater.set_sleep_progress("prune", 0, 0)
                sleep_gen = brain.sleep_cycle(wm_coreset_epoch=wm_coreset_epoch,
                                        wm_addendum_epoch=wm_addendum_epoch,
                                        wm_batch=wm_batch,
                                        pol_steps=pol_steps,
                                        wm_target_loss=wm_target_loss)
                status = "sleeping..."

        # consume one sleep step per frame to keep the UI responsive
        should_step = (sleep_gen is not None) and (not dream_theater.is_paused or dream_theater.step_once)
        if should_step:
            dream_theater.step_once = False
            try:
                label, value = next(sleep_gen)
                if label == "prune":
                    d_intra = value.get("dedup_intra", 0)
                    d_core = value.get("dedup_coreset", 0)
                    d_mode = value.get("dedup_mode", "geom")
                    status = (f"dedup ({d_mode}): -{d_intra} intra, -{d_core} coreset refreshed "
                              f"(coreset {value['coreset']})")
                    dream_theater.set_sleep_progress("prune", 1, 1)
                elif label in ("wm", "wm_coreset", "wm_addendum"):
                    wm_step = getattr(brain, "sleep_wm_step", 0)
                    wm_total = getattr(brain, "sleep_wm_epochs", 0)
                    wb = getattr(brain, "sleep_wm_batch", 0)
                    wB = getattr(brain, "sleep_wm_batches", 0)
                    wn = getattr(brain, "sleep_wm_n", 0)
                    overtime = wm_step - wm_total
                    sling = getattr(brain, "sleep_wm_slingshot", False)
                    if overtime > 0:
                        ot_str = f" (+{overtime} overtime, slingshot)" if sling else f" (+{overtime} overtime)"
                    elif sling:
                        ot_str = " (slingshot)"
                    else:
                        ot_str = ""
                    g_total = wm_total * max(1, wB)
                    g_step = (wm_step - 1) * max(1, wB) + wb
                    name = {"wm_addendum": "addendum (50/50 replay)",
                            "wm_coreset": "coreset"}.get(label, "wm")
                    status = (f"learning {name} epoch {wm_step}/{wm_total}{ot_str} "
                              f"batch {wb}/{wB} ({wn} seq, mse {value:.4f})")
                    dream_theater.set_sleep_progress(label, g_step, g_total)
                elif label == "pol":
                    pol_stats = getattr(brain, "last_pol_stats", None)
                    pol_step = getattr(brain, "sleep_pol_step", 0)
                    pol_total = getattr(brain, "sleep_pol_steps", 0)
                    if pol_stats is not None:
                        obs_n = pol_stats.get("obs", 0)
                        wm_n = pol_stats.get("wm", 0)
                        status = (f"sleeping: pol {value:.4f} "
                                  f"(AWR {pol_stats['accepted']}/{pol_stats['total']} active | "
                                  f"{obs_n} obs, {wm_n} wm)")
                    else:
                        status = f"sleeping: pol {value:.4f}"
                    dream_theater.set_sleep_progress("pol", pol_step, pol_total)
                    if brain.last_dream_record is not None:
                        dream_theater.update(brain.last_dream_record)
                elif label == "done":
                    stats = value
                    sleep_gen = None
                    dream_theater.sleep_active = False
                    dream_theater.set_sleep_progress("done", 1, 1)
                    if save:
                        brain.save_checkpoint(wm_path, pol_path, buf_path)
                    print("[sleep]", stats)
                    evicted = stats.get("cycle", {}).get("evicted", 0)
                    evict_info = f" [evicted: -{evicted}]" if evicted > 0 else ""
                    status = (f"slept: coreset {stats.get('coreset_size', brain.buffer.coreset_size)}{evict_info} "
                              f"wm {stats['wm_loss']:.4f} pol {stats['pol_loss']:.4f}")
            except StopIteration:
                sleep_gen = None
                dream_theater.sleep_active = False

        dream_theater.draw(screen, font, font_small, status=status, brain=brain)
        pygame.display.flip()

    pygame.quit()


def main() -> None:
    cfg = CFG.load_config()

    params: dict = {
        **CFG.BRAIN_DEFAULTS,
        "save": True,
        "wm_path": _WM_CKPT, "pol_path": _POL_CKPT, "buf_path": _BUF_CKPT,
    }

    # Override with config.toml
    CFG.apply_toml(params, cfg)

    p = argparse.ArgumentParser(description="Dream Theater — standalone sleep cycle visualizer")
    p.add_argument("--device", type=str, default=argparse.SUPPRESS,
                   choices=["cpu", "cuda", "auto", "cpu-fast"],
                   help="computation device: 'cpu' (default safe), 'cuda', 'auto', or 'cpu-fast'")
    p.add_argument("--pol-steps", type=int, default=argparse.SUPPRESS,
                   help="policy training steps per sleep cycle")
    p.add_argument("--wm-coreset-epoch", type=int, default=argparse.SUPPRESS,
                   help="world model training epochs per sleep cycle (Coreset phase)")
    p.add_argument("--wm-addendum-epoch", type=int, default=argparse.SUPPRESS,
                   help="world model training epochs per sleep cycle (Addendum phase)")
    p.add_argument("--wm-target-loss", type=float, default=argparse.SUPPRESS,
                   help="target loss for World Model coreset training")
    p.add_argument("--compile", action="store_true", default=argparse.SUPPRESS,
                   dest="compile_wm",
                   help="torch.compile the World Model training step")
    p.add_argument("--no-save", action="store_true", default=argparse.SUPPRESS,
                   help="do not write the trained checkpoints back at the end of "
                        "the sleep cycle (view-only)")
    p.add_argument("--wm", type=str, default=argparse.SUPPRESS,
                   dest="wm_path",
                   help="Path to WorldModel checkpoint (default: repo wm_ckpt.pt)")
    p.add_argument("--pol", type=str, default=argparse.SUPPRESS,
                   dest="pol_path",
                   help="Path to Policy checkpoint (default: repo pol_ckpt.pt)")
    p.add_argument("--buffer", type=str, default=argparse.SUPPRESS,
                   dest="buf_path",
                   help="Path to ExperienceBuffer checkpoint (default: repo buf_ckpt.pt)")
    p.add_argument("--lr-wm", type=float, default=argparse.SUPPRESS,
                   help="World Model learning rate")
    p.add_argument("--lr-pol", type=float, default=argparse.SUPPRESS,
                   help="Policy learning rate")
    p.add_argument("--wm-batch", type=int, default=argparse.SUPPRESS,
                   help="World Model training batch size")
    p.add_argument("--seed", type=int, default=argparse.SUPPRESS,
                   help="random seed")

    args = p.parse_args()
    cli = vars(args)

    # CLI overrides config
    CFG.merge_cli(params, cli)
    if "no_save" in params:
        params["save"] = not params.pop("no_save")

    # Resolve device: accel (config.toml) overrides device
    accel = params.pop("accel", False)
    if accel:
        params["device"] = "auto"

    try:
        run(**params)
    except KeyboardInterrupt:
        pygame.quit()


if __name__ == "__main__":
    main()
