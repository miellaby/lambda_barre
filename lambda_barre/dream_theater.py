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


def run(device: str | None = None,
        pol_steps: int = 32,
        wm_epochs: int = 512,
        wm_target_loss: float = 0.01,
        use_kv_cache: bool = True,
        pol_eos_lookahead: bool = True,
        compile_wm: bool = False,
        seq_steps: int = 10,
        n_ctx: int = 4,
        save: bool = True,
        wm_path: str = _WM_CKPT,
        pol_path: str = _POL_CKPT,
        buf_path: str = _BUF_CKPT,
        # Architecture / training params (from config.toml or defaults)
        d_model: int = 64, nhead: int = 4, wm_layers: int = 3,
        dim_ff: int = 384, wm_dropout: float = 0.0,
        pol_hidden: int = 256,
        lr_wm: float = 3e-4, lr_pol: float = 2e-4,
        act_std: float = 0.05, explore_std: float = 0.2,
        wm_batch: int = 64,
        seed: int = 0,
        buffer_capacity: int = 1000, buffer_pool_capacity: int = 1000) -> None:
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

    brain = Brain(device=device, compile_wm=compile_wm, seq_steps=seq_steps, n_ctx=n_ctx,
                  d_model=d_model, nhead=nhead, layers=wm_layers,
                  dim_ff=dim_ff, dropout=wm_dropout,
                  pol_hidden=pol_hidden,
                  lr_wm=lr_wm, lr_pol=lr_pol,
                  act_std=act_std, explore_std=explore_std,
                  seed=seed,
                  buffer_capacity=buffer_capacity,
                  buffer_pool_capacity=buffer_pool_capacity)
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

    dream_theater = DreamTheater(R.WIDTH, R.HEIGHT)
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
                sleep_gen = brain.sleep(wm_epochs=wm_epochs, pol_steps=pol_steps,
                                        wm_target_loss=wm_target_loss,
                                        use_kv_cache=use_kv_cache,
                                        pol_eos_lookahead=pol_eos_lookahead)
                status = "sleeping..."

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
                elif label in ("wm", "wm_coreset", "wm_addendum"):
                    wm_step = getattr(brain, "sleep_wm_step", 0)
                    wm_total = getattr(brain, "sleep_wm_epochs", 0)
                    if label == "wm_addendum":
                        status = f"learning addendum {wm_step}/{wm_total} (loss {value:.2f})"
                    elif label == "wm_coreset":
                        status = f"learning coreset {wm_step}/{wm_total} (loss {value:.2f})"
                    else:
                        status = f"learning wm {wm_step}/{wm_total} (loss {value:.2f})"
                elif label == "filter":
                    status = f"surprise filter: -{value['dropped']} known, {value['kept']} kept"
                elif label == "pol":
                    pol_stats = getattr(brain, "last_pol_stats", None)
                    if pol_stats is not None:
                        status = (f"sleeping: pol {value:.2f} "
                                  f"(hindsight {pol_stats['accepted']}/{pol_stats['total']} accepted)")
                    else:
                        status = f"sleeping: pol {value:.2f}"
                    if brain.last_dream_record is not None:
                        dream_theater.update(brain.last_dream_record)
                elif label == "done":
                    stats = value
                    sleep_gen = None
                    if save:
                        brain.save_checkpoint(wm_path, pol_path, buf_path)
                    print("[sleep]", stats)
                    f_stats = getattr(brain, "last_filter_stats", None)
                    drop_info = f" [dropped: {f_stats['dropped']}]" if f_stats and f_stats.get("dropped", 0) > 0 else ""
                    status = (f"slept: coreset {stats.get('coreset_size', brain.buffer.coreset_size)}{drop_info} "
                              f"wm {stats['wm_loss']:.2f} pol {stats['pol_loss']:.2f}")
            except StopIteration:
                sleep_gen = None

        dream_theater.draw(screen, font, font_small, status=status, brain=brain)
        pygame.display.flip()

    pygame.quit()


def main() -> None:
    cfg = CFG.load_config()

    # Built-in defaults (same as run() signature)
    params: dict = {
        "device": os.environ.get("LAMBDA_DEVICE", "cpu"),
        "accel": bool(os.environ.get("LAMBDA_ACCEL")),
        "pol_steps": 32, "wm_epochs": 512, "wm_target_loss": 0.01,
        "use_kv_cache": True, "pol_eos_lookahead": True,
        "compile_wm": False,
        "seq_steps": 10, "n_ctx": 4,
        "d_model": 64, "nhead": 4, "wm_layers": 3,
        "dim_ff": 384, "wm_dropout": 0.0,
        "pol_hidden": 256,
        "lr_wm": 3e-4, "lr_pol": 2e-4,
        "act_std": 0.05, "explore_std": 0.2,
        "wm_batch": 64,
        "seed": 0,
        "buffer_capacity": 1000, "buffer_pool_capacity": 1000,
        # CLI-only
        "save": True,
        "wm_path": _WM_CKPT, "pol_path": _POL_CKPT, "buf_path": _BUF_CKPT,
    }

    # Override with config.toml
    CFG.apply_toml(params, cfg)

    p = argparse.ArgumentParser(description="Dream Theater — standalone sleep cycle visualizer")
    p.add_argument("--device", type=str, default=argparse.SUPPRESS,
                   choices=["cpu", "cuda", "auto", "cpu-fast"],
                   help="computation device: 'cpu' (default safe), 'cuda', 'auto', or 'cpu-fast'")
    p.add_argument("--accel", action="store_true", default=argparse.SUPPRESS,
                   help="enable maximum hardware acceleration (CUDA if available, else CPU)")
    p.add_argument("--pol-steps", type=int, default=argparse.SUPPRESS,
                   help="policy training steps per sleep cycle")
    p.add_argument("--wm-epochs", type=int, default=argparse.SUPPRESS,
                   help="world model training epochs per sleep cycle")
    p.add_argument("--wm-target-loss", type=float, default=argparse.SUPPRESS,
                   help="target loss for World Model coreset training")
    p.add_argument("--no-kv-cache", action="store_true", default=argparse.SUPPRESS,
                   help="disable the persistent KV-cache for the imagined policy rollouts")
    p.add_argument("--no-eos-lookahead", action="store_true", default=argparse.SUPPRESS,
                   help="disable the EOS-lookahead (hindsight) policy supervision "
                        "and fall back to the legacy stochastic candidates")
    p.add_argument("--compile", action="store_true", default=argparse.SUPPRESS,
                   dest="compile_wm",
                   help="torch.compile the World Model training step")
    p.add_argument("--seq-steps", type=int, default=argparse.SUPPRESS,
                   help="number of multi-step transitions per sequence")
    p.add_argument("--n-ctx", type=int, default=argparse.SUPPRESS,
                   help="decision context step index")
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
    # Architecture / training params (also settable via config.toml)
    p.add_argument("--d-model", type=int, default=argparse.SUPPRESS,
                   help="transformer latent dimension")
    p.add_argument("--nhead", type=int, default=argparse.SUPPRESS,
                   help="transformer attention heads")
    p.add_argument("--wm-layers", type=int, default=argparse.SUPPRESS,
                   help="transformer encoder layers")
    p.add_argument("--dim-ff", type=int, default=argparse.SUPPRESS,
                   help="transformer feed-forward hidden dimension")
    p.add_argument("--wm-dropout", type=float, default=argparse.SUPPRESS,
                   help="transformer dropout rate")
    p.add_argument("--pol-hidden", type=int, default=argparse.SUPPRESS,
                   help="policy network hidden layer width")
    p.add_argument("--lr-wm", type=float, default=argparse.SUPPRESS,
                   help="World Model learning rate")
    p.add_argument("--lr-pol", type=float, default=argparse.SUPPRESS,
                   help="Policy learning rate")
    p.add_argument("--act-std", type=float, default=argparse.SUPPRESS,
                   help="action sampling std during wake")
    p.add_argument("--explore-std", type=float, default=argparse.SUPPRESS,
                   help="action sampling std during imagination")
    p.add_argument("--wm-batch", type=int, default=argparse.SUPPRESS,
                   help="World Model training batch size")
    p.add_argument("--seed", type=int, default=argparse.SUPPRESS,
                   help="random seed")
    p.add_argument("--buffer-capacity", type=int, default=argparse.SUPPRESS,
                   help="session journal capacity (sequences)")
    p.add_argument("--buffer-pool-capacity", type=int, default=argparse.SUPPRESS,
                   help="coreset long-term capacity (sequences)")

    args = p.parse_args()
    cli = vars(args)

    # CLI overrides config
    CFG.merge_cli(params, cli)
    if "no_save" in params:
        params["save"] = not params.pop("no_save")

    # Resolve device: --accel overrides --device
    accel = params.pop("accel", False)
    if accel:
        params["device"] = "auto"

    try:
        run(**params)
    except KeyboardInterrupt:
        pygame.quit()


if __name__ == "__main__":
    main()
