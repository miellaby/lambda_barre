"""WM Manager: single-sequence World Model training lab.

Extends WMTheater with a live training loop on the selected coreset sequence:
- Select a sequence exactly like in WM Theater (slider, arrows, sort modes).
- Row 1: ground truth. Row 2: WM autoregressive rollout conditioned on s0..s4a4.
- [T] toggles training: each epoch runs one WM optimization step on the selected
  sequence only (causal pass, alternating with an EOS-lookahead pass like the
  sleep cycle), then the rollout in row 2 is recomputed so the new inference is
  visible after every epoch.
- [Space] runs a single epoch while paused, [R] resets the loss history,
  [ / ] halve/double lr_wm live, [Up/Down] change epochs per frame,
  [V] saves the WM back to the checkpoint file.

Usage:
    python -m lambda_barre.wm_manager
    python -m lambda_barre.wm_manager --wm wm_ckpt.pt --buffer buf_ckpt.pt --index 0 --sort coreset
"""
from __future__ import annotations

import argparse
import os
import time

import pygame
import torch

from . import config as CFG
from .brain import Brain, ExperienceBuffer, _WM_LOOKAHEAD_PERIOD
from .dream import CARD_BORDER, TEXT_MUTED, TEXT_WHITE
from .wm_theater import FONT_AA, FONT_NAME, WMTheater

STRIP_H = 74
LOSS_WINDOW = 240


class WMManager(WMTheater):
    """WMTheater + epoch-by-epoch WM training on the selected sequence."""

    def __init__(self, brain: Brain, buffer: ExperienceBuffer, wm_path: str = "wm_ckpt.pt",
                 phase_epochs: int = 512, phase_batch: int = 64,
                 phase_target: float | None = 0.01, **kwargs):
        super().__init__(brain, buffer, **kwargs)
        self.wm_path = wm_path
        self.training = False
        self.phase_gen = None           # générateur train_wm_phase actif (touche W)
        self.phase_epochs = phase_epochs
        self.phase_batch = phase_batch
        self.phase_target = phase_target
        self.epoch = 0
        self.losses: list[tuple[int, float, bool]] = []  # (epoch, loss, lookahead)
        self.force_amp = 0.5              # amplitude du forçage EOS (touches - / =)
        self.epf = 1                    # epochs per frame while training
        self.saved_flash: float | None = None
        self.footer_hints = ("Nav: [← / →]  |  [S] Sort  |  [L] Lookahead  |  [T] Train seq  |  "
                             "[W] Phase WM  |  [Space] 1 epoch  |  [R] Reset  |  [ / ] lr  |  "
                             "[↑/↓] epf  |  [V] Save WM  |  [Esc] Quit")
        self.footer_hints_la = self.footer_hints
        self.title_str = "WM MANAGER"

    # --- training -----------------------------------------------------------

    def train_one_epoch(self) -> float:
        """One WM optimization step on the selected sequence only, then drop the
        cached rollout so the next draw recomputes the inference."""
        if self.num_seqs == 0:
            return float("nan")
        seq = self.buffer._coreset[self.seq_idx]
        self.epoch += 1
        lookahead = (self.epoch % _WM_LOOKAHEAD_PERIOD == 0)
        self.brain.world.train()
        loss = self.brain._train_wm_batch([seq], lookahead=lookahead)
        self.losses.append((self.epoch, loss, lookahead))
        if len(self.losses) > 2000:
            self.losses = self.losses[-2000:]
        self._refresh_current_seq()
        return loss

    def start_phase(self) -> None:
        """Lance une phase WM équivalente au sleep (même générateur
        train_wm_phase), avec la séquence courante pré-sélectionnée dans
        chaque batch."""
        self.phase_gen = self.brain.train_wm_phase(
            self.phase_epochs, self.phase_batch, self.phase_target,
            source="coreset",
            fixed_seq=lambda: self.buffer._coreset[self.seq_idx],
            buffer=self.buffer, label="wm_phase")

    def _refresh_current_seq(self) -> None:
        self.cache.pop((self.seq_idx, False), None)
        self.cache.pop((self.seq_idx, True), None)

    def update(self) -> None:
        if self.training:
            for _ in range(self.epf):
                self.train_one_epoch()
        if self.phase_gen is not None:
            for _ in range(self.epf):
                try:
                    _, ep, loss, lookahead = next(self.phase_gen)
                except StopIteration:
                    self.phase_gen = None
                    break
                self.epoch += 1
                self.losses.append((self.epoch, loss, lookahead))
                if len(self.losses) > 2000:
                    self.losses = self.losses[-2000:]
                self._refresh_current_seq()

    def _scale_lr(self, factor: float) -> None:
        for g in self.brain.opt_wm.param_groups:
            g["lr"] = min(1e-2, max(1e-7, g["lr"] * factor))

    def save_wm(self) -> None:
        torch.save({"world": self.brain.world.state_dict(),
                    "opt_wm": self.brain.opt_wm.state_dict()}, self.wm_path)
        self.saved_flash = time.time()
        print(f"Saved World Model to {self.wm_path}")

    # --- events -------------------------------------------------------------

    def handle_event(self, ev: pygame.event.Event) -> bool:
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_t:
                self.training = not self.training
                if self.training:
                    self.brain.world.train()
                return True
            elif ev.key == pygame.K_w:
                if self.phase_gen is None:
                    self.start_phase()
                else:
                    self.phase_gen = None
                return True
            elif ev.key == pygame.K_SPACE:
                self.train_one_epoch()
                return True
            elif ev.key == pygame.K_r:
                self.epoch = 0
                self.losses.clear()
                return True
            elif ev.key == pygame.K_LEFTBRACKET:
                self._scale_lr(0.5)
                return True
            elif ev.key == pygame.K_RIGHTBRACKET:
                self._scale_lr(2.0)
                return True
            elif ev.key == pygame.K_UP:
                self.epf = min(50, self.epf + 1)
                return True
            elif ev.key == pygame.K_DOWN:
                self.epf = max(1, self.epf - 1)
                return True
            elif ev.key == pygame.K_v:
                self.save_wm()
                return True
            elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_COMMA):
                self.force_amp = max(0.05, round(self.force_amp - 0.05, 2))
                self.cache.clear()
                return True
            elif ev.key in (pygame.K_EQUALS, pygame.K_KP_PLUS, pygame.K_PERIOD):
                self.force_amp = min(1.0, round(self.force_amp + 0.05, 2))
                self.cache.clear()
                return True
        return super().handle_event(ev)

    # --- drawing ------------------------------------------------------------

    def _draw_train_strip(self, screen: pygame.Surface,
                          font_small: pygame.font.Font,
                          font_tiny: pygame.font.Font) -> None:
        y0 = self.height - 24 - 8 - STRIP_H
        rect = pygame.Rect(12, y0, self.width - 24, STRIP_H)
        pygame.draw.rect(screen, (20, 24, 34), rect, border_radius=4)
        active_c = (60, 200, 110) if self.training else CARD_BORDER
        pygame.draw.rect(screen, active_c, rect, 2 if self.training else 1, border_radius=4)

        x = rect.x + 10
        if self.phase_gen is not None:
            badge, badge_c = "PHASE WM (coreset + seq courante)", (255, 210, 90)
        elif self.training:
            badge, badge_c = "TRAINING", (100, 245, 160)
        else:
            badge, badge_c = "PAUSED", TEXT_MUTED
        screen.blit(font_small.render(badge, FONT_AA, badge_c), (x, y0 + 6))
        lr = self.brain.opt_wm.param_groups[0]["lr"]
        screen.blit(font_tiny.render(
            f"Epoch {self.epoch}   epf {self.epf}   lr {lr:.2e}   force_amp {self.force_amp:.2f} (-/=)", FONT_AA, TEXT_WHITE),
            (x, y0 + 24))
        if self.losses:
            _, loss, la = self.losses[-1]
            impact = getattr(self.brain, "last_wm_impact", float("nan"))
            c = (200, 150, 255) if la else (110, 200, 255)
            screen.blit(font_tiny.render(
                f"loss {loss:.5f} ({'lookahead' if la else 'causal'})   impact {impact:.3f}",
                FONT_AA, c), (x, y0 + 40))
        if self.saved_flash is not None and time.time() - self.saved_flash < 2.0:
            screen.blit(font_tiny.render(f"WM saved -> {self.wm_path}", FONT_AA, (255, 220, 90)),
                        (x, y0 + 56))

        # Loss sparkline (right side of the strip)
        sw, sh = 300, STRIP_H - 16
        sx, sy = rect.right - sw - 10, y0 + 8
        pygame.draw.rect(screen, (16, 19, 27), (sx, sy, sw, sh))
        pts = self.losses[-LOSS_WINDOW:]
        if len(pts) > 1:
            mx = max(p[1] for p in pts) or 1.0
            prev = None
            for i, (_, loss, la) in enumerate(pts):
                px = sx + 2 + i * (sw - 4) / (len(pts) - 1)
                py = sy + sh - 3 - min(1.0, loss / mx) * (sh - 6)
                c = (185, 110, 245) if la else (90, 200, 255)
                if prev is not None:
                    pygame.draw.line(screen, c, prev, (px, py), 1)
                prev = (px, py)
        screen.blit(font_tiny.render("loss history (blue=causal, purple=lookahead)", FONT_AA, TEXT_MUTED),
                    (sx, sy + sh + 1))

    def draw(self, screen: pygame.Surface, font: pygame.font.Font,
             font_small: pygame.font.Font, font_tiny: pygame.font.Font) -> None:
        super().draw(screen, font, font_small, font_tiny)
        self._draw_train_strip(screen, font_small, font_tiny)


def run_manager(buffer_path: str = "buf_ckpt.pt",
                wm_path: str = "wm_ckpt.pt",
                pol_path: str = "pol_ckpt.pt",
                initial_index: int = 0,
                device: str = "cpu",
                initial_sort: str = "relief_desc",
                d_model: int = 64, nhead: int = 4, wm_layers: int = 3,
                dim_ff: int = 384, wm_dropout: float = 0.0,
                pol_hidden: int = 256,
                lr_wm: float = 3e-4, lr_pol: float = 2e-4,
                act_std: float = 0.0,
                wm_batch: int = 64,
                wm_coreset_epoch: int = 512,
                wm_target_loss: float = 0.01,
                seed: int = 0,
                buffer_capacity: int = 1000, buffer_pool_capacity: int = 1000,
                seq_steps: int = 26, n_ctx: int = 4,
                compile_wm: bool = False,
                eos_force_amp_lo: float = 0.1, eos_force_amp_hi: float = 0.5,
                eos_force_amplify: float = 1.0,
                wm_impact_floor: float = 0.25,
                lookahead_relief_mult: float = 1.0,
                wm_w_a4_lookahead: float = 1.0,
                eos_decay: float = 0.9, eos_poids_chemin: float = 0.1, eos_max_p: float = 8.0,
                **_ignored) -> None:
    os.environ["SDL_HINT_RENDER_SCALE_QUALITY"] = "1"
    try:
        import ctypes
        ctypes.CDLL("libSDL2-2.0.so.0").SDL_SetHint(b"SDL_HINT_RENDER_SCALE_QUALITY", b"1")
    except Exception:
        pass
    pygame.init()
    pygame.display.set_caption("WM Manager — Single-Sequence Training Lab")

    width, height = 1600, 730
    flags = pygame.SCALED | pygame.RESIZABLE
    screen = pygame.display.set_mode((width, height), flags)
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(FONT_NAME, 14, bold=True)
    font_small = pygame.font.SysFont(FONT_NAME, 11)
    font_tiny = pygame.font.SysFont(FONT_NAME, 10)

    brain = Brain(device=device, compile_wm=compile_wm, seq_steps=seq_steps, n_ctx=n_ctx,
                  d_model=d_model, nhead=nhead, layers=wm_layers,
                  dim_ff=dim_ff, dropout=wm_dropout,
                  pol_hidden=pol_hidden,
                  lr_wm=lr_wm, lr_pol=lr_pol,
                  act_std=act_std,
                  seed=seed,
                  buffer_capacity=buffer_capacity,
                  buffer_pool_capacity=buffer_pool_capacity,
                  eos_force_amp_lo=eos_force_amp_lo,
                  eos_force_amp_hi=eos_force_amp_hi,
                  eos_force_amplify=eos_force_amplify,
                  wm_impact_floor=wm_impact_floor,
                  lookahead_relief_mult=lookahead_relief_mult,
                  wm_w_a4_lookahead=wm_w_a4_lookahead,
                  eos_decay=eos_decay, eos_poids_chemin=eos_poids_chemin,
                  eos_max_p=eos_max_p)
    if os.path.exists(wm_path):
        brain.load_checkpoint(
            wm_path=wm_path,
            pol_path=pol_path if os.path.exists(pol_path) else "nonexistent.pt",
        )
        print(f"Loaded World Model from {wm_path}")
    else:
        print("Warning: World Model checkpoint not found, using initialized weights.")

    buf = ExperienceBuffer(capacity=buffer_capacity, pool_capacity=buffer_pool_capacity,
                           seq_len=seq_steps + 1, n_ctx=n_ctx, seed=seed,
                           eos_decay=eos_decay, eos_poids_chemin=eos_poids_chemin)
    if os.path.exists(buffer_path):
        buf.load(buffer_path)
        print(f"Loaded {len(buf._coreset)} sequences from {buffer_path}")
    else:
        print(f"Warning: Buffer checkpoint {buffer_path} not found.")

    manager = WMManager(brain, buf, wm_path=wm_path, width=width, height=height,
                        initial_sort=initial_sort,
                        phase_epochs=wm_coreset_epoch, phase_batch=wm_batch,
                        phase_target=wm_target_loss)
    manager.set_seq_idx(initial_index)

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
                break
            elif ev.type == pygame.KEYDOWN and ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and (ev.mod & pygame.KMOD_ALT or pygame.key.get_mods() & pygame.KMOD_ALT):
                pygame.display.toggle_fullscreen()
                continue
            elif ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
                break
            manager.handle_event(ev)

        manager.update()
        manager.draw(screen, font, font_small, font_tiny)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def main() -> None:
    cfg = CFG.load_config()

    params: dict = {
        "buffer_path": "buf_ckpt.pt",
        "wm_path": "wm_ckpt.pt",
        "pol_path": "pol_ckpt.pt",
        "initial_index": 0,
        "device": os.environ.get("LAMBDA_DEVICE", "cpu"),
        "initial_sort": "relief_desc",
        "accel": False,
        "d_model": 64, "nhead": 4, "wm_layers": 3,
        "dim_ff": 384, "wm_dropout": 0.0,
        "pol_hidden": 256,
        "lr_wm": 3e-4, "lr_pol": 2e-4,
        "act_std": 0.0,
        "wm_batch": 64,
        "wm_coreset_epoch": 512,
        "wm_target_loss": 0.01,
        "seed": 0,
        "buffer_capacity": 1000, "buffer_pool_capacity": 1000,
        "seq_steps": 26, "n_ctx": 4,
        "compile_wm": False,
        "eos_force_amp_lo": 0.015, "eos_force_amp_hi": 0.0375,
        "eos_force_amplify": 1.0,
        "wm_impact_floor": 0.25,
        "lookahead_relief_mult": 1.0,
        "wm_w_a4_lookahead": 1.0,
        "eos_decay": 0.9, "eos_poids_chemin": 0.1, "eos_max_p": 8.0,
    }
    CFG.apply_toml(params, cfg)

    parser = argparse.ArgumentParser(description="WM Manager GUI — single-sequence training lab")
    parser.add_argument("--buffer", type=str, default=argparse.SUPPRESS, dest="buffer_path",
                        help="Path to ExperienceBuffer checkpoint (default: buf_ckpt.pt)")
    parser.add_argument("--wm", type=str, default=argparse.SUPPRESS, dest="wm_path",
                        help="Path to WorldModel checkpoint (default: wm_ckpt.pt)")
    parser.add_argument("--pol", type=str, default=argparse.SUPPRESS, dest="pol_path",
                        help="Path to Policy checkpoint (default: pol_ckpt.pt)")
    parser.add_argument("--index", type=int, default=argparse.SUPPRESS, dest="initial_index",
                        help="Initial sequence index to display (default: 0)")
    parser.add_argument("--device", type=str, default=argparse.SUPPRESS,
                        choices=["cpu", "cuda", "auto", "cpu-fast"],
                        help="Torch device (default: cpu)")
    parser.add_argument("--sort", type=str, default=argparse.SUPPRESS, dest="initial_sort",
                        choices=["relief_desc", "relief_asc", "abs_desc", "coreset"],
                        help="Sort mode (default: relief_desc)")
    parser.add_argument("--lr-wm", type=float, default=argparse.SUPPRESS,
                        help="World Model learning rate")
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="random seed")
    parser.add_argument("--compile", action="store_true", default=argparse.SUPPRESS,
                        dest="compile_wm", help="torch.compile the World Model training step")

    args = parser.parse_args()
    CFG.merge_cli(params, vars(args))

    # Resolve device: accel (config.toml) overrides device
    accel = params.pop("accel", False)
    if accel:
        params["device"] = "auto"

    run_manager(**params)


if __name__ == "__main__":
    main()
