"""WM Manager (refonte): surfer sur le Coreset, afficher l'enregistrement et
les deux générations du WM (dénouement le plus probable µ / dénouement élu),
plus une boucle d'entraînement live sur la séquence sélectionnée.

- 3 rangées comme WM Theater : RÉEL / WM µ / WM ÉLU.
- [T] toggles training: chaque epoch = un step WM sur la séquence seule, les
  rangées sont recalculées (inférence visible après chaque epoch).
- [Space] une epoch, [W] phase WM amplifiée (séquence courante dans chaque
  batch), [S] phase WM complète (tout le coreset, batchs normaux),
  [R] reset historique, [ / ] lr, [↑/↓] epochs/frame, [V] save.

Usage:
    python -m lambda_barre.wm_manager
    python -m lambda_barre.wm_manager --wm wm_ckpt.pt --buffer buf_ckpt.pt --index 0
"""
from __future__ import annotations

import argparse
import math
import os
import time

import pygame
import torch

from . import config as CFG
from .brain import Brain, ExperienceBuffer
from .dream import CARD_BORDER, TEXT_MUTED, TEXT_WHITE
from .wm_theater import FONT_AA, FONT_NAME, WMTheater

STRIP_H = 64
LOSS_WINDOW = 240


class WMManager(WMTheater):
    """WMTheater + epoch-by-epoch WM training on the selected sequence."""

    def __init__(self, brain: Brain, buffer: ExperienceBuffer, wm_path: str = "wm_ckpt.pt",
                 phase_epochs: int = 512, phase_batch: int = 64,
                 phase_target: float | None = 0.01, ctx_eps: float = 1e-3, **kwargs):
        super().__init__(brain, buffer, **kwargs)
        self.ctx_eps = ctx_eps
        self.bottom_margin = 24 + 8 + STRIP_H   # footer + strip d'entraînement
        self.wm_path = wm_path
        self.training = False
        self.phase_gen = None
        self._phase_amplified = False
        self.phase_epochs = phase_epochs
        self.phase_batch = phase_batch
        self.phase_target = phase_target
        self.epoch = 0
        self.losses: list[tuple[int, float]] = []
        self.epf = 1
        self.saved_flash: float | None = None
        self.footer_hints = ("[←/→] Nav | [F]ilter | [O]rder | "
                             "[T]/[W]/[S] Train seq/++/all | "
                             "[Space] epoch | [R]eset | [/] lr | "
                             "[↑/↓] epf | [V] Save | [Esc] Quit")
        self.title_str = "WM MANAGER"

    # --- training -----------------------------------------------------------
    def train_one_epoch(self) -> float:
        """Un step WM sur la séquence sélectionnée seule, puis invalide le
        cache du rollout pour que le prochain draw recalcule l'inférence."""
        if self.num_seqs == 0:
            return float("nan")
        seq = self.buffer._coreset[self.seq_idx]
        self.epoch += 1
        self.brain.world.train()
        loss = self.brain._train_wm_batch([seq])
        self.losses.append((self.epoch, loss))
        if len(self.losses) > 2000:
            self.losses = self.losses[-2000:]
        self.cache.pop(self.seq_idx, None)
        return loss

    def start_phase(self, amplified: bool = True) -> None:
        """Phase WM complète (train_wm_phase sur le coreset). ``amplified=True``
        ([W]) : la séquence courante est pré-sélectionnée dans chaque batch ;
        ``amplified=False`` ([S]) : batchs normaux sur tout le coreset."""
        self._phase_amplified = amplified
        self.phase_gen = self.brain.train_wm_phase(
            self.phase_epochs, self.phase_batch, self.phase_target,
            source="coreset",
            fixed_seq=(lambda: self.buffer._coreset[self.seq_idx]
                       if amplified and self.buffer._coreset else None),
            label="wm_phase+seq" if amplified else "wm_phase")

    def update(self) -> None:
        super().update()   # debounce de la navigation (glider / flèches)
        if self.training:
            for _ in range(self.epf):
                self.train_one_epoch()
        if self.phase_gen is not None:
            for _ in range(self.epf):
                try:
                    _, ep, loss = next(self.phase_gen)
                except StopIteration:
                    self.phase_gen = None
                    break
                self.epoch += 1
                self.losses.append((self.epoch, loss))
                if len(self.losses) > 2000:
                    self.losses = self.losses[-2000:]
                self.cache.pop(self.seq_idx, None)

    def _scale_lr(self, factor: float) -> None:
        for g in self.brain.opt_wm.param_groups:
            g["lr"] = min(1e-2, max(1e-7, g["lr"] * factor))

    def save_wm(self) -> None:
        wm_loss = (self.brain.last_batch_mse
                   if not math.isnan(self.brain.last_batch_mse)
                   else self.brain.last_wm_loss)
        torch.save({"world": self.brain.world.state_dict(),
                    "opt_wm": self.brain.opt_wm.state_dict(),
                    "last_wm_loss": wm_loss}, self.wm_path)
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
                    self.start_phase(amplified=True)
                else:
                    self.phase_gen = None
                return True
            elif ev.key == pygame.K_s:
                # phase WM sur tout le coreset (sans la séquence courante forcée)
                if self.phase_gen is None:
                    self.start_phase(amplified=False)
                else:
                    self.phase_gen = None
                return True
            elif ev.key == pygame.K_o:
                self.cycle_sort()
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
            elif ev.key == pygame.K_f:
                # filtre par similarité de contexte : la séquence courante devient
                # référence ; re-appui = reset
                if self.ctx_filter_on:
                    self.clear_ctx_reference()
                    print("[ctx-filter] reset — all coreset sequences visible")
                else:
                    n = self.set_ctx_reference(self.seq_idx)
                    print(f"[ctx-filter] ref=idx {self._ref_idx} kept={n}/{self.num_seqs} "
                          f"eps={self.ctx_eps:g}")
                    try:
                        os.makedirs(".tmp", exist_ok=True)
                        with open(".tmp/ctx_filter.txt", "w") as fh:
                            fh.write("\n".join(map(str, self._filtered_idx or [])))
                    except OSError:
                        pass
                return True
        return super().handle_event(ev)

    # --- drawing ------------------------------------------------------------
    def _draw_train_strip(self, screen, font_small, font_tiny) -> None:
        y0 = self.height - 24 - 8 - STRIP_H
        rect = pygame.Rect(12, y0, self.width - 24, STRIP_H)
        pygame.draw.rect(screen, (20, 24, 34), rect, border_radius=4)
        active_c = (60, 200, 110) if self.training else CARD_BORDER
        pygame.draw.rect(screen, active_c, rect, 2 if self.training else 1, border_radius=4)
        x = rect.x + 10
        if self.phase_gen is not None:
            badge = ("WM PHASE (coreset + current seq)" if self._phase_amplified
                     else "WM PHASE (full coreset)")
            if getattr(self.brain, "sleep_wm_slingshot", False):
                badge += " [SLINGSHOT]"
            badge_c = (255, 160, 80) if getattr(self.brain, "sleep_wm_slingshot", False) else (255, 210, 90)
        elif self.training:
            badge, badge_c = "TRAINING", (100, 245, 160)
        else:
            badge, badge_c = "PAUSED", TEXT_MUTED
        screen.blit(font_small.render(badge, FONT_AA, badge_c), (x, y0 + 6))
        lr = self.brain.opt_wm.param_groups[0]["lr"]
        screen.blit(font_tiny.render(
            f"Epoch {self.epoch}   epf {self.epf}   lr {lr:.2e}", FONT_AA, TEXT_WHITE),
            (x, y0 + 22))
        if self.losses:
            _, loss = self.losses[-1]
            nll = getattr(self.brain, "last_batch_nll", float("nan"))
            txt = (f"mse {loss:.5f}   nll {nll:.3f}" if not math.isnan(nll)
                   else f"mse {loss:.5f}")
            screen.blit(font_tiny.render(txt, FONT_AA, (110, 200, 255)),
                        (x, y0 + 38))
        if self.saved_flash is not None and time.time() - self.saved_flash < 2.0:
            screen.blit(font_tiny.render(f"WM saved -> {self.wm_path}", FONT_AA, (255, 220, 90)),
                        (x, y0 + 52))
        # sparkline
        sw, sh = 300, STRIP_H - 14
        sx, sy = rect.right - sw - 10, y0 + 7
        pygame.draw.rect(screen, (16, 19, 27), (sx, sy, sw, sh))
        pts = self.losses[-LOSS_WINDOW:]
        if len(pts) > 1:
            mx = max(p[1] for p in pts) or 1.0
            prev = None
            for i, (_, loss) in enumerate(pts):
                px = sx + 2 + i * (sw - 4) / (len(pts) - 1)
                py = sy + sh - 3 - min(1.0, loss / mx) * (sh - 6)
                if prev is not None:
                    pygame.draw.line(screen, (90, 200, 255), prev, (px, py), 1)
                prev = (px, py)
        screen.blit(font_tiny.render("loss history", FONT_AA, TEXT_MUTED), (sx, sy + sh + 1))

    def draw(self, screen, font, font_small, font_tiny=None) -> None:
        if font_tiny is None:
            font_tiny = font_small
        super().draw(screen, font, font_small, font_tiny)
        self._draw_train_strip(screen, font_small, font_tiny)


def run_manager(buffer_path: str = "buf_ckpt.pt", wm_path: str = "wm_ckpt.pt",
                pol_path: str = "pol_ckpt.pt", initial_index: int = 0,
                initial_sort: str = "coreset", ctx_eps: float = 1e-3,
                wm_batch: int = 64, wm_coreset_epoch: int = 512,
                wm_target_loss: float = 0.004,
                **params) -> None:
    os.environ["SDL_HINT_RENDER_SCALE_QUALITY"] = "1"
    pygame.init()
    pygame.display.set_caption("WM Manager — refonte")
    width, height = 960, 520
    screen = pygame.display.set_mode((width, height), pygame.SCALED | pygame.RESIZABLE)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(FONT_NAME, 14, bold=True)
    font_small = pygame.font.SysFont(FONT_NAME, 11)
    font_tiny = pygame.font.SysFont(FONT_NAME, 10)

    brain = Brain(**CFG.brain_kwargs(params))
    if os.path.exists(wm_path):
        if os.path.exists(pol_path):
            brain.load_checkpoint(wm_path=wm_path, pol_path=pol_path)
        else:
            wm = torch.load(wm_path, weights_only=False)
            exact = brain.load_world_state(wm["world"])
            if exact and "opt_wm" in wm:
                brain.opt_wm.load_state_dict(wm["opt_wm"])
            if exact and "last_wm_loss" in wm:
                brain.last_wm_loss = float(wm["last_wm_loss"])
        print(f"Loaded World Model from {wm_path}")
    else:
        print("Warning: WM checkpoint not found, using initialized weights.")

    if os.path.exists(buffer_path):
        ok = brain.buffer.load(buffer_path)
        print(f"Loaded {brain.buffer.coreset_size} sequences from {buffer_path}" if ok
              else f"{buffer_path}: layout incompatible (legacy ?) — coreset vide")
    else:
        print(f"Warning: Buffer checkpoint {buffer_path} not found.")

    manager = WMManager(brain, brain.buffer, wm_path=wm_path, width=width, height=height,
                        initial_sort=initial_sort, ctx_eps=ctx_eps,
                        phase_epochs=wm_coreset_epoch, phase_batch=wm_batch,
                        phase_target=wm_target_loss)
    manager.set_seq_idx(initial_index)

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
                break
            elif ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
                break
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_RETURN and (
                    ev.mod & pygame.KMOD_ALT or pygame.key.get_mods() & pygame.KMOD_ALT):
                pygame.display.toggle_fullscreen()
                continue
            manager.handle_event(ev)
        manager.update()
        manager.draw(screen, font, font_small, font_tiny)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


def main() -> None:
    cfg = CFG.load_config()
    params: dict = {
        **CFG.BRAIN_DEFAULTS,
        "buffer_path": "buf_ckpt.pt", "wm_path": "wm_ckpt.pt", "pol_path": "pol_ckpt.pt",
        "initial_index": 0, "initial_sort": "coreset", "ctx_eps": 1e-3,
    }
    CFG.apply_toml(params, cfg)
    p = argparse.ArgumentParser(description="WM Manager — coreset browser + training (refonte)")
    p.add_argument("--buffer", type=str, default=argparse.SUPPRESS, dest="buffer_path")
    p.add_argument("--wm", type=str, default=argparse.SUPPRESS, dest="wm_path")
    p.add_argument("--pol", type=str, default=argparse.SUPPRESS, dest="pol_path")
    p.add_argument("--index", type=int, default=argparse.SUPPRESS, dest="initial_index")
    p.add_argument("--device", type=str, default=argparse.SUPPRESS,
                   choices=["cpu", "cuda", "auto", "cpu-fast"])
    p.add_argument("--sort", type=str, default=argparse.SUPPRESS, dest="initial_sort",
                   choices=["coreset", "relief_desc", "relief_asc", "abs_desc"])
    p.add_argument("--ctx-eps", type=float, default=argparse.SUPPRESS, dest="ctx_eps",
                   help="seuil MSE sur les signaux du contexte pour le filtre [F]")
    p.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    p.add_argument("--compile", action="store_true", default=argparse.SUPPRESS, dest="compile_wm")
    CFG.merge_cli(params, vars(p.parse_args()))
    accel = params.pop("accel", False)
    if accel:
        params["device"] = "auto"
    run_manager(**params)


if __name__ == "__main__":
    main()
