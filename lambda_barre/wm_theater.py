"""WM Theater (refonte) — surfer sur le Coreset et comparer l'enregistrement
aux générations du World Model.

Pour chaque séquence du Coreset, trois rangées :
- RÉEL : la fenêtre temporelle reconstruite (s0..s10) ;
- WM µ : génération depuis [contexte, dénouement le plus probable] (moyenne
  conditionnelle prédite par le WM) ;
- WM ÉLU : génération depuis [contexte, dénouement élu] (échantillonnage
  top-K/top-p + élection f(probable, favorable) = argmax log p + w_fav·relief).

Usage:
    python -m lambda_barre.wm_theater
    python -m lambda_barre.wm_theater --buffer buf_ckpt.pt --wm wm_ckpt.pt --index 0
"""
from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass, field

import pygame
import torch

from . import body as B
from . import config as CFG
from .brain import Brain, ExperienceBuffer, relief_score, sequence_to_window
from .dream import (
    CARD_BG,
    CARD_BORDER,
    CAND_COLORS,
    DREAM_BG,
    HEADER_BG,
    TEXT_ACCENT,
    TEXT_MUTED,
    TEXT_WHITE,
    WINNER_BORDER,
    DreamStep,
    decode_state_tokens,
    draw_thumbnail,
)
from .tokenize import SIG_OFFSET, STATE_TOKENS, salve_cost

COST_COLORS = [
    (245, 140, 50),   # EF
    (235, 75, 75),    # DO
    (185, 110, 245),  # CO
    (240, 220, 60),   # IN
    (60, 215, 215),   # VE
]
COST_LABELS = ["EF", "DO", "CO", "IN", "VE"]

ROW_COLORS = {
    "real": (200, 200, 210),
    "argmax": (110, 200, 255),
    "elected": (255, 210, 70),
}
ROW_LABELS = {
    "real": "RÉEL",
    "argmax": "WM µ (dénouement le plus probable)",
    "elected": "WM ÉLU (dénouement optimal)",
}

FONT_NAME = "dejavusansmono,ubuntumono,liberationmono,monospace"
FONT_AA = False

SORT_MODES = ["coreset", "relief_desc", "relief_asc", "abs_desc"]


@dataclass
class WMTheaterRecord:
    seq_idx: int
    total_seqs: int
    vivacity: float
    real_steps: list[DreamStep]
    argmax_steps: list[DreamStep]
    elected_steps: list[DreamStep]
    eos_observed: list[float]
    eos_argmax: list[float]
    eos_elected: list[float]
    p_elected: float
    real_relief: float
    argmax_relief: float
    elected_relief: float
    cos_argmax: float | None
    cos_elected: float | None
    rmse_posture_elected: float
    rmse_cost_elected: float


def compute_posture_rmse(real_toks, pred_toks) -> float:
    real_info = decode_state_tokens(real_toks, facing=1)
    pred_info = decode_state_tokens(pred_toks, facing=1)
    diffs = [
        real_info["tronc_angle"] - pred_info["tronc_angle"],
        real_info["tail_angle"] - pred_info["tail_angle"],
        real_info["angle_f"] - pred_info["angle_f"],
        (real_info["dist_f"] - pred_info["dist_f"]) / B.LIMB_MAX,
        real_info["angle_b"] - pred_info["angle_b"],
        (real_info["dist_b"] - pred_info["dist_b"]) / B.LIMB_MAX,
    ]
    return math.sqrt(sum(d * d for d in diffs) / len(diffs))


def compute_cost_rmse(real_costs, pred_costs) -> float:
    diffs = [r - p for r, p in zip(real_costs, pred_costs)]
    return math.sqrt(sum(d * d for d in diffs) / len(diffs))


def _cos(a, b) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _salve_costs(s_toks) -> list[float]:
    return [float(c) for c in s_toks[10][SIG_OFFSET:SIG_OFFSET + 5]]


def _gen_to_steps(brain, gen, real_steps) -> tuple[list[DreamStep], list[float]]:
    """Transforme une génération [contexte, dénouement] en rangée temporelle
    s0..s_{seq_steps} : contexte réel, a* sur s_{n_ctx}, états générés
    s_{n_ctx+1}..s_{seq_steps} (conséquence inversée réordonnée)."""
    conseq = gen["conseq"][0]                                   # [conseq_len, 25]
    a_star = brain._decode_action_batch(gen["action"])[0].tolist()
    n_im = brain.n_imagine
    blocks = [conseq[j * STATE_TOKENS:(j + 1) * STATE_TOKENS].tolist()
              for j in range(n_im)]                             # s_{seq_steps}..s_{n_ctx+1}
    blocks.reverse()                                            # ordre temporel s_{n_ctx+1}..s_{seq_steps}
    steps: list[DreamStep] = []
    for i in range(brain.n_ctx + 1):
        if i == brain.n_ctx:
            steps.append(DreamStep(state_tokens=real_steps[i].state_tokens,
                                   action=a_star, step_cost=real_steps[i].step_cost,
                                   label=f"s{i}+a*"))
        else:
            steps.append(real_steps[i])
    for j, s_toks in enumerate(blocks):
        steps.append(DreamStep(state_tokens=s_toks, action=None,
                               step_cost=salve_cost(s_toks) if len(s_toks) >= 13 else 0.0,
                               label=f"s{brain.n_ctx + 1 + j}^"))
    return steps, a_star


@torch.inference_mode()
def rollout_record(brain: Brain, seq, seq_idx: int, total_seqs: int, vivacity: float) -> WMTheaterRecord:
    """Construit les 3 rangées d'une séquence : réel, WM-µ, WM-élu."""
    device = brain.device
    window = sequence_to_window(seq, n_ctx=brain.n_ctx, seq_steps=brain.seq_steps)

    # rangée réelle
    real_steps: list[DreamStep] = []
    for i, salve in enumerate(window):
        a = None
        if i < len(window) - 1:
            a_toks = torch.tensor([salve[STATE_TOKENS:]], dtype=torch.float32, device=device)
            a_vals = brain._decode_action_batch(a_toks)[0].tolist()
            if any(abs(v) > 1e-9 for v in a_vals):
                a = a_vals
        real_steps.append(DreamStep(state_tokens=[list(t) for t in salve[:STATE_TOKENS]],
                                    action=a, step_cost=salve_cost(salve),
                                    label=f"s{i}" + ("a" if a else "")))

    vals = brain._batch_tensors([seq])
    ctx = vals[:, :brain.ctx_len]
    eos_obs = vals[:, brain.eos_pos, SIG_OFFSET:SIG_OFFSET + 6]

    mu = brain.predict_denouement(ctx)                          # dénouement le plus probable
    gen_mu = brain.generate_from_denouement(ctx, mu)
    eos_elu, p_elu = brain.sample_denouements(ctx, eos_observed=eos_obs)  # dénouement élu
    gen_el = brain.generate_from_denouement(ctx, eos_elu)

    argmax_steps, a_mu = _gen_to_steps(brain, gen_mu, real_steps)
    elected_steps, a_el = _gen_to_steps(brain, gen_el, real_steps)

    a4_real = brain._decode_action_batch(
        vals[:, brain.action_pos:brain.action_pos + 3, :])[0].tolist()

    # métriques sur la conséquence (s_{n_ctx+1}..s_{seq_steps})
    p_errs, c_errs = [], []
    for k in range(brain.n_ctx + 1, brain.seq_steps + 1):
        p_rmse = compute_posture_rmse(real_steps[k].state_tokens, elected_steps[k].state_tokens)
        c_rmse = compute_cost_rmse(_salve_costs(real_steps[k].state_tokens),
                                   _salve_costs(elected_steps[k].state_tokens))
        p_errs.append(p_rmse)
        c_errs.append(c_rmse)

    return WMTheaterRecord(
        seq_idx=seq_idx,
        total_seqs=total_seqs,
        vivacity=vivacity,
        real_steps=real_steps,
        argmax_steps=argmax_steps,
        elected_steps=elected_steps,
        eos_observed=eos_obs[0].tolist(),
        eos_argmax=mu[0].tolist(),
        eos_elected=eos_elu[0].tolist(),
        p_elected=float(p_elu[0]),
        real_relief=float(relief_score(seq, eos_pos=brain.eos_pos)),
        argmax_relief=float(relief_score(mu, p=brain.eos_max_p)[0]),
        elected_relief=float(relief_score(eos_elu, p=brain.eos_max_p)[0]),
        cos_argmax=_cos(a_mu, a4_real),
        cos_elected=_cos(a_el, a4_real),
        rmse_posture_elected=sum(p_errs) / max(1, len(p_errs)),
        rmse_cost_elected=sum(c_errs) / max(1, len(c_errs)),
    )


class WMTheater:
    """Coreset browser: 3 rows per sequence (real / WM argmax / WM elected)."""

    def __init__(self, brain: Brain, buffer: ExperienceBuffer,
                 width: int = 1120, height: int = 600,
                 initial_sort: str = "relief_desc"):
        self.brain = brain
        self.buffer = buffer
        self.width = width
        self.height = height
        self.sort_mode = initial_sort if initial_sort in SORT_MODES else "coreset"
        self.sorted_idx = 0
        self.cache: dict[int, WMTheaterRecord] = {}
        self.footer_hints = "Nav: [← / →]  |  [S] Sort  |  [R] Random  |  drag: scroll mignatures  |  [Esc] Quit"
        self.title_str = "WM THEATER"
        # géométrie du container scrollable des mignatures
        self.thumb_w, self.thumb_h, self.gap = 76, 56, 4
        self.header_h = 62
        self.bottom_margin = 24          # footer ; le manager ajoute sa strip
        self.scroll_x = 0
        self.scroll_y = 0
        self._drag: tuple | None = None  # (mouse_pos, scroll au départ du drag)
        self._update_sorted_indices()

    @property
    def _container_rect(self) -> pygame.Rect:
        """Zone scrollable (mignatures + lignes EOS) : sous le header, au-dessus
        du footer (et de la strip du manager)."""
        top = self.header_h
        bottom = self.height - self.bottom_margin
        return pygame.Rect(0, top, self.width, max(0, bottom - top))

    def _content_size(self, rec: WMTheaterRecord | None) -> tuple[int, int]:
        if rec is None:
            return (0, 0)
        n = len(rec.real_steps)
        w = 16 + n * (self.thumb_w + self.gap)
        h = 3 * (16 + self.thumb_h + 8) + 4 + 3 * 14   # 3 rows + marge + lignes EOS
        return (w, h)

    def _clamp_scroll(self, rec: WMTheaterRecord | None) -> None:
        cw, ch = self._content_size(rec)
        cont = self._container_rect
        self.scroll_x = max(0, min(self.scroll_x, max(0, cw - cont.w)))
        self.scroll_y = max(0, min(self.scroll_y, max(0, ch - cont.h)))

    @property
    def num_seqs(self) -> int:
        return self.buffer.coreset_size

    @property
    def seq_idx(self) -> int:
        if not self.order:
            return 0
        return self.order[self.sorted_idx % len(self.order)]

    def _seq_relief(self, i: int) -> float:
        return float(relief_score(self.buffer._coreset[i], eos_pos=self.brain.eos_pos))

    def _update_sorted_indices(self) -> None:
        n = self.num_seqs
        if self.sort_mode == "coreset":
            self.order = list(range(n))
        else:
            rels = [self._seq_relief(i) for i in range(n)]
            if self.sort_mode == "relief_desc":
                self.order = sorted(range(n), key=lambda i: -rels[i])
            elif self.sort_mode == "relief_asc":
                self.order = sorted(range(n), key=lambda i: rels[i])
            else:  # abs_desc
                self.order = sorted(range(n), key=lambda i: -abs(rels[i]))
        self.sorted_idx = min(self.sorted_idx, max(0, len(self.order) - 1))

    def set_sort(self, mode: str) -> None:
        self.sort_mode = mode
        self.sorted_idx = 0
        self._update_sorted_indices()

    def cycle_sort(self) -> None:
        i = SORT_MODES.index(self.sort_mode)
        self.set_sort(SORT_MODES[(i + 1) % len(SORT_MODES)])

    def set_seq_idx(self, idx: int) -> None:
        if not self.order:
            return
        idx = max(0, min(idx, len(self.order) - 1))
        self.sorted_idx = idx

    def set_sorted_idx(self, idx: int) -> None:
        self.set_seq_idx(idx)

    def prev_seq(self) -> None:
        self.set_seq_idx(self.sorted_idx - 1)

    def next_seq(self) -> None:
        self.set_seq_idx(self.sorted_idx + 1)

    def random_seq(self) -> None:
        if self.order:
            self.sorted_idx = random.randrange(len(self.order))

    def get_record(self, idx: int | None = None) -> WMTheaterRecord | None:
        if self.num_seqs == 0:
            return None
        i = self.seq_idx if idx is None else idx
        if i not in self.cache:
            viv = (self.buffer._coreset_vivacity[i]
                   if i < len(self.buffer._coreset_vivacity) else 1.0)
            self.cache[i] = rollout_record(self.brain, self.buffer._coreset[i],
                                           i, self.num_seqs, viv)
            if len(self.cache) > 32:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[i]

    def invalidate_cache(self) -> None:
        self.cache.clear()

    def handle_event(self, ev: pygame.event.Event) -> bool:
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_LEFT:
                self.prev_seq()
                return True
            elif ev.key == pygame.K_RIGHT:
                self.next_seq()
                return True
            elif ev.key == pygame.K_s:
                self.cycle_sort()
                return True
            elif ev.key == pygame.K_r:
                self.random_seq()
                return True
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self._container_rect.collidepoint(ev.pos):
                self._drag = ev.pos
                return True
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self._drag = None
        elif ev.type == pygame.MOUSEMOTION and self._drag is not None:
            self._apply_scroll(ev.pos[0] - self._drag[0], ev.pos[1] - self._drag[1])
            self._drag = ev.pos
            return True
        elif ev.type == pygame.MOUSEWHEEL:
            pos = getattr(ev, "pos", None) or pygame.mouse.get_pos()
            if self._container_rect.collidepoint(pos):
                self._apply_scroll(-ev.x * 40, ev.y * 40)
                return True
        return False

    def _apply_scroll(self, dx: float, dy: float) -> None:
        """Applique un delta de scroll (convention drag naturel : scroll -= delta
        souris). Scroll horizontal prioritaire : si le contenu déborde en
        largeur mais pas en hauteur, le mouvement vertical (trackpad deux
        doigts, molette) est mappé sur l'horizontal."""
        rec = self.get_record()
        cont = self._container_rect
        cw, ch = self._content_size(rec)
        h_overflow = cw - cont.w > 0
        v_overflow = ch - cont.h > 0
        if h_overflow and not v_overflow:
            self.scroll_x -= dx + dy
        else:
            self.scroll_x -= dx
            self.scroll_y -= dy
        self._clamp_scroll(rec)

    # --- drawing ------------------------------------------------------------
    def _draw_row(self, screen, rec: WMTheaterRecord, key: str, x0: int, y0: int,
                  font_small, font_tiny) -> int:
        """Dessine une rangée de mignatures à partir de (x0, y0) — coordonnées
        déjà translatées du scroll par l'appelant. Retourne le y de fin."""
        color = ROW_COLORS[key]
        steps = {"real": rec.real_steps, "argmax": rec.argmax_steps,
                 "elected": rec.elected_steps}[key]
        screen.blit(font_small.render(ROW_LABELS[key], FONT_AA, color), (x0, y0))
        yy = y0 + 16
        x = x0
        for st in steps:
            rect = pygame.Rect(x, yy, self.thumb_w, self.thumb_h)
            draw_thumbnail(screen, rect, st, font_tiny, facing=1, cand_color=color)
            x += self.thumb_w + self.gap
        return yy + self.thumb_h + 8

    def _draw_header(self, screen, rec: WMTheaterRecord | None, font, font_small) -> int:
        pygame.draw.rect(screen, HEADER_BG, (0, 0, self.width, self.header_h - 6))
        title = f"{self.title_str}  |  seq {self.sorted_idx + 1}/{len(self.order)} (idx {rec.seq_idx if rec else '-'})  |  sort {self.sort_mode}"
        screen.blit(font.render(title, FONT_AA, TEXT_WHITE), (16, 8))
        if rec is not None:
            info = (f"relief reel {rec.real_relief:+.3f} | mu {rec.argmax_relief:+.3f} | "
                    f"elu {rec.elected_relief:+.3f} (p={rec.p_elected:.2f}) | "
                    f"cos(a*,a4) mu {rec.cos_argmax:+.2f} elu {rec.cos_elected:+.2f} | "
                    f"rmse posture {rec.rmse_posture_elected:.3f} cout {rec.rmse_cost_elected:.3f}")
            screen.blit(font_small.render(info, FONT_AA, TEXT_ACCENT), (16, 32))
        return self.header_h

    def draw(self, screen, font, font_small, font_tiny=None) -> None:
        if font_tiny is None:
            font_tiny = font_small
        screen.fill(DREAM_BG)
        rec = self.get_record()
        self._draw_header(screen, rec, font, font_small)
        # container scrollable des mignatures (drag / molette), dessiné avec clip
        cont = self._container_rect
        self._clamp_scroll(rec)
        prev_clip = screen.get_clip()
        screen.set_clip(cont)
        pygame.draw.rect(screen, (14, 16, 22), cont)
        if rec is None:
            screen.blit(font.render("coreset vide", FONT_AA, TEXT_MUTED),
                        (cont.x + 16 - self.scroll_x, cont.y - self.scroll_y))
        else:
            ox = cont.x + 16 - self.scroll_x
            oy = cont.y - self.scroll_y
            y = self._draw_row(screen, rec, "real", ox, oy, font_small, font_tiny)
            y = self._draw_row(screen, rec, "argmax", ox, y, font_small, font_tiny)
            y = self._draw_row(screen, rec, "elected", ox, y, font_small, font_tiny)
            yy = y + 4
            for lbl, eos, col in (("EOS obs", rec.eos_observed, TEXT_MUTED),
                                  ("EOS mu", rec.eos_argmax, ROW_COLORS["argmax"]),
                                  ("EOS elu", rec.eos_elected, ROW_COLORS["elected"])):
                txt = f"{lbl}: " + " ".join(f"{v:.2f}" for v in eos)
                screen.blit(font_tiny.render(txt, FONT_AA, col), (ox, yy))
                yy += 14
        screen.set_clip(prev_clip)
        pygame.draw.rect(screen, CARD_BORDER, (0, self.height - 24, self.width, 24))
        screen.blit(font_small.render(self.footer_hints, FONT_AA, TEXT_MUTED),
                    (16, self.height - 20))


def run_theater(buffer_path: str = "buf_ckpt.pt", wm_path: str = "wm_ckpt.pt",
                pol_path: str = "pol_ckpt.pt", initial_index: int = 0,
                device: str = "cpu", initial_sort: str = "relief_desc",
                d_model: int = 64, nhead: int = 4, wm_layers: int = 3,
                dim_ff: int = 384, wm_dropout: float = 0.0, pol_hidden: int = 256,
                lr_wm: float = 3e-4, lr_pol: float = 2e-4, seed: int = 0,
                buffer_capacity: int = 1000, buffer_pool_capacity: int = 1000,
                seq_steps: int = 10, n_ctx: int = 4,
                eos_decay: float = 0.9, eos_poids_chemin: float = 0.1,
                eos_max_p: float = 8.0, compile_wm: bool = False,
                **_ignored) -> None:
    os.environ["SDL_HINT_RENDER_SCALE_QUALITY"] = "1"
    pygame.init()
    pygame.display.set_caption("WM Theater — refonte")
    width, height = 1120, 600
    screen = pygame.display.set_mode((width, height), pygame.SCALED | pygame.RESIZABLE)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(FONT_NAME, 14, bold=True)
    font_small = pygame.font.SysFont(FONT_NAME, 11)
    font_tiny = pygame.font.SysFont(FONT_NAME, 10)

    brain = Brain(device=device, compile_wm=compile_wm, seq_steps=seq_steps, n_ctx=n_ctx,
                  d_model=d_model, nhead=nhead, layers=wm_layers, dim_ff=dim_ff,
                  dropout=wm_dropout, pol_hidden=pol_hidden, lr_wm=lr_wm, lr_pol=lr_pol,
                  seed=seed, buffer_capacity=buffer_capacity,
                  buffer_pool_capacity=buffer_pool_capacity,
                  eos_decay=eos_decay, eos_poids_chemin=eos_poids_chemin, eos_max_p=eos_max_p)
    if os.path.exists(wm_path):
        brain.load_checkpoint(wm_path=wm_path,
                              pol_path=pol_path if os.path.exists(pol_path) else "nonexistent.pt")
        print(f"Loaded World Model from {wm_path}")
    else:
        print("Warning: WM checkpoint not found, using initialized weights.")

    buf = ExperienceBuffer(capacity=buffer_capacity, pool_capacity=buffer_pool_capacity,
                           seq_len=seq_steps + 1, n_ctx=n_ctx, seed=seed,
                           eos_decay=eos_decay, eos_poids_chemin=eos_poids_chemin)
    if os.path.exists(buffer_path):
        ok = buf.load(buffer_path)
        print(f"Loaded {buf.coreset_size} sequences from {buffer_path}" if ok
              else f"{buffer_path}: layout incompatible (legacy ?) — coreset vide")
    else:
        print(f"Warning: Buffer checkpoint {buffer_path} not found.")

    theater = WMTheater(brain, buf, width=width, height=height, initial_sort=initial_sort)
    theater.set_seq_idx(initial_index)

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
            theater.handle_event(ev)
        theater.draw(screen, font, font_small, font_tiny)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


def main() -> None:
    cfg = CFG.load_config()
    params: dict = {
        "buffer_path": "buf_ckpt.pt", "wm_path": "wm_ckpt.pt", "pol_path": "pol_ckpt.pt",
        "initial_index": 0,
        "device": os.environ.get("LAMBDA_DEVICE", "cpu"),
        "initial_sort": "relief_desc", "accel": False,
        "d_model": 64, "nhead": 4, "wm_layers": 3,
        "dim_ff": 384, "wm_dropout": 0.0, "pol_hidden": 256,
        "lr_wm": 3e-4, "lr_pol": 2e-4, "seed": 0,
        "buffer_capacity": 1000, "buffer_pool_capacity": 1000,
        "seq_steps": 10, "n_ctx": 4,
        "eos_decay": 0.9, "eos_poids_chemin": 0.1, "eos_max_p": 8.0,
        "compile_wm": False,
    }
    CFG.apply_toml(params, cfg)
    p = argparse.ArgumentParser(description="WM Theater — coreset browser (refonte)")
    p.add_argument("--buffer", type=str, default=argparse.SUPPRESS, dest="buffer_path")
    p.add_argument("--wm", type=str, default=argparse.SUPPRESS, dest="wm_path")
    p.add_argument("--pol", type=str, default=argparse.SUPPRESS, dest="pol_path")
    p.add_argument("--index", type=int, default=argparse.SUPPRESS, dest="initial_index")
    p.add_argument("--device", type=str, default=argparse.SUPPRESS,
                   choices=["cpu", "cuda", "auto", "cpu-fast"])
    p.add_argument("--sort", type=str, default=argparse.SUPPRESS, dest="initial_sort",
                   choices=SORT_MODES)
    p.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    p.add_argument("--compile", action="store_true", default=argparse.SUPPRESS, dest="compile_wm")
    CFG.merge_cli(params, vars(p.parse_args()))
    accel = params.pop("accel", False)
    if accel:
        params["device"] = "auto"
    run_theater(**params)


if __name__ == "__main__":
    main()
