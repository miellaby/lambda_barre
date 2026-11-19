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
import time
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
    step_popover_lines,
    draw_popover,
)
from .tokenize import SALVE_TOKENS, SIG_OFFSET, STATE_TOKENS, salve_cost

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

# hauteur max des IHM : 520 px (écran 600 - barre de tâche 32 - titre ~28 - marge)


@dataclass
class WMTheaterRecord:
    seq_idx: int
    total_seqs: int
    vivacity: float
    real_steps: list[DreamStep]        # 11 salves temporelles s0..s10 (métriques)
    disp_real: list[DreamStep]         # vignettes : s0..s{n_ctx-1}, s{n_ctx}, EOS reel, s{n_ctx}+a*, s{n_ctx+1}..s{seq_steps}
    disp_argmax: list[DreamStep]       # idem, EOS = mu (plus probable), a_dec = a*_mu
    disp_elected: list[DreamStep]      # idem, EOS = elu (optimal), a_dec = a*_elu
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


def _eos_vignette(eos_slots, label: str) -> DreamStep:
    """Vignette bargraph signé du dénouement (entre s{n_ctx} seule et s{n_ctx}+a_dec)."""
    return DreamStep(state_tokens=[], action=None, eos_slots=list(eos_slots),
                     label=label)


def _gen_to_steps(brain, gen, real_steps, eos_slots, b: int = 0) -> tuple[list[DreamStep], list[float]]:
    """Transforme une génération [contexte, dénouement] en rangée d'affichage :
    s0..s{n_ctx-1}+a, s{n_ctx}, dénouement (bargraph), s{n_ctx}+a*, puis la conséquence rétro-causale
    générée réordonnée en temporel s_{n_ctx+1}^..s_{seq_steps}^. Chaque unité
    de la conséquence générée est [s_k inversée (13 tokens), a_{k-1} (3)] ;
    la dernière unité n'a que s_{n_ctx+1} (a_dec est à part, dans gen["action"])."""
    conseq = gen["conseq"][b]                                   # [conseq_len, 29]
    a_star = brain._decode_action_batch(gen["action"])[b].tolist()
    n_im = brain.n_imagine
    units = [conseq[j * SALVE_TOKENS:(j + 1) * SALVE_TOKENS].tolist()
             for j in range(n_im - 1)]                          # unités complètes s_k+a_{k-1}
    units.append(conseq[(n_im - 1) * SALVE_TOKENS:].tolist())  # dernière unité : s_{n_ctx+1} seule
    # Feature rétro-causale : chaque état s_k de la conséquence est affiché avec
    # l'action qui le produit, a_{k-1} (contenue dans la même unité) — la chaîne
    # remonte jusqu'à la décision a_{n_ctx} (première action, affichée à part).
    blocks = []
    for u in units:
        s_toks = u[:STATE_TOKENS][::-1]                         # ré-inverser s_k
        a_toks = u[STATE_TOKENS:] if len(u) >= SALVE_TOKENS else None
        blocks.append((s_toks, a_toks))
    blocks.reverse()                                            # ordre temporel s_{n_ctx+1}..s_{seq_steps}
    n_ctx = brain.n_ctx
    s_dec = real_steps[n_ctx]
    steps: list[DreamStep] = (
        list(real_steps[:n_ctx])                                # s0..s3 (+a)
        + [s_dec]                                               # s{n_ctx} seule
        + [_eos_vignette(eos_slots, "EOS")]                     # vignette bargraph
        + [DreamStep(state_tokens=s_dec.state_tokens, action=a_star,
                     step_cost=s_dec.step_cost, label=f"s{n_ctx}+a*")]  # s{n_ctx}+a*
    )
    for j, (s_toks, a_toks) in enumerate(blocks):
        a = None
        # dernière salve générée (k=seq_steps) : pas d'action affichée, comme
        # dans la rangée réelle (a_{seq_steps} n'existe pas dans la séquence)
        if a_toks is not None and j < len(blocks) - 1:
            a_vals = brain._decode_action_batch(torch.tensor(
                [a_toks], dtype=torch.float32, device=brain.device))[0].tolist()
            if any(abs(v) > 1e-9 for v in a_vals):
                a = a_vals
        steps.append(DreamStep(state_tokens=s_toks, action=a,
                               step_cost=salve_cost(s_toks) if len(s_toks) >= 13 else 0.0,
                               label=f"s{n_ctx + 1 + j}^"))
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
    # dénouement élu : PAS de règle de l'observé ici — le manager montre le vrai
    # élu (max relief des tirages) ; la règle eos_observed reste réservée à
    # l'entraînement de la policy (_policy_step)
    eos_elu, p_elu = brain.sample_denouements(ctx)
    # les deux générations (µ et élu) sont batchées en UNE seule passe incrémentale
    # (le rollout token-par-token domine le temps d'affichage)
    ctx2 = torch.cat([ctx, ctx], dim=0)
    eos2 = torch.cat([mu, eos_elu], dim=0)
    gen2 = brain.generate_from_denouement(ctx2, eos2)
    gen_mu = {k: v[:1] for k, v in gen2.items()}
    gen_el = {k: v[1:] for k, v in gen2.items()}

    disp_argmax, a_mu = _gen_to_steps(brain, gen_mu, real_steps, mu[0].tolist(), b=0)
    disp_elected, a_el = _gen_to_steps(brain, gen_el, real_steps, eos_elu[0].tolist(), b=0)

    a_dec_real = brain._decode_action_batch(
        vals[:, brain.action_pos:brain.action_pos + 3, :])[0].tolist()
    # rangée réelle : s0..s{n_ctx-1}, s{n_ctx}, EOS réel (bargraph), s{n_ctx}+a_dec, puis la conséquence
    s_dec = real_steps[brain.n_ctx]
    disp_real = (list(real_steps[:brain.n_ctx]) + [s_dec]
                 + [_eos_vignette(eos_obs[0].tolist(), "EOS réel")]
                 + [DreamStep(state_tokens=s_dec.state_tokens, action=a_dec_real,
                              step_cost=s_dec.step_cost, label=f"s{brain.n_ctx}+a_dec")]
                 + list(real_steps[brain.n_ctx + 1:]))

    # métriques sur la conséquence (s_{n_ctx+1}..s_{seq_steps}) : dans les rangées
    # d'affichage, s_k siège à l'index k+2 (vignettes EOS et s{n_ctx}+a* insérées)
    p_errs, c_errs = [], []
    for k in range(brain.n_ctx + 1, brain.seq_steps + 1):
        p_rmse = compute_posture_rmse(real_steps[k].state_tokens,
                                      disp_elected[k + 2].state_tokens)
        c_rmse = compute_cost_rmse(_salve_costs(real_steps[k].state_tokens),
                                   _salve_costs(disp_elected[k + 2].state_tokens))
        p_errs.append(p_rmse)
        c_errs.append(c_rmse)

    return WMTheaterRecord(
        seq_idx=seq_idx,
        total_seqs=total_seqs,
        vivacity=vivacity,
        real_steps=real_steps,
        disp_real=disp_real,
        disp_argmax=disp_argmax,
        disp_elected=disp_elected,
        eos_observed=eos_obs[0].tolist(),
        eos_argmax=mu[0].tolist(),
        eos_elected=eos_elu[0].tolist(),
        p_elected=float(p_elu[0]),
        real_relief=float(relief_score(seq, eos_pos=brain.eos_pos)),
        argmax_relief=float(relief_score(mu)[0]),
        elected_relief=float(relief_score(eos_elu)[0]),
        cos_argmax=_cos(a_mu, a_dec_real),
        cos_elected=_cos(a_el, a_dec_real),
        rmse_posture_elected=sum(p_errs) / max(1, len(p_errs)),
        rmse_cost_elected=sum(c_errs) / max(1, len(c_errs)),
    )


class WMTheater:
    """Coreset browser: 3 rows per sequence (real / WM argmax / WM elected)."""

    def __init__(self, brain: Brain, buffer: ExperienceBuffer,
                 width: int = 960, height: int = 520,
                 initial_sort: str = "coreset"):
        self.brain = brain
        self.buffer = buffer
        self.width = width
        self.height = height
        self.sort_mode = initial_sort if initial_sort in SORT_MODES else "coreset"
        self.sorted_idx = 0
        self.cache: dict[int, WMTheaterRecord] = {}
        self.footer_hints = "Nav: [← / →] / glider  |  [S] Sort  |  [R] Random  |  drag: scroll mignatures  |  [Esc] Quit"
        self.title_str = "WM THEATER"
        # géométrie du container scrollable des mignatures
        self.thumb_w, self.thumb_h, self.gap = 76, 56, 4
        self.header_h = 62
        self.bottom_margin = 24          # footer ; le manager ajoute sa strip
        self.scroll_x = 0
        self.scroll_y = 0
        self._drag: tuple | None = None  # (mouse_pos, scroll au départ du drag)
        self.mouse_pos = (0, 0)          # survol des vignettes (popover)
        self._thumbs: list = []          # (rect, DreamStep) dessinés à la frame
        # slider de sélection + debouncing de la navigation (dernier index demandé
        # appliqué après debounce_s d'inactivité : évite de recalculer un rollout
        # à chaque frame pendant un scroll rapide)
        self.debounce_s = 0.25
        self._pending_idx: int | None = None
        self._pending_t = 0.0
        self._slider_rect: pygame.Rect | None = None
        self._slider_drag = False
        # filtre par similarité de contexte (touche F dans le manager) :
        # ctx_ref = signaux du contexte de référence, _filtered_idx = indices gardés
        self.ctx_filter_on = False
        self.ctx_eps = 1e-3
        self.ctx_ref: torch.Tensor | None = None
        self._ref_idx = -1
        self._filtered_idx: list[int] | None = None
        self._update_sorted_indices()

    def set_ctx_reference(self, idx: int) -> int:
        """Filtre le coreset aux séquences dont le contexte est similaire à celui
        de la séquence ``idx`` (MSE sur les signaux du contexte <= ``ctx_eps``).
        Retourne le nombre de séquences gardées."""
        if not self.buffer._coreset:
            return 0
        idx = max(0, min(idx, self.buffer.coreset_size - 1))
        allv = self.brain._batch_tensors(self.buffer._coreset)
        ctxs = allv[:, :self.brain.ctx_len, SIG_OFFSET:]          # [N, ctx_len, 16]
        ref = ctxs[idx:idx + 1]
        d = ((ctxs - ref) ** 2).mean(dim=(1, 2))                  # [N]
        keep = (d <= self.ctx_eps).nonzero().flatten().tolist()
        if idx not in keep:
            keep.append(idx)
        self.ctx_ref = ref
        self._ref_idx = idx
        self.ctx_filter_on = True
        self._filtered_idx = sorted(keep)
        self._update_sorted_indices()
        self.sorted_idx = self.order.index(idx) if idx in self.order else 0
        self.cache.clear()
        return len(keep)

    def clear_ctx_reference(self) -> None:
        """Reset du filtre : toutes les séquences du coreset redeviennent visibles."""
        self.ctx_filter_on = False
        self.ctx_ref = None
        self._ref_idx = -1
        self._filtered_idx = None
        self._update_sorted_indices()
        self.sorted_idx = 0
        self.cache.clear()

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
        n = len(rec.disp_real)
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
        base = self._filtered_idx if (self.ctx_filter_on and self._filtered_idx) \
            else list(range(n))
        if self.sort_mode == "coreset":
            self.order = list(base)
        else:
            rels = {i: self._seq_relief(i) for i in base}
            if self.sort_mode == "relief_desc":
                self.order = sorted(base, key=lambda i: -rels[i])
            elif self.sort_mode == "relief_asc":
                self.order = sorted(base, key=lambda i: rels[i])
            else:  # abs_desc
                self.order = sorted(base, key=lambda i: -abs(rels[i]))
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

    def _request_nav(self, idx: int) -> None:
        """Demande de navigation debouncée : seul le dernier index demandé est
        appliqué (dans ``update``) après ``debounce_s`` d'inactivité."""
        if not self.order:
            return
        self._pending_idx = max(0, min(idx, len(self.order) - 1))
        self._pending_t = time.monotonic()

    def update(self) -> None:
        """Applique la navigation en attente si le debounce est écoulé."""
        if self._pending_idx is not None \
                and time.monotonic() - self._pending_t >= self.debounce_s:
            self.sorted_idx = self._pending_idx
            self._pending_idx = None

    def _slider_to(self, x: int) -> None:
        r = self._slider_rect
        if r is None or not self.order:
            return
        frac = (x - r.x) / max(1, r.w)
        self._request_nav(int(frac * (len(self.order) - 1)))

    def prev_seq(self) -> None:
        self._request_nav(self.sorted_idx - 1)

    def next_seq(self) -> None:
        self._request_nav(self.sorted_idx + 1)

    def random_seq(self) -> None:
        if self.order:
            self._request_nav(random.randrange(len(self.order)))

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
            if self._slider_rect is not None and self._slider_rect.collidepoint(ev.pos):
                self._slider_drag = True
                self._slider_to(ev.pos[0])
                return True
            if self._container_rect.collidepoint(ev.pos):
                self._drag = ev.pos
                return True
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self._slider_drag = False
            self._drag = None
        elif ev.type == pygame.MOUSEMOTION:
            self.mouse_pos = ev.pos
            if self._slider_drag and self._slider_rect is not None:
                self._slider_to(ev.pos[0])
                return True
            if self._drag is not None:
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
        steps = {"real": rec.disp_real, "argmax": rec.disp_argmax,
                 "elected": rec.disp_elected}[key]
        screen.blit(font_small.render(ROW_LABELS[key], FONT_AA, color), (x0, y0))
        yy = y0 + 16
        x = x0
        for st in steps:
            rect = pygame.Rect(x, yy, self.thumb_w, self.thumb_h)
            draw_thumbnail(screen, rect, st, font_tiny, facing=1, cand_color=color)
            self._thumbs.append((rect, st))
            x += self.thumb_w + self.gap
        return yy + self.thumb_h + 8

    def _draw_header(self, screen, rec: WMTheaterRecord | None, font, font_small) -> int:
        pygame.draw.rect(screen, HEADER_BG, (0, 0, self.width, self.header_h - 6))
        title = f"{self.title_str}  |  seq {self.sorted_idx + 1}/{len(self.order)} (idx {rec.seq_idx if rec else '-'})  |  sort {self.sort_mode}"
        if self.ctx_filter_on:
            title += (f"  |  ctx-filter ref={self._ref_idx} kept={len(self.order)}"
                      f"/{self.num_seqs} eps={self.ctx_eps:g}")
        s_title = font.render(title, FONT_AA, TEXT_WHITE)
        screen.blit(s_title, (16, 8))
        # slider de sélection de séquence, 1re ligne à côté du sort
        n = len(self.order)
        lbl = font_small.render("glider", FONT_AA, TEXT_MUTED)
        sw = min(240, max(90, self.width - (16 + s_title.get_width() + 20 + lbl.get_width() + 8) - 16))
        sx = 16 + s_title.get_width() + 20 + lbl.get_width() + 8
        sy = 10
        self._slider_rect = pygame.Rect(sx, sy, sw, 10)
        screen.blit(lbl, (16 + s_title.get_width() + 20, 8))
        pygame.draw.rect(screen, (40, 46, 60), self._slider_rect, border_radius=5)
        pygame.draw.rect(screen, CARD_BORDER, self._slider_rect, 1, border_radius=5)
        if n > 0:
            def cx_of(i):
                return self._slider_rect.x + 6 + int(i / max(1, n - 1) * (self._slider_rect.w - 12))
            if self._pending_idx is not None:   # dernier index demandé (debounced)
                pygame.draw.circle(screen, TEXT_MUTED,
                                   (cx_of(self._pending_idx), self._slider_rect.centery), 4)
            pygame.draw.circle(screen, TEXT_ACCENT,
                               (cx_of(self.sorted_idx), self._slider_rect.centery), 6)
        if rec is not None:
            info = (f"relief reel {rec.real_relief:+.3f} | mu {rec.argmax_relief:+.3f} | "
                    f"elu {rec.elected_relief:+.3f} (p={rec.p_elected:.2f}) | "
                    f"cos(a*,a_dec) mu {rec.cos_argmax:+.2f} elu {rec.cos_elected:+.2f} | "
                    f"rmse posture {rec.rmse_posture_elected:.3f} cout {rec.rmse_cost_elected:.3f}")
            screen.blit(font_small.render(info, FONT_AA, TEXT_ACCENT), (16, 32))
        return self.header_h

    def draw(self, screen, font, font_small, font_tiny=None) -> None:
        if font_tiny is None:
            font_tiny = font_small
        screen.fill(DREAM_BG)
        self._thumbs.clear()
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
        # popover de la vignette survolée (hors clip du container scrollable)
        for rect, st in self._thumbs:
            if rect.collidepoint(self.mouse_pos):
                draw_popover(screen, font_tiny, self.mouse_pos, step_popover_lines(st))
                break
        pygame.draw.rect(screen, CARD_BORDER, (0, self.height - 24, self.width, 24))
        screen.blit(font_small.render(self.footer_hints, FONT_AA, TEXT_MUTED),
                    (16, self.height - 20))


def run_theater(buffer_path: str = "buf_ckpt.pt", wm_path: str = "wm_ckpt.pt",
                pol_path: str = "pol_ckpt.pt", initial_index: int = 0,
                initial_sort: str = "coreset", **params) -> None:
    os.environ["SDL_HINT_RENDER_SCALE_QUALITY"] = "1"
    pygame.init()
    pygame.display.set_caption("WM Theater — refonte")
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
            brain.world.load_state_dict(wm["world"])
            if "opt_wm" in wm:
                brain.opt_wm.load_state_dict(wm["opt_wm"])
        print(f"Loaded World Model from {wm_path}")
    else:
        print("Warning: WM checkpoint not found, using initialized weights.")

    if os.path.exists(buffer_path):
        ok = brain.buffer.load(buffer_path)
        print(f"Loaded {brain.buffer.coreset_size} sequences from {buffer_path}" if ok
              else f"{buffer_path}: layout incompatible (legacy ?) — coreset vide")
    else:
        print(f"Warning: Buffer checkpoint {buffer_path} not found.")

    theater = WMTheater(brain, brain.buffer, width=width, height=height, initial_sort=initial_sort)
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
        theater.update()
        theater.draw(screen, font, font_small, font_tiny)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


def main() -> None:
    cfg = CFG.load_config()
    params: dict = {
        **CFG.BRAIN_DEFAULTS,
        "buffer_path": "buf_ckpt.pt", "wm_path": "wm_ckpt.pt", "pol_path": "pol_ckpt.pt",
        "initial_index": 0, "initial_sort": "coreset",
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
