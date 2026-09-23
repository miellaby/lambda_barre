"""World Model Theater: Visualizer and convergence diagnostic for the World Model.

Loads sequences from the Coreset (ExperienceBuffer) and compares the ground truth
transitions against the World Model's autoregressive rollout conditioned on s0..s4a4:
- Row 1: Ground Truth (Coreset) sequence (s0 -> s10) with animal thumbnail and instantaneous costs
- Row 2: World Model inference: context (s0..s4a4) + autoregressive predictions (s5^ -> s10^)
  with predicted animal thumbnail, predicted instantaneous costs, and error metrics.

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
from .brain import Brain, ExperienceBuffer
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
from .tokenize import SIG_OFFSET, DenseEncoder, _signal_slot

# Colors for instantaneous costs
COST_COLORS = [
    (245, 140, 50),   # EF: Effort (Orange)
    (235, 75, 75),    # DO: Douleur (Red)
    (185, 110, 245),  # CO: Courbature (Violet)
    (240, 220, 60),   # IN: Instabilité (Yellow)
    (60, 215, 215),   # VE: Vertige (Teal)
]
COST_LABELS = ["EF", "DO", "CO", "IN", "VE"]

CONTEXT_BG = (22, 25, 33)
CONTEXT_BORDER = (45, 52, 68)
PRED_BORDER = (100, 180, 255)


@dataclass
class WMStepData:
    step_idx: int
    real_step: DreamStep
    pred_step: DreamStep | None
    real_costs: list[float]          # 5 floats [EF, DO, CO, IN, VE]
    pred_costs: list[float] | None   # 5 floats [EF, DO, CO, IN, VE]
    posture_rmse: float | None       # Physical posture RMSE
    cost_rmse: float | None          # Immediate costs RMSE
    real_salve: list[list[float]] = field(default_factory=list)
    pred_salve: list[list[float]] | None = None


@dataclass
class WMTheaterRecord:
    seq_idx: int
    total_seqs: int
    vivacity: float
    steps: list[WMStepData]
    overall_posture_rmse: float
    overall_cost_rmse: float


def compute_posture_rmse(real_toks: list[list[float]], pred_toks: list[list[float]]) -> float:
    """Compute physical posture RMSE between real and predicted state tokens."""
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


def compute_cost_rmse(real_costs: list[float], pred_costs: list[float]) -> float:
    """Compute RMSE across the 5 instantaneous cost channels."""
    diffs = [r - p for r, p in zip(real_costs, pred_costs)]
    return math.sqrt(sum(d * d for d in diffs) / len(diffs))


def rollout_sequence(brain: Brain, seq: list[list[list[float]]],
                     seq_idx: int, total_seqs: int, vivacity: float) -> WMTheaterRecord:
    """Perform conditioning on s0..s4a4 and autoregressive rollout for s5..s10."""
    device = brain.device
    num_salves = len(seq)
    steps: list[WMStepData] = []

    # 1. Decode ground truth steps and costs
    real_steps: list[DreamStep] = []
    real_costs_list: list[list[float]] = []

    for i in range(num_salves):
        s_toks = seq[i][:13]
        costs = [float(c) for c in s_toks[10][SIG_OFFSET : SIG_OFFSET + 5]]
        real_costs_list.append(costs)

        if i < num_salves - 1:
            a_toks = torch.tensor(seq[i][13:16], dtype=torch.float32, device=device).unsqueeze(0)
            a_vals = brain._decode_action_batch(a_toks)[0].tolist()
            lbl = f"s{i}+a{i}"
        else:
            a_vals = None
            lbl = f"s{i} (EOS)"

        real_steps.append(DreamStep(state_tokens=s_toks, action=a_vals, label=lbl))

    # 2. Build context up to s4a4 using brain._batch_tensors to ensure
    # 100% exact parity with training format (including zero-padded Token 12).
    vals = brain._batch_tensors([seq])  # [1, L, 25]
    n_ctx = min(5, num_salves - 1)
    ctx = vals[:, :n_ctx * 16, :].clone()

    # Context steps (0..n_ctx-1): real only
    for i in range(n_ctx):
        steps.append(WMStepData(
            step_idx=i,
            real_step=real_steps[i],
            pred_step=None,
            real_costs=real_costs_list[i],
            pred_costs=None,
            posture_rmse=None,
            cost_rmse=None,
            real_salve=seq[i],
            pred_salve=None,
        ))

    # 3. Autoregressive prediction starting after s4a4 (s5 -> s10)
    pred_posture_errors = []
    pred_cost_errors = []

    with torch.no_grad():
        for k in range(n_ctx, num_salves):
            # Predict next 13 state tokens
            pred_s = brain.world.predict_next_state(ctx)  # [1, 13, 25]
            pred_s_toks = pred_s[0].tolist()
            pred_costs = [float(c) for c in pred_s_toks[10][SIG_OFFSET : SIG_OFFSET + 5]]

            # Error metrics
            p_rmse = compute_posture_rmse(real_steps[k].state_tokens, pred_s_toks)
            c_rmse = compute_cost_rmse(real_costs_list[k], pred_costs)
            pred_posture_errors.append(p_rmse)
            pred_cost_errors.append(c_rmse)

            if k < num_salves - 1:
                # Use ground truth action for next transition
                a_vals = real_steps[k].action
                pred_step = DreamStep(state_tokens=pred_s_toks, action=a_vals, label=f"s{k}^+a{k}")
                # Append predicted state and real action to context
                # Zero out Token 12 signal slots of intermediate predicted state to match training distribution
                pred_s_zeroed = pred_s.clone()
                pred_s_zeroed[:, 12, SIG_OFFSET:] = 0.0
                real_a_toks = vals[:, k * 16 + 13 : (k + 1) * 16, :]
                ctx = torch.cat([ctx, pred_s_zeroed, real_a_toks], dim=1)
                pred_salve_toks = pred_s_toks + seq[k][13:16]
            else:
                pred_step = DreamStep(state_tokens=pred_s_toks, action=None, label=f"s{k}^ (EOS)")
                pred_salve_toks = pred_s_toks

            steps.append(WMStepData(
                step_idx=k,
                real_step=real_steps[k],
                pred_step=pred_step,
                real_costs=real_costs_list[k],
                pred_costs=pred_costs,
                posture_rmse=p_rmse,
                cost_rmse=c_rmse,
                real_salve=seq[k],
                pred_salve=pred_salve_toks,
            ))

    avg_p_rmse = sum(pred_posture_errors) / max(1, len(pred_posture_errors))
    avg_c_rmse = sum(pred_cost_errors) / max(1, len(pred_cost_errors))

    return WMTheaterRecord(
        seq_idx=seq_idx,
        total_seqs=total_seqs,
        vivacity=vivacity,
        steps=steps,
        overall_posture_rmse=avg_p_rmse,
        overall_cost_rmse=avg_c_rmse,
    )


def draw_cost_histogram(surface: pygame.Surface, rect: pygame.Rect,
                        costs: list[float], font_tiny: pygame.font.Font,
                        title: str = "", is_predicted: bool = False) -> None:
    """Render a mini histogram for the 5 instantaneous cost components."""
    rx, ry, rw, rh = rect.x, rect.y, rect.width, rect.height

    # Background card
    bg = (20, 24, 32) if not is_predicted else (24, 28, 40)
    border = CARD_BORDER if not is_predicted else (60, 80, 110)
    pygame.draw.rect(surface, bg, rect, border_radius=3)
    pygame.draw.rect(surface, border, rect, 1, border_radius=3)

    # Header label
    if title:
        s_title = font_tiny.render(title, True, TEXT_MUTED)
        surface.blit(s_title, (rx + 4, ry + 2))

    # Total cost sum
    total_val = sum(costs)
    s_tot = font_tiny.render(f"Σ {total_val:.2f}", True, TEXT_WHITE if total_val > 0.05 else TEXT_MUTED)
    surface.blit(s_tot, (rx + rw - s_tot.get_width() - 4, ry + 2))

    # 5 Bars layout
    n_bars = 5
    bar_w = 11
    bar_gap = 4
    total_bars_w = n_bars * bar_w + (n_bars - 1) * bar_gap
    start_x = rx + (rw - total_bars_w) // 2
    base_y = ry + rh - 13
    max_h = rh - 28

    for i in range(n_bars):
        bx = start_x + i * (bar_w + bar_gap)
        val = max(0.0, min(1.0, costs[i] if i < len(costs) else 0.0))
        bh = max(1, int(val * max_h))

        # Empty bar background
        pygame.draw.rect(surface, (36, 42, 56), (bx, base_y - max_h, bar_w, max_h), border_radius=1)

        # Active filled bar
        c = COST_COLORS[i]
        if is_predicted:
            # Slightly softer tint for predicted
            c = tuple(min(255, int(v * 0.95)) for v in c)
        pygame.draw.rect(surface, c, (bx, base_y - bh, bar_w, bh), border_radius=1)

        # Bar label
        lbl = COST_LABELS[i]
        s_lbl = font_tiny.render(lbl, True, TEXT_MUTED)
        surface.blit(s_lbl, (bx + (bar_w - s_lbl.get_width()) // 2, base_y + 2))


def draw_token_popin(surface: pygame.Surface, encoder: DenseEncoder,
                     font_small: pygame.font.Font, font_tiny: pygame.font.Font,
                     title: str, salve: list[list[float]],
                     card_rect: pygame.Rect, screen_w: int, screen_h: int) -> None:
    """Render an overlay pop-in tooltip displaying decoded tokens of the hovered salve."""
    if not salve:
        return

    pop_w = 510
    pop_h = 192

    # Horizontal positioning: centered on card, clamped to screen margins
    px = max(12, min(screen_w - pop_w - 12, card_rect.centerx - pop_w // 2))

    # Vertical positioning: above or below card to avoid covering it
    if card_rect.centery < screen_h // 2:
        py = card_rect.bottom + 8
        if py + pop_h > screen_h - 12:
            py = max(12, card_rect.top - pop_h - 8)
    else:
        py = card_rect.top - pop_h - 8
        if py < 12:
            py = min(screen_h - pop_h - 12, card_rect.bottom + 8)

    pop_rect = pygame.Rect(px, py, pop_w, pop_h)

    # Dark translucent drop shadow
    s_shadow = pygame.Surface((pop_w, pop_h), pygame.SRCALPHA)
    s_shadow.fill((0, 0, 0, 160))
    surface.blit(s_shadow, (px + 4, py + 4))

    # Main pop-in container
    pygame.draw.rect(surface, (18, 22, 32), pop_rect, border_radius=6)
    pygame.draw.rect(surface, (90, 160, 255), pop_rect, 2, border_radius=6)

    # Header bar
    s_title = font_small.render(title, True, (255, 215, 80))
    surface.blit(s_title, (px + 10, py + 6))
    s_cnt = font_tiny.render(f"{len(salve)} tokens", True, TEXT_MUTED)
    surface.blit(s_cnt, (px + pop_w - s_cnt.get_width() - 10, py + 8))

    pygame.draw.line(surface, (45, 55, 75), (px + 8, py + 26), (px + pop_w - 8, py + 26), 1)

    # Format tokens with compact Vision summary
    raw_lines = encoder.decode(salve)
    formatted: list[tuple[str, str, str]] = []
    for i, line in enumerate(raw_lines):
        parts = line.split(" ", 1)
        tag = parts[0]
        body = parts[1] if len(parts) > 1 else ""

        if i == 4 and len(salve) > 4:
            v_tok = salve[4]
            vals = [_signal_slot(v_tok, k) for k in range(16)]
            avg_v = sum(vals) / 16.0
            body = f"Vis(16): moy={avg_v:.2f} [{min(vals):.2f}-{max(vals):.2f}]"

        kind = "A" if tag.startswith("[A") else "S"
        formatted.append((tag, kind, body))

    # 2 columns layout (8 lines max per column)
    col_w = (pop_w - 28) // 2
    line_h = 18
    base_y = py + 32

    for idx, (tag, kind, body) in enumerate(formatted):
        col = 0 if idx < 8 else 1
        row = idx if idx < 8 else (idx - 8)
        tx = px + 10 + col * (col_w + 8)
        ty = base_y + row * line_h

        tag_c = (255, 205, 90) if kind == "A" else (100, 200, 255)
        s_tag = font_tiny.render(tag, True, tag_c)
        surface.blit(s_tag, (tx, ty))

        s_body = font_tiny.render(body, True, (215, 222, 235))
        surface.blit(s_body, (tx + s_tag.get_width() + 4, ty))


class WMTheater:
    """Standalone interactive visualizer for World Model inference."""

    def __init__(self, brain: Brain, buffer: ExperienceBuffer,
                 width: int = 1140, height: int = 700):
        self.brain = brain
        self.buffer = buffer
        self.encoder = DenseEncoder()
        self.width = width
        self.height = height

        self.seq_idx = 0
        self.num_seqs = len(self.buffer._coreset)
        self.cache: dict[int, WMTheaterRecord] = {}

        # Slider interaction
        self.slider_rect = pygame.Rect(250, 48, 540, 16)
        self.slider_dragging = False

        # Button rects
        self.btn_prev = pygame.Rect(140, 42, 90, 28)
        self.btn_next = pygame.Rect(810, 42, 90, 28)
        self.btn_rand = pygame.Rect(920, 42, 100, 28)

        # Dimensions for columns
        self.col_w = 94
        self.col_gap = 6
        self.margin_x = (self.width - (11 * self.col_w + 10 * self.col_gap)) // 2

        # Precompute initial sequence
        if self.num_seqs > 0:
            self._load_record(self.seq_idx)

    def _load_record(self, idx: int) -> WMTheaterRecord:
        idx = max(0, min(self.num_seqs - 1, idx))
        if idx not in self.cache:
            seq = self.buffer._coreset[idx]
            viv = (self.buffer._coreset_vivacity[idx]
                   if idx < len(self.buffer._coreset_vivacity) else 1.0)
            self.cache[idx] = rollout_sequence(
                self.brain, seq, idx, self.num_seqs, viv)
        return self.cache[idx]

    def set_seq_idx(self, idx: int) -> None:
        if self.num_seqs <= 0:
            return
        self.seq_idx = max(0, min(self.num_seqs - 1, idx))
        self._load_record(self.seq_idx)

    def prev_seq(self) -> None:
        self.set_seq_idx((self.seq_idx - 1) % max(1, self.num_seqs))

    def next_seq(self) -> None:
        self.set_seq_idx((self.seq_idx + 1) % max(1, self.num_seqs))

    def random_seq(self) -> None:
        if self.num_seqs > 0:
            self.set_seq_idx(random.randint(0, self.num_seqs - 1))

    def handle_event(self, ev: pygame.event.Event) -> bool:
        """Handle keyboard and mouse events. Returns True if handled."""
        if ev.type == pygame.KEYDOWN:
            if ev.key in (pygame.K_LEFT, pygame.K_a):
                self.prev_seq()
                return True
            elif ev.key in (pygame.K_RIGHT, pygame.K_d):
                self.next_seq()
                return True
            elif ev.key == pygame.K_r:
                self.random_seq()
                return True
            elif ev.key == pygame.K_HOME:
                self.set_seq_idx(0)
                return True
            elif ev.key == pygame.K_END:
                self.set_seq_idx(self.num_seqs - 1)
                return True

        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            mx, my = ev.pos
            if self.btn_prev.collidepoint(mx, my):
                self.prev_seq()
                return True
            elif self.btn_next.collidepoint(mx, my):
                self.next_seq()
                return True
            elif self.btn_rand.collidepoint(mx, my):
                self.random_seq()
                return True
            elif (self.slider_rect.x - 8 <= mx <= self.slider_rect.x + self.slider_rect.width + 8
                  and self.slider_rect.y - 10 <= my <= self.slider_rect.y + self.slider_rect.height + 10):
                self.slider_dragging = True
                self._update_slider(mx)
                return True

        elif ev.type == pygame.MOUSEMOTION:
            if self.slider_dragging:
                self._update_slider(ev.pos[0])
                return True

        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            if self.slider_dragging:
                self.slider_dragging = False
                return True

        elif ev.type == pygame.MOUSEWHEEL:
            if ev.y > 0:
                self.prev_seq()
            elif ev.y < 0:
                self.next_seq()
            return True

        return False

    def _update_slider(self, mx: int) -> None:
        if self.num_seqs <= 1:
            return
        t = (mx - self.slider_rect.x) / self.slider_rect.width
        t = max(0.0, min(1.0, t))
        idx = int(round(t * (self.num_seqs - 1)))
        self.set_seq_idx(idx)

    def draw(self, screen: pygame.Surface, font: pygame.font.Font,
             font_small: pygame.font.Font, font_tiny: pygame.font.Font) -> None:
        """Render the complete World Model Theater storyboard."""
        screen.fill(DREAM_BG)

        if self.num_seqs == 0:
            msg = font.render("Coreset is empty — no sequences to display.", True, TEXT_WHITE)
            screen.blit(msg, (self.width // 2 - msg.get_width() // 2, self.height // 2))
            return

        rec = self._load_record(self.seq_idx)

        # 1. Header Bar
        header_h = 78
        pygame.draw.rect(screen, HEADER_BG, (0, 0, self.width, header_h))
        pygame.draw.line(screen, CARD_BORDER, (0, header_h), (self.width, header_h), 1)

        # Title & Key Metrics
        title_str = (
            f"WORLD MODEL THEATER  |  Seq {rec.seq_idx + 1}/{rec.total_seqs}  |  "
            f"Vivacity: {rec.vivacity:.2f}  |  "
            f"Rollout Posture RMSE: {rec.overall_posture_rmse:.3f}  |  "
            f"Cost RMSE: {rec.overall_cost_rmse:.3f}"
        )
        s_title = font.render(title_str, True, TEXT_WHITE)
        screen.blit(s_title, (16, 12))

        # Prev button
        pygame.draw.rect(screen, (34, 40, 54), self.btn_prev, border_radius=4)
        pygame.draw.rect(screen, CARD_BORDER, self.btn_prev, 1, border_radius=4)
        s_prev = font_small.render("< Prev [←]", True, TEXT_WHITE)
        screen.blit(s_prev, (self.btn_prev.x + (self.btn_prev.width - s_prev.get_width()) // 2,
                             self.btn_prev.y + 6))

        # Slider track
        pygame.draw.rect(screen, (28, 32, 44), self.slider_rect, border_radius=4)
        pygame.draw.rect(screen, CARD_BORDER, self.slider_rect, 1, border_radius=4)

        # Slider fill
        ratio = rec.seq_idx / max(1, self.num_seqs - 1)
        fill_w = int(ratio * self.slider_rect.width)
        if fill_w > 0:
            fill_rect = pygame.Rect(self.slider_rect.x, self.slider_rect.y, fill_w, self.slider_rect.height)
            pygame.draw.rect(screen, (50, 110, 190), fill_rect, border_radius=4)

        # Slider handle
        hx = self.slider_rect.x + fill_w
        hy = self.slider_rect.centery
        pygame.draw.circle(screen, (110, 200, 255), (hx, hy), 8)
        pygame.draw.circle(screen, TEXT_WHITE, (hx, hy), 8, 2)

        # Next button
        pygame.draw.rect(screen, (34, 40, 54), self.btn_next, border_radius=4)
        pygame.draw.rect(screen, CARD_BORDER, self.btn_next, 1, border_radius=4)
        s_next = font_small.render("Next [→] >", True, TEXT_WHITE)
        screen.blit(s_next, (self.btn_next.x + (self.btn_next.width - s_next.get_width()) // 2,
                             self.btn_next.y + 6))

        # Random button
        pygame.draw.rect(screen, (34, 40, 54), self.btn_rand, border_radius=4)
        pygame.draw.rect(screen, CARD_BORDER, self.btn_rand, 1, border_radius=4)
        s_rand = font_small.render("Random [R]", True, TEXT_ACCENT)
        screen.blit(s_rand, (self.btn_rand.x + (self.btn_rand.width - s_rand.get_width()) // 2,
                             self.btn_rand.y + 6))

        # 2. Section 1: Ground Truth Sequence (Coreset)
        sec1_y = header_h + 12
        s_sec1 = font.render("GROUND TRUTH (Coreset)  —  Posture réelle & Coûts instantanés (s0 → s10)",
                             True, TEXT_WHITE)
        screen.blit(s_sec1, (self.margin_x, sec1_y))

        row1_y = sec1_y + 24
        card_h = 246

        hover_card_info: tuple[str, list[list[float]], pygame.Rect] | None = None
        mouse_pos = pygame.mouse.get_pos()

        for i, step_data in enumerate(rec.steps):
            cx = self.margin_x + i * (self.col_w + self.col_gap)
            card_rect = pygame.Rect(cx, row1_y, self.col_w, card_h)

            is_context = (i < 5)
            is_hovered = card_rect.collidepoint(mouse_pos)
            if is_hovered:
                lbl_text = f"s{i}+a{i}" if i < 10 else "s10 (EOS)"
                hover_card_info = (f"SALVE RÉELLE {lbl_text}", step_data.real_salve, card_rect)

            bg = CONTEXT_BG if is_context else CARD_BG
            if is_hovered:
                border_c = (140, 210, 255)
            else:
                border_c = CONTEXT_BORDER if is_context else CARD_BORDER

            pygame.draw.rect(screen, bg, card_rect, border_radius=4)
            pygame.draw.rect(screen, border_c, card_rect, 2 if is_hovered else 1, border_radius=4)

            # Card Header label
            lbl_text = f"s{i}+a{i}" if i < 10 else "s10 (EOS)"
            c_hdr = TEXT_ACCENT if is_context else TEXT_WHITE
            s_hdr = font_small.render(lbl_text, True, c_hdr)
            screen.blit(s_hdr, (cx + (self.col_w - s_hdr.get_width()) // 2, row1_y + 4))

            # Animal thumbnail
            th_rect = pygame.Rect(cx + 4, row1_y + 20, self.col_w - 8, 70)
            cand_c = CAND_COLORS[0] if step_data.real_step.action is not None else None
            draw_thumbnail(screen, th_rect, step_data.real_step, font_tiny, facing=1,
                           cand_color=cand_c, highlight_label="")

            # Instantaneous Costs Mini-Histogram
            cost_rect = pygame.Rect(cx + 4, row1_y + 96, self.col_w - 8, 142)
            draw_cost_histogram(screen, cost_rect, step_data.real_costs, font_tiny,
                                title="Réel", is_predicted=False)

        # 3. Section 2: World Model Inférence (Rollout après s4a4)
        sec2_y = row1_y + card_h + 14
        s_sec2 = font.render(
            "PRÉDICTION WORLD MODEL  —  Conditionnement réel [s0..s4a4]  →  Rollout autorégressif [s5^..s10^]",
            True, WINNER_BORDER)
        screen.blit(s_sec2, (self.margin_x, sec2_y))

        row2_y = sec2_y + 24

        for i, step_data in enumerate(rec.steps):
            cx = self.margin_x + i * (self.col_w + self.col_gap)
            card_rect = pygame.Rect(cx, row2_y, self.col_w, card_h)

            is_context = (i < 5)
            is_hovered = card_rect.collidepoint(mouse_pos)

            if is_context:
                if is_hovered:
                    hover_card_info = (f"SALVE CONTEXTE s{i}", step_data.real_salve, card_rect)

                # Dimmed context reminder card
                border_c = (140, 210, 255) if is_hovered else CONTEXT_BORDER
                pygame.draw.rect(screen, CONTEXT_BG, card_rect, border_radius=4)
                pygame.draw.rect(screen, border_c, card_rect, 2 if is_hovered else 1, border_radius=4)

                s_hdr = font_small.render(f"[Contexte s{i}]", True, TEXT_MUTED)
                screen.blit(s_hdr, (cx + (self.col_w - s_hdr.get_width()) // 2, row2_y + 4))

                # Context animal thumbnail (dimmed)
                th_rect = pygame.Rect(cx + 4, row2_y + 20, self.col_w - 8, 70)
                draw_thumbnail(screen, th_rect, step_data.real_step, font_tiny, facing=1,
                               cand_color=None, highlight_label="")

                # Context costs mini-histogram
                cost_rect = pygame.Rect(cx + 4, row2_y + 96, self.col_w - 8, 142)
                draw_cost_histogram(screen, cost_rect, step_data.real_costs, font_tiny,
                                    title="Contexte", is_predicted=False)

            else:
                lbl_text = f"s{i}^+a{i}" if i < 10 else "s10^ (EOS)"
                if is_hovered:
                    hover_card_info = (f"SALVE PRÉDITE {lbl_text}", step_data.pred_salve or [], card_rect)

                # Predicted step card
                border_c = WINNER_BORDER if is_hovered else PRED_BORDER
                pygame.draw.rect(screen, (24, 30, 42), card_rect, border_radius=4)
                pygame.draw.rect(screen, border_c, card_rect, 2 if is_hovered else 1, border_radius=4)

                s_hdr = font_small.render(lbl_text, True, WINNER_BORDER)
                screen.blit(s_hdr, (cx + (self.col_w - s_hdr.get_width()) // 2, row2_y + 4))

                # Predicted animal thumbnail
                th_rect = pygame.Rect(cx + 4, row2_y + 20, self.col_w - 8, 70)
                cand_c = CAND_COLORS[0] if (step_data.pred_step and step_data.pred_step.action is not None) else None
                if step_data.pred_step:
                    draw_thumbnail(screen, th_rect, step_data.pred_step, font_tiny, facing=1,
                                   cand_color=cand_c, is_winner=True, highlight_label="")

                # Predicted costs mini-histogram
                cost_rect = pygame.Rect(cx + 4, row2_y + 96, self.col_w - 8, 116)
                draw_cost_histogram(screen, cost_rect, step_data.pred_costs or [0.0]*5, font_tiny,
                                    title="Prédit", is_predicted=True)

                # Error metrics badge at the bottom of the card
                p_err = step_data.posture_rmse or 0.0
                c_err = step_data.cost_rmse or 0.0

                c_color = (110, 230, 140) if p_err < 0.04 else ((255, 210, 70) if p_err < 0.08 else (240, 100, 100))
                s_err1 = font_tiny.render(f"Δpost: {p_err:.3f}", True, c_color)
                s_err2 = font_tiny.render(f"Δcoût: {c_err:.3f}", True, TEXT_WHITE)
                screen.blit(s_err1, (cx + 6, row2_y + 214))
                screen.blit(s_err2, (cx + 6, row2_y + 228))

        # 4. Footer & Legend
        footer_y = self.height - 24
        hints = "Navigation : [← / →] Séquence préc/suiv  |  [R] Aléatoire  |  Survol : Détail tokens  |  [Échap / Q] Quitter"
        s_hints = font_small.render(hints, True, TEXT_MUTED)
        screen.blit(s_hints, (self.margin_x, footer_y))

        # Costs legend
        leg_x = self.width - self.margin_x - 360
        s_leg_title = font_tiny.render("Coûts :", True, TEXT_MUTED)
        screen.blit(s_leg_title, (leg_x, footer_y + 1))
        cur_lx = leg_x + 40
        for code, c in zip(COST_LABELS, COST_COLORS):
            pygame.draw.rect(screen, c, (cur_lx, footer_y + 3, 8, 8), border_radius=1)
            s_c = font_tiny.render(code, True, TEXT_WHITE)
            screen.blit(s_c, (cur_lx + 11, footer_y + 1))
            cur_lx += 42

        # 5. Pop-in overlay on top of everything
        if hover_card_info is not None:
            pop_title, pop_salve, pop_rect = hover_card_info
            draw_token_popin(screen, self.encoder, font_small, font_tiny,
                             pop_title, pop_salve, pop_rect, self.width, self.height)


def run_theater(buffer_path: str = "buf_ckpt.pt",
                wm_path: str = "wm_ckpt.pt",
                initial_index: int = 0,
                device: str = "cpu") -> None:
    """Entry point for running the WM Theater GUI."""
    pygame.init()
    pygame.display.set_caption("World Model Theater — Diagnostic de Convergence")

    width, height = 1140, 700
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("monospace", 13, bold=True)
    font_small = pygame.font.SysFont("monospace", 11)
    font_tiny = pygame.font.SysFont("monospace", 9)

    brain = Brain(device=device)
    if os.path.exists(wm_path):
        brain.load_checkpoint(wm_path=wm_path, pol_path="nonexistent.pt")
        print(f"Loaded World Model from {wm_path}")
    else:
        print(f"Warning: World Model checkpoint {wm_path} not found, using initialized weights.")

    buf = ExperienceBuffer(capacity=1000, pool_capacity=1000)
    if os.path.exists(buffer_path):
        buf.load(buffer_path)
        print(f"Loaded {len(buf._coreset)} sequences from {buffer_path}")
    else:
        print(f"Warning: Buffer checkpoint {buffer_path} not found.")

    theater = WMTheater(brain, buf, width=width, height=height)
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
            theater.handle_event(ev)

        theater.draw(screen, font, font_small, font_tiny)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="World Model Theater GUI")
    parser.add_argument("--buffer", type=str, default="buf_ckpt.pt",
                        help="Path to ExperienceBuffer checkpoint (default: buf_ckpt.pt)")
    parser.add_argument("--wm", type=str, default="wm_ckpt.pt",
                        help="Path to WorldModel checkpoint (default: wm_ckpt.pt)")
    parser.add_argument("--index", type=int, default=0,
                        help="Initial sequence index to display (default: 0)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device (default: cpu)")
    args = parser.parse_args()

    run_theater(
        buffer_path=args.buffer,
        wm_path=args.wm,
        initial_index=args.index,
        device=args.device,
    )


if __name__ == "__main__":
    main()
