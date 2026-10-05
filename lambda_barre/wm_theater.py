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
import time
from dataclasses import dataclass, field

import pygame
import torch

from . import body as B
from .brain import Brain, ExperienceBuffer, sequence_relief, compute_context_time_weights
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
FONT_NAME = "dejavusansmono,ubuntumono,liberationmono,monospace"
FONT_AA = False


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
    real_relief: float = 0.0
    pred_relief: float = 0.0
    early_cost: float = 0.0
    late_cost_real: float = 0.0
    late_cost_pred: float = 0.0


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
                     seq_idx: int, total_seqs: int, vivacity: float,
                     lookahead: bool = False) -> WMTheaterRecord:
    """Perform conditioning on s0..s4a4 and autoregressive rollout for s5..s10,
    or counterfactual hindsight imagination under forced terminal EOS when lookahead=True."""
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

    if not lookahead:
        # Standard teacher-forced autoregressive rollout
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

        # Denouement / Relief metrics
        n_ctx_cutoff = n_ctx - 1
        real_relief = sequence_relief(seq, n_ctx=n_ctx_cutoff)
        pred_seq = [s.real_salve for s in steps[:n_ctx]] + [s.pred_salve for s in steps[n_ctx:]]
        pred_relief = sequence_relief(pred_seq, n_ctx=n_ctx_cutoff)

        early_weights = compute_context_time_weights(n_ctx)
        early_cost = sum(w * sum(s.real_costs) for w, s in zip(early_weights, steps[:n_ctx]))
        late_cost_real = sum(sum(s.real_costs) for s in steps[n_ctx:]) / max(1, len(steps) - n_ctx)
        late_cost_pred = sum(sum(s.pred_costs or [0.0] * 5) for s in steps[n_ctx:]) / max(1, len(steps) - n_ctx)

    else:
        # Counterfactual lookahead imagination:
        # 1. Baseline rollout with recorded a4 -> eos_pi
        # 2. Forced favorable terminal EOS -> eos_better
        # 3. Hindsight action a4* generated by WM conditioned on [ctx, eos_better]
        # 4. Closed-loop rollout with Policy reaction for s5*..s10*
        res = brain.rollout_lookahead_from_sequence(seq)
        steps_gen = res["steps_gen"]
        gen_toks = res["gen_toks"]
        gen_action = res["gen_action"]

        # Steps 0..3: pure context
        for i in range(4):
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

        # Step 4: context state s4 with lookahead generated hindsight action a4*
        a4_star_action = gen_action[0].tolist()
        a4_star_toks = gen_toks[0].tolist()
        pred_salve_4 = seq[4][:13] + a4_star_toks
        pred_step_4 = DreamStep(state_tokens=real_steps[4].state_tokens,
                                action=a4_star_action,
                                label="s4+a4*")
        steps.append(WMStepData(
            step_idx=4,
            real_step=real_steps[4],
            pred_step=pred_step_4,
            real_costs=real_costs_list[4],
            pred_costs=real_costs_list[4],
            posture_rmse=0.0,
            cost_rmse=0.0,
            real_salve=seq[4],
            pred_salve=pred_salve_4,
        ))

        # Steps 5..10: consequence rollout
        pred_posture_errors = []
        pred_cost_errors = []
        for k in range(5, num_salves):
            g_idx = k - 4  # steps_gen[1..6]
            g_step = steps_gen[g_idx]
            pred_s_toks = g_step.state_tokens
            pred_costs = [float(c) for c in pred_s_toks[10][SIG_OFFSET : SIG_OFFSET + 5]]
            p_rmse = compute_posture_rmse(real_steps[k].state_tokens, pred_s_toks)
            c_rmse = compute_cost_rmse(real_costs_list[k], pred_costs)
            pred_posture_errors.append(p_rmse)
            pred_cost_errors.append(c_rmse)

            if k < num_salves - 1:
                act_vals = g_step.action
                act_toks = brain._encode_action_batch(torch.tensor([act_vals], device=device))[0].tolist()
                pred_salve_toks = pred_s_toks + act_toks
                pred_step = DreamStep(state_tokens=pred_s_toks, action=act_vals, label=f"s{k}*+a{k}*")
            else:
                pred_salve_toks = pred_s_toks
                pred_step = DreamStep(state_tokens=pred_s_toks, action=None, label=f"s{k}* (EOS)")

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

        n_ctx_cutoff = 4
        real_relief = sequence_relief(seq, n_ctx=n_ctx_cutoff)
        pred_seq = [s.real_salve for s in steps[:4]] + [steps[4].pred_salve] + [s.pred_salve for s in steps[5:]]
        pred_relief = sequence_relief(pred_seq, n_ctx=n_ctx_cutoff)

        early_weights = compute_context_time_weights(5)
        early_cost = sum(w * sum(s.real_costs) for w, s in zip(early_weights, steps[:5]))
        late_cost_real = sum(sum(s.real_costs) for s in steps[5:]) / max(1, len(steps) - 5)
        late_cost_pred = sum(sum(s.pred_costs or [0.0] * 5) for s in steps[5:]) / max(1, len(steps) - 5)

    return WMTheaterRecord(
        seq_idx=seq_idx,
        total_seqs=total_seqs,
        vivacity=vivacity,
        steps=steps,
        overall_posture_rmse=avg_p_rmse,
        overall_cost_rmse=avg_c_rmse,
        real_relief=real_relief,
        pred_relief=pred_relief,
        early_cost=early_cost,
        late_cost_real=late_cost_real,
        late_cost_pred=late_cost_pred,
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
        s_title = font_tiny.render(title, FONT_AA, TEXT_MUTED)
        surface.blit(s_title, (rx + 4, ry + 2))

    # Total cost sum
    total_val = sum(costs)
    s_tot = font_tiny.render(f"Σ {total_val:.2f}", FONT_AA, TEXT_WHITE if total_val > 0.05 else TEXT_MUTED)
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
        s_lbl = font_tiny.render(lbl, FONT_AA, TEXT_MUTED)
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
    s_title = font_small.render(title, FONT_AA, (255, 215, 80))
    surface.blit(s_title, (px + 10, py + 6))
    s_cnt = font_tiny.render(f"{len(salve)} tokens", FONT_AA, TEXT_MUTED)
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
        s_tag = font_tiny.render(tag, FONT_AA, tag_c)
        surface.blit(s_tag, (tx, ty))

        s_body = font_tiny.render(body, FONT_AA, (215, 222, 235))
        surface.blit(s_body, (tx + s_tag.get_width() + 4, ty))


class WMTheater:
    """Standalone interactive visualizer for World Model inference."""

    def __init__(self, brain: Brain, buffer: ExperienceBuffer,
                 width: int = 1140, height: int = 700,
                 initial_sort: str = "relief_desc"):
        self.brain = brain
        self.buffer = buffer
        self.encoder = DenseEncoder()
        self.width = width
        self.height = height

        self.num_seqs = len(self.buffer._coreset)
        self.show_lookahead: bool = False
        self.cache: dict[tuple[int, bool], WMTheaterRecord] = {}

        # Compute reliefs across all coreset sequences for instant sorting
        n_ctx_cutoff = getattr(brain, "n_ctx", 4)
        self.all_reliefs: list[float] = [
            sequence_relief(seq, n_ctx=n_ctx_cutoff)
            for seq in self.buffer._coreset
        ]

        # Sort mode: "relief_desc" (Recovery first), "relief_asc" (Falls first),
        #            "abs_desc" (Extremes first), "coreset" (Raw order)
        valid_sorts = ("relief_desc", "relief_asc", "abs_desc", "coreset")
        self.sort_mode = initial_sort if initial_sort in valid_sorts else "relief_desc"
        self.sorted_indices: list[int] = []
        self.sorted_idx = 0
        self.seq_idx = 0

        # Slider interaction with debouncing
        self.slider_rect = pygame.Rect(98, 48, 310, 16)
        self.slider_dragging = False
        self.slider_preview_idx = 0
        self.last_slider_motion_time = 0.0
        self.pending_slider_idx: int | None = None

        self._update_sorted_indices()

        # Navigation buttons
        self.btn_prev = pygame.Rect(16, 42, 76, 28)
        self.btn_next = pygame.Rect(414, 42, 76, 28)
        self.btn_lookahead = pygame.Rect(496, 42, 94, 28)

        # Sort buttons
        self.btn_sort_relief_desc = pygame.Rect(596, 42, 126, 28)
        self.btn_sort_relief_asc = pygame.Rect(728, 42, 126, 28)
        self.btn_sort_abs_desc = pygame.Rect(860, 42, 126, 28)
        self.btn_sort_coreset = pygame.Rect(992, 42, 132, 28)

        # Dimensions for columns
        self.col_w = 94
        self.col_gap = 6
        self.margin_x = (self.width - (11 * self.col_w + 10 * self.col_gap)) // 2

        # Precompute initial sequence
        if self.sorted_indices:
            self._load_record(self.sorted_indices[self.sorted_idx])

    def toggle_lookahead(self) -> None:
        """Toggle between teacher-forced autoregressive rollout and lookahead imagination."""
        self.show_lookahead = not self.show_lookahead
        self._load_record(self.seq_idx)

    def _update_sorted_indices(self) -> None:
        if self.sort_mode == "relief_desc":
            self.sorted_indices = sorted(range(self.num_seqs), key=lambda i: self.all_reliefs[i], reverse=True)
        elif self.sort_mode == "relief_asc":
            self.sorted_indices = sorted(range(self.num_seqs), key=lambda i: self.all_reliefs[i])
        elif self.sort_mode == "abs_desc":
            self.sorted_indices = sorted(range(self.num_seqs), key=lambda i: abs(self.all_reliefs[i]), reverse=True)
        else:  # "coreset"
            self.sorted_indices = list(range(self.num_seqs))

        if not self.sorted_indices:
            self.sorted_idx = 0
            self.seq_idx = 0
        else:
            if self.seq_idx in self.sorted_indices:
                self.sorted_idx = self.sorted_indices.index(self.seq_idx)
            else:
                self.sorted_idx = 0
                self.seq_idx = self.sorted_indices[0]
        self.slider_preview_idx = self.sorted_idx

    def set_sort(self, mode: str) -> None:
        valid_sorts = ("relief_desc", "relief_asc", "abs_desc", "coreset")
        if mode not in valid_sorts or mode == self.sort_mode:
            return
        self.sort_mode = mode
        self._update_sorted_indices()
        if self.sorted_indices:
            self._load_record(self.sorted_indices[self.sorted_idx])

    def cycle_sort(self) -> None:
        modes = ["relief_desc", "relief_asc", "abs_desc", "coreset"]
        nxt = modes[(modes.index(self.sort_mode) + 1) % len(modes)]
        self.set_sort(nxt)

    def _load_record(self, idx: int) -> WMTheaterRecord:
        idx = max(0, min(self.num_seqs - 1, idx))
        cache_key = (idx, self.show_lookahead)
        if cache_key not in self.cache:
            seq = self.buffer._coreset[idx]
            viv = (self.buffer._coreset_vivacity[idx]
                   if idx < len(self.buffer._coreset_vivacity) else 1.0)
            self.cache[cache_key] = rollout_sequence(
                self.brain, seq, idx, self.num_seqs, viv, lookahead=self.show_lookahead)
        return self.cache[cache_key]

    def set_sorted_idx(self, idx: int) -> None:
        if not self.sorted_indices:
            return
        self.sorted_idx = max(0, min(len(self.sorted_indices) - 1, idx))
        self.slider_preview_idx = self.sorted_idx
        self.seq_idx = self.sorted_indices[self.sorted_idx]
        self._load_record(self.seq_idx)

    def set_seq_idx(self, idx: int) -> None:
        if self.num_seqs <= 0:
            return
        idx = max(0, min(self.num_seqs - 1, idx))
        if idx in self.sorted_indices:
            self.set_sorted_idx(self.sorted_indices.index(idx))
        else:
            self.seq_idx = idx
            self.set_sort("coreset")
            self.set_sorted_idx(idx)

    def prev_seq(self) -> None:
        if not self.sorted_indices:
            return
        self.set_sorted_idx((self.sorted_idx - 1) % len(self.sorted_indices))

    def next_seq(self) -> None:
        if not self.sorted_indices:
            return
        self.set_sorted_idx((self.sorted_idx + 1) % len(self.sorted_indices))

    def random_seq(self) -> None:
        if self.sorted_indices:
            self.set_sorted_idx(random.randint(0, len(self.sorted_indices) - 1))

    def handle_event(self, ev: pygame.event.Event) -> bool:
        """Handle keyboard and mouse events. Returns True if handled."""
        if ev.type == pygame.KEYDOWN:
            if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and (ev.mod & pygame.KMOD_ALT or pygame.key.get_mods() & pygame.KMOD_ALT):
                pygame.display.toggle_fullscreen()
                return True
            elif ev.key in (pygame.K_LEFT, pygame.K_a):
                self.prev_seq()
                return True
            elif ev.key in (pygame.K_RIGHT, pygame.K_d):
                self.next_seq()
                return True
            elif ev.key == pygame.K_s:
                self.cycle_sort()
                return True
            elif ev.key == pygame.K_l:
                self.toggle_lookahead()
                return True
            elif ev.key == pygame.K_HOME:
                self.set_sorted_idx(0)
                return True
            elif ev.key == pygame.K_END:
                if self.sorted_indices:
                    self.set_sorted_idx(len(self.sorted_indices) - 1)
                return True

        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            mx, my = ev.pos
            if self.btn_prev.collidepoint(mx, my):
                self.prev_seq()
                return True
            elif self.btn_next.collidepoint(mx, my):
                self.next_seq()
                return True
            elif self.btn_lookahead.collidepoint(mx, my):
                self.toggle_lookahead()
                return True
            elif self.btn_sort_relief_desc.collidepoint(mx, my):
                self.set_sort("relief_desc")
                return True
            elif self.btn_sort_relief_asc.collidepoint(mx, my):
                self.set_sort("relief_asc")
                return True
            elif self.btn_sort_abs_desc.collidepoint(mx, my):
                self.set_sort("abs_desc")
                return True
            elif self.btn_sort_coreset.collidepoint(mx, my):
                self.set_sort("coreset")
                return True
            elif (self.slider_rect.x - 8 <= mx <= self.slider_rect.x + self.slider_rect.width + 8
                  and self.slider_rect.y - 10 <= my <= self.slider_rect.y + self.slider_rect.height + 10):
                self.slider_dragging = True
                t = max(0.0, min(1.0, (mx - self.slider_rect.x) / self.slider_rect.width))
                self.slider_preview_idx = int(round(t * (self.num_seqs - 1)))
                self.last_slider_motion_time = time.time()
                self.pending_slider_idx = self.slider_preview_idx
                if self.sorted_indices:
                    cand_seq = self.sorted_indices[self.slider_preview_idx]
                    if (cand_seq, self.show_lookahead) in self.cache:
                        self.set_sorted_idx(self.slider_preview_idx)
                        self.pending_slider_idx = None
                return True

        elif ev.type == pygame.MOUSEMOTION:
            if self.slider_dragging:
                t = max(0.0, min(1.0, (ev.pos[0] - self.slider_rect.x) / self.slider_rect.width))
                new_idx = int(round(t * (self.num_seqs - 1)))
                if new_idx != self.slider_preview_idx:
                    self.slider_preview_idx = new_idx
                    self.last_slider_motion_time = time.time()
                    self.pending_slider_idx = new_idx
                    if self.sorted_indices:
                        cand_seq = self.sorted_indices[self.slider_preview_idx]
                        if (cand_seq, self.show_lookahead) in self.cache:
                            self.set_sorted_idx(self.slider_preview_idx)
                            self.pending_slider_idx = None
                return True

        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            if self.slider_dragging:
                self.slider_dragging = False
                commit_idx = self.slider_preview_idx
                self.pending_slider_idx = None
                self.set_sorted_idx(commit_idx)
                return True

        elif ev.type == pygame.MOUSEWHEEL:
            if ev.y > 0:
                self.prev_seq()
            elif ev.y < 0:
                self.next_seq()
            return True

        return False

    def draw(self, screen: pygame.Surface, font: pygame.font.Font,
             font_small: pygame.font.Font, font_tiny: pygame.font.Font) -> None:
        """Render the complete World Model Theater storyboard."""
        # Debounce check while dragging: auto-commit if paused >= 150 ms
        if self.pending_slider_idx is not None and (time.time() - self.last_slider_motion_time >= 0.15):
            idx_to_load = self.pending_slider_idx
            self.pending_slider_idx = None
            self.set_sorted_idx(idx_to_load)

        screen.fill(DREAM_BG)

        if self.num_seqs == 0:
            msg = font.render("Coreset is empty — no sequences to display.", FONT_AA, TEXT_WHITE)
            screen.blit(msg, (self.width // 2 - msg.get_width() // 2, self.height // 2))
            return

        header_h = 80
        pygame.draw.rect(screen, HEADER_BG, (0, 0, self.width, header_h))
        pygame.draw.line(screen, CARD_BORDER, (0, header_h), (self.width, header_h), 1)

        # Prev button
        pygame.draw.rect(screen, (34, 40, 54), self.btn_prev, border_radius=4)
        pygame.draw.rect(screen, CARD_BORDER, self.btn_prev, 1, border_radius=4)
        s_prev = font_small.render("< Prev [←]", FONT_AA, TEXT_WHITE)
        screen.blit(s_prev, (self.btn_prev.x + (self.btn_prev.width - s_prev.get_width()) // 2,
                             self.btn_prev.y + 6))

        # Slider track
        pygame.draw.rect(screen, (28, 32, 44), self.slider_rect, border_radius=4)
        pygame.draw.rect(screen, CARD_BORDER, self.slider_rect, 1, border_radius=4)

        # Slider fill and handle (uses slider_preview_idx if dragging, else sorted_idx)
        active_slider_idx = self.slider_preview_idx if self.slider_dragging else self.sorted_idx
        ratio = (active_slider_idx / max(1, self.num_seqs - 1)) if self.num_seqs > 1 else 0.0
        fill_w = int(ratio * self.slider_rect.width)
        if fill_w > 0:
            fill_rect = pygame.Rect(self.slider_rect.x, self.slider_rect.y, fill_w, self.slider_rect.height)
            pygame.draw.rect(screen, (50, 110, 190), fill_rect, border_radius=4)

        # Slider handle
        hx = self.slider_rect.x + fill_w
        hy = self.slider_rect.centery
        handle_color = (255, 230, 90) if self.slider_dragging else (110, 200, 255)
        pygame.draw.circle(screen, handle_color, (hx, hy), 7)
        pygame.draw.circle(screen, TEXT_WHITE, (hx, hy), 7, 2)

        # Next button
        pygame.draw.rect(screen, (34, 40, 54), self.btn_next, border_radius=4)
        pygame.draw.rect(screen, CARD_BORDER, self.btn_next, 1, border_radius=4)
        s_next = font_small.render("Next [→] >", FONT_AA, TEXT_WHITE)
        screen.blit(s_next, (self.btn_next.x + (self.btn_next.width - s_next.get_width()) // 2,
                             self.btn_next.y + 6))

        # Lookahead toggle button
        la_active = self.show_lookahead
        bg = (48, 28, 64) if la_active else (34, 40, 54)
        border_c = (185, 110, 245) if la_active else CARD_BORDER
        txt_c = (225, 175, 255) if la_active else TEXT_WHITE
        border_w = 2 if la_active else 1
        pygame.draw.rect(screen, bg, self.btn_lookahead, border_radius=4)
        pygame.draw.rect(screen, border_c, self.btn_lookahead, border_w, border_radius=4)
        btn_txt = "★ Lookahead [L]" if la_active else "Rollout [L]"
        s_la = font_small.render(btn_txt, FONT_AA, txt_c)
        screen.blit(s_la, (self.btn_lookahead.x + (self.btn_lookahead.width - s_la.get_width()) // 2,
                           self.btn_lookahead.y + 6))

        # Sort buttons
        # 1. Relief ↓ (Recovery first)
        is_r_desc = (self.sort_mode == "relief_desc")
        bg = (24, 60, 38) if is_r_desc else (30, 36, 48)
        border = (80, 220, 130) if is_r_desc else CARD_BORDER
        pygame.draw.rect(screen, bg, self.btn_sort_relief_desc, border_radius=4)
        pygame.draw.rect(screen, border, self.btn_sort_relief_desc, 2 if is_r_desc else 1, border_radius=4)
        s_lbl = font_small.render("★ Relief ↓", FONT_AA, (100, 245, 160) if is_r_desc else TEXT_MUTED)
        screen.blit(s_lbl, (self.btn_sort_relief_desc.x + (self.btn_sort_relief_desc.width - s_lbl.get_width()) // 2,
                            self.btn_sort_relief_desc.y + 6))

        # 2. Relief ↑ (Falls first)
        is_r_asc = (self.sort_mode == "relief_asc")
        bg = (60, 24, 30) if is_r_asc else (30, 36, 48)
        border = (240, 90, 100) if is_r_asc else CARD_BORDER
        pygame.draw.rect(screen, bg, self.btn_sort_relief_asc, border_radius=4)
        pygame.draw.rect(screen, border, self.btn_sort_relief_asc, 2 if is_r_asc else 1, border_radius=4)
        s_lbl = font_small.render("⚠ Relief ↑", FONT_AA, (255, 120, 130) if is_r_asc else TEXT_MUTED)
        screen.blit(s_lbl, (self.btn_sort_relief_asc.x + (self.btn_sort_relief_asc.width - s_lbl.get_width()) // 2,
                            self.btn_sort_relief_asc.y + 6))

        # 3. |Relief| ↓ (Extremes first)
        is_abs = (self.sort_mode == "abs_desc")
        bg = (60, 52, 20) if is_abs else (30, 36, 48)
        border = (255, 210, 70) if is_abs else CARD_BORDER
        pygame.draw.rect(screen, bg, self.btn_sort_abs_desc, border_radius=4)
        pygame.draw.rect(screen, border, self.btn_sort_abs_desc, 2 if is_abs else 1, border_radius=4)
        s_lbl = font_small.render("⚡ |Relief| ↓", FONT_AA, (255, 225, 90) if is_abs else TEXT_MUTED)
        screen.blit(s_lbl, (self.btn_sort_abs_desc.x + (self.btn_sort_abs_desc.width - s_lbl.get_width()) // 2,
                            self.btn_sort_abs_desc.y + 6))

        # 4. Coreset (Raw)
        is_core = (self.sort_mode == "coreset")
        bg = (38, 56, 82) if is_core else (30, 36, 48)
        border = (110, 200, 255) if is_core else CARD_BORDER
        pygame.draw.rect(screen, bg, self.btn_sort_coreset, border_radius=4)
        pygame.draw.rect(screen, border, self.btn_sort_coreset, 2 if is_core else 1, border_radius=4)
        s_lbl = font_small.render("Coreset (Raw)", FONT_AA, TEXT_WHITE if is_core else TEXT_MUTED)
        screen.blit(s_lbl, (self.btn_sort_coreset.x + (self.btn_sort_coreset.width - s_lbl.get_width()) // 2,
                            self.btn_sort_coreset.y + 6))

        rec = self._load_record(self.seq_idx)

        # Header Title (Line 1 left)
        disp_idx = (self.slider_preview_idx + 1) if self.slider_dragging else (self.sorted_idx + 1)
        title_str = (
            f"WORLD MODEL THEATER  |  Seq {disp_idx}/{self.num_seqs} (#{rec.seq_idx + 1})  |  "
            f"Viv: {rec.vivacity:.2f}"
        )
        s_title = font.render(title_str, FONT_AA, TEXT_WHITE)
        screen.blit(s_title, (16, 12))

        # DENOUEMENT PROMINENT BANNER (Line 1 right)
        banner_rect = pygame.Rect(506, 8, 618, 26)
        is_recov = rec.real_relief > 0.005
        is_worse = rec.real_relief < -0.005
        b_bg = (18, 46, 32) if is_recov else ((46, 18, 24) if is_worse else (26, 30, 42))
        b_border = (60, 200, 110) if is_recov else ((220, 70, 80) if is_worse else (65, 75, 95))
        pygame.draw.rect(screen, b_bg, banner_rect, border_radius=4)
        pygame.draw.rect(screen, b_border, banner_rect, 1, border_radius=4)

        # Badge tag
        tag_text = "★ RECOVERY" if is_recov else ("⚠ WORSENING" if is_worse else "● NEUTRAL")
        tag_c = (100, 245, 160) if is_recov else ((255, 120, 130) if is_worse else (180, 190, 210))
        s_btag = font_small.render(f"{tag_text} ({rec.real_relief:+.3f})", FONT_AA, tag_c)
        screen.blit(s_btag, (banner_rect.x + 8, banner_rect.y + 6))

        # Details in banner
        cost_diff_pct = ((rec.late_cost_real - rec.early_cost) / max(0.01, rec.early_cost)) * 100.0
        details_str = (
            f"Early Cost {rec.early_cost:.2f} ──▶ Late {rec.late_cost_real:.2f} ({cost_diff_pct:+.0f}%)   |   "
            f"Pred Relief: {rec.pred_relief:+.3f} (Δ {abs(rec.real_relief - rec.pred_relief):.3f})"
        )
        s_bdet = font_tiny.render(details_str, FONT_AA, TEXT_WHITE)
        screen.blit(s_bdet, (banner_rect.x + s_btag.get_width() + 14, banner_rect.y + 7))

        # 2. Section 1: Ground Truth Sequence (Coreset)
        sec1_y = header_h + 10
        s_sec1 = font.render("GROUND TRUTH", FONT_AA, TEXT_WHITE)
        screen.blit(s_sec1, (self.margin_x, sec1_y))
        s_ctx_lbl = font_small.render("[ Context s0..s4 ]", FONT_AA, TEXT_MUTED)
        screen.blit(s_ctx_lbl, (self.margin_x + s_sec1.get_width() + 8, sec1_y + 2))

        den_x = self.margin_x + 5 * (self.col_w + self.col_gap)
        s_den_lbl = font.render(
            f"CONSEQUENCE & DENOUEMENT (s5..s10) ──▶  Relief: {rec.real_relief:+.3f}",
            FONT_AA, tag_c)
        screen.blit(s_den_lbl, (den_x + 6, sec1_y))

        row1_y = sec1_y + 22
        card_h = 246

        # Draw vertical context-denouement divider line between column 4 and column 5
        div_x = self.margin_x + 5 * self.col_w + 4 * self.col_gap + self.col_gap // 2
        pygame.draw.line(screen, (55, 65, 85), (div_x, row1_y), (div_x, row1_y + card_h), 1)

        hover_card_info: tuple[str, list[list[float]], pygame.Rect] | None = None
        mouse_pos = pygame.mouse.get_pos()

        for i, step_data in enumerate(rec.steps):
            cx = self.margin_x + i * (self.col_w + self.col_gap)
            card_rect = pygame.Rect(cx, row1_y, self.col_w, card_h)

            is_context = (i < 5)
            is_eos = (i == len(rec.steps) - 1)
            is_hovered = card_rect.collidepoint(mouse_pos)
            if is_hovered:
                lbl_text = f"s{i}+a{i}" if not is_eos else "s10 (EOS)"
                hover_card_info = (f"REAL SALVE {lbl_text}", step_data.real_salve, card_rect)

            if is_eos:
                bg = (20, 38, 28) if is_recov else ((38, 20, 24) if is_worse else CARD_BG)
                border_c = (255, 255, 255) if is_hovered else ((80, 220, 130) if is_recov else ((240, 80, 90) if is_worse else CARD_BORDER))
                border_w = 2
            else:
                bg = CONTEXT_BG if is_context else CARD_BG
                border_c = (140, 210, 255) if is_hovered else (CONTEXT_BORDER if is_context else CARD_BORDER)
                border_w = 2 if is_hovered else 1

            pygame.draw.rect(screen, bg, card_rect, border_radius=4)
            pygame.draw.rect(screen, border_c, card_rect, border_w, border_radius=4)

            # Card Header label
            if is_eos:
                lbl_text = "★ REAL EOS"
                c_hdr = tag_c
            else:
                lbl_text = f"s{i}+a{i}" if i < 10 else "s10"
                c_hdr = TEXT_ACCENT if is_context else TEXT_WHITE
            s_hdr = font_small.render(lbl_text, FONT_AA, c_hdr)
            screen.blit(s_hdr, (cx + (self.col_w - s_hdr.get_width()) // 2, row1_y + 4))

            # Animal thumbnail
            th_rect = pygame.Rect(cx + 4, row1_y + 20, self.col_w - 8, 70)
            cand_c = CAND_COLORS[0] if step_data.real_step.action is not None else None
            draw_thumbnail(screen, th_rect, step_data.real_step, font_tiny, facing=1,
                           cand_color=cand_c, highlight_label="")

            # Instantaneous Costs Mini-Histogram
            cost_rect = pygame.Rect(cx + 4, row1_y + 96, self.col_w - 8, 142)
            hist_title = "EOS" if is_eos else "Real"
            draw_cost_histogram(screen, cost_rect, step_data.real_costs, font_tiny,
                                title=hist_title, is_predicted=False)

        # 3. Section 2: World Model Inférence (Rollout après s4a4 ou Lookahead imagination)
        sec2_y = row1_y + card_h + 12
        if self.show_lookahead:
            s_sec2 = font.render("LOOKAHEAD IMAGINATION", FONT_AA, (185, 130, 255))
            screen.blit(s_sec2, (self.margin_x, sec2_y))
            s_ctx2_lbl = font_small.render("[ Context s0..s3 | Hindsight a4* ]", FONT_AA, TEXT_MUTED)
            screen.blit(s_ctx2_lbl, (self.margin_x + s_sec2.get_width() + 8, sec2_y + 2))

            pred_tag_c = (185, 130, 255) if rec.pred_relief > 0 else (240, 110, 120)
            s_roll_lbl = font.render(
                f"HINDSIGHT (s5*..s10*) ──▶ Relief: {rec.pred_relief:+.3f}  |  Post RMSE: {rec.overall_posture_rmse:.3f}",
                FONT_AA, pred_tag_c)
            screen.blit(s_roll_lbl, (den_x + 6, sec2_y))
        else:
            s_sec2 = font.render("WORLD MODEL", FONT_AA, WINNER_BORDER)
            screen.blit(s_sec2, (self.margin_x, sec2_y))
            s_ctx2_lbl = font_small.render("[ Real Context s0..s4 ]", FONT_AA, TEXT_MUTED)
            screen.blit(s_ctx2_lbl, (self.margin_x + s_sec2.get_width() + 8, sec2_y + 2))

            pred_tag_c = (90, 200, 255) if rec.pred_relief > 0 else (240, 110, 120)
            s_roll_lbl = font.render(
                f"ROLLOUT (s5^..s10^) ──▶ Pred: {rec.pred_relief:+.3f}  |  Post RMSE: {rec.overall_posture_rmse:.3f}",
                FONT_AA, pred_tag_c)
            screen.blit(s_roll_lbl, (den_x + 6, sec2_y))

        row2_y = sec2_y + 22
        pygame.draw.line(screen, (55, 65, 85), (div_x, row2_y), (div_x, row2_y + card_h), 1)

        for i, step_data in enumerate(rec.steps):
            cx = self.margin_x + i * (self.col_w + self.col_gap)
            card_rect = pygame.Rect(cx, row2_y, self.col_w, card_h)

            is_pure_context = (i < 4) if self.show_lookahead else (i < 5)
            is_hindsight_a4 = (self.show_lookahead and i == 4)
            is_eos = (i == len(rec.steps) - 1)
            is_hovered = card_rect.collidepoint(mouse_pos)

            if is_pure_context:
                if is_hovered:
                    hover_card_info = (f"CONTEXT SALVE s{i}", step_data.real_salve, card_rect)

                border_c = (140, 210, 255) if is_hovered else CONTEXT_BORDER
                pygame.draw.rect(screen, CONTEXT_BG, card_rect, border_radius=4)
                pygame.draw.rect(screen, border_c, card_rect, 2 if is_hovered else 1, border_radius=4)

                s_hdr = font_small.render(f"[Context s{i}]", FONT_AA, TEXT_MUTED)
                screen.blit(s_hdr, (cx + (self.col_w - s_hdr.get_width()) // 2, row2_y + 4))

                th_rect = pygame.Rect(cx + 4, row2_y + 20, self.col_w - 8, 70)
                draw_thumbnail(screen, th_rect, step_data.real_step, font_tiny, facing=1,
                               cand_color=None, highlight_label="")

                cost_rect = pygame.Rect(cx + 4, row2_y + 96, self.col_w - 8, 142)
                draw_cost_histogram(screen, cost_rect, step_data.real_costs, font_tiny,
                                    title="Context", is_predicted=False)

            elif is_hindsight_a4:
                lbl_text = "s4+a4*"
                if is_hovered:
                    hover_card_info = (f"HINDSIGHT SALVE {lbl_text}", step_data.pred_salve or [], card_rect)

                bg = (34, 26, 46)
                border_c = (220, 160, 255) if is_hovered else (185, 110, 245)
                pygame.draw.rect(screen, bg, card_rect, border_radius=4)
                pygame.draw.rect(screen, border_c, card_rect, 2 if is_hovered else 1, border_radius=4)

                s_hdr = font_small.render("★ a4* [Hindsight]", FONT_AA, (220, 160, 255))
                screen.blit(s_hdr, (cx + (self.col_w - s_hdr.get_width()) // 2, row2_y + 4))

                th_rect = pygame.Rect(cx + 4, row2_y + 20, self.col_w - 8, 70)
                if step_data.pred_step:
                    draw_thumbnail(screen, th_rect, step_data.pred_step, font_tiny, facing=1,
                                   cand_color=CAND_COLORS[0], is_winner=True, highlight_label="")

                cost_rect = pygame.Rect(cx + 4, row2_y + 96, self.col_w - 8, 116)
                draw_cost_histogram(screen, cost_rect, step_data.pred_costs or [0.0]*5, font_tiny,
                                    title="Context", is_predicted=True)

                s_err1 = font_tiny.render("Hindsight Action", FONT_AA, (200, 150, 255))
                s_err2 = font_tiny.render("a4* (forced EOS)", FONT_AA, TEXT_WHITE)
                screen.blit(s_err1, (cx + 6, row2_y + 214))
                screen.blit(s_err2, (cx + 6, row2_y + 228))

            else:
                lbl_text = "★ PRED EOS" if is_eos else (f"s{i}*+a{i}*" if self.show_lookahead else f"s{i}^+a{i}")
                if is_hovered:
                    tag_prefix = "LOOKAHEAD" if self.show_lookahead else "PREDICTED"
                    hover_card_info = (f"{tag_prefix} SALVE {lbl_text}", step_data.pred_salve or [], card_rect)

                if is_eos:
                    bg = (20, 34, 42) if rec.pred_relief > 0 else (38, 22, 26)
                    border_c = (255, 255, 255) if is_hovered else ((90, 200, 255) if rec.pred_relief > 0 else (240, 110, 120))
                    border_w = 2
                else:
                    bg = (28, 24, 38) if self.show_lookahead else (24, 30, 42)
                    border_c = (200, 140, 255) if (is_hovered and self.show_lookahead) else (WINNER_BORDER if is_hovered else ((160, 110, 220) if self.show_lookahead else PRED_BORDER))
                    border_w = 2 if is_hovered else 1

                pygame.draw.rect(screen, bg, card_rect, border_radius=4)
                pygame.draw.rect(screen, border_c, card_rect, border_w, border_radius=4)

                default_hdr_c = (210, 160, 255) if self.show_lookahead else WINNER_BORDER
                hdr_color = pred_tag_c if is_eos else default_hdr_c
                s_hdr = font_small.render(lbl_text, FONT_AA, hdr_color)
                screen.blit(s_hdr, (cx + (self.col_w - s_hdr.get_width()) // 2, row2_y + 4))

                th_rect = pygame.Rect(cx + 4, row2_y + 20, self.col_w - 8, 70)
                cand_c = CAND_COLORS[0] if (step_data.pred_step and step_data.pred_step.action is not None) else None
                if step_data.pred_step:
                    draw_thumbnail(screen, th_rect, step_data.pred_step, font_tiny, facing=1,
                                   cand_color=cand_c, is_winner=True, highlight_label="")

                cost_rect = pygame.Rect(cx + 4, row2_y + 96, self.col_w - 8, 116)
                hist_title = "EOS" if is_eos else ("Imagined" if self.show_lookahead else "Pred")
                draw_cost_histogram(screen, cost_rect, step_data.pred_costs or [0.0]*5, font_tiny,
                                    title=hist_title, is_predicted=True)

                p_err = step_data.posture_rmse or 0.0
                c_err = step_data.cost_rmse or 0.0

                c_color = (110, 230, 140) if p_err < 0.04 else ((255, 210, 70) if p_err < 0.08 else (240, 100, 100))
                s_err1 = font_tiny.render(f"Δpost: {p_err:.3f}", FONT_AA, c_color)
                s_err2 = font_tiny.render(f"Δcost: {c_err:.3f}", FONT_AA, TEXT_WHITE)
                screen.blit(s_err1, (cx + 6, row2_y + 214))
                screen.blit(s_err2, (cx + 6, row2_y + 228))

        # 4. Footer & Legend
        footer_y = self.height - 24
        hints = "Nav: [← / →]  |  [S] Sort  |  [L] Lookahead  |  Hover: Tokens  |  [Esc / Q] Quit"
        s_hints = font_small.render(hints, FONT_AA, TEXT_MUTED)
        screen.blit(s_hints, (self.margin_x, footer_y))

        # Costs legend
        leg_x = self.width - self.margin_x - 360
        s_leg_title = font_tiny.render("Costs:", FONT_AA, TEXT_MUTED)
        screen.blit(s_leg_title, (leg_x, footer_y + 1))
        cur_lx = leg_x + 40
        for code, c in zip(COST_LABELS, COST_COLORS):
            pygame.draw.rect(screen, c, (cur_lx, footer_y + 3, 8, 8), border_radius=1)
            s_c = font_tiny.render(code, FONT_AA, TEXT_WHITE)
            screen.blit(s_c, (cur_lx + 11, footer_y + 1))
            cur_lx += 42

        # 5. Pop-in overlay on top of everything
        if hover_card_info is not None:
            pop_title, pop_salve, pop_rect = hover_card_info
            draw_token_popin(screen, self.encoder, font_small, font_tiny,
                             pop_title, pop_salve, pop_rect, self.width, self.height)


def run_theater(buffer_path: str = "buf_ckpt.pt",
                wm_path: str = "wm_ckpt.pt",
                pol_path: str = "pol_ckpt.pt",
                initial_index: int = 0,
                device: str = "cpu",
                initial_sort: str = "relief_desc") -> None:
    os.environ["SDL_HINT_RENDER_SCALE_QUALITY"] = "1"
    try:
        import ctypes
        ctypes.CDLL("libSDL2-2.0.so.0").SDL_SetHint(b"SDL_HINT_RENDER_SCALE_QUALITY", b"1")
    except Exception:
        pass
    pygame.init()
    pygame.display.set_caption("World Model Theater — Convergence Diagnostic")

    width, height = 1140, 700
    flags = pygame.SCALED | pygame.RESIZABLE
    screen = pygame.display.set_mode((width, height), flags)
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(FONT_NAME, 14, bold=True)
    font_small = pygame.font.SysFont(FONT_NAME, 11)
    font_tiny = pygame.font.SysFont(FONT_NAME, 10)

    brain = Brain(device=device)
    if os.path.exists(wm_path) or os.path.exists(pol_path):
        brain.load_checkpoint(
            wm_path=wm_path if os.path.exists(wm_path) else "nonexistent.pt",
            pol_path=pol_path if os.path.exists(pol_path) else "nonexistent.pt",
        )
        if os.path.exists(wm_path):
            print(f"Loaded World Model from {wm_path}")
        if os.path.exists(pol_path):
            print(f"Loaded Policy from {pol_path}")
    else:
        print("Warning: World Model or Policy checkpoints not found, using initialized weights.")

    buf = ExperienceBuffer(capacity=1000, pool_capacity=1000)
    if os.path.exists(buffer_path):
        buf.load(buffer_path)
        print(f"Loaded {len(buf._coreset)} sequences from {buffer_path}")
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
            elif ev.type == pygame.KEYDOWN and ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and (ev.mod & pygame.KMOD_ALT or pygame.key.get_mods() & pygame.KMOD_ALT):
                pygame.display.toggle_fullscreen()
                continue
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
    parser.add_argument("--pol", type=str, default="pol_ckpt.pt",
                        help="Path to Policy checkpoint (default: pol_ckpt.pt)")
    parser.add_argument("--index", type=int, default=0,
                        help="Initial sequence index to display (default: 0)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device (default: cpu)")
    parser.add_argument("--sort", type=str, default="relief_desc",
                        choices=["relief_desc", "relief_asc", "abs_desc", "coreset"],
                        help="Sort mode: 'relief_desc' (default: recovery first), 'relief_asc' (falls first), 'abs_desc' (extremes first), or 'coreset' (raw)")
    args = parser.parse_args()

    run_theater(
        buffer_path=args.buffer,
        wm_path=args.wm,
        pol_path=args.pol,
        initial_index=args.index,
        device=args.device,
        initial_sort=args.sort,
    )


if __name__ == "__main__":
    main()

