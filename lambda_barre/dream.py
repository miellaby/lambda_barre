"""Dream Theater: Visualizer for sleep policy training.

Replaces the physics world during sleep with a storyboard of thumbnails (vignettes):
- Context row: s0+a0, s1+a1, s2+a2, s3+a3, s4 (decision state)
- Future rows: the imagined trajectories of each candidate action
  (hindsight supervision: policy baseline vs EOS-lookahead generation;
  legacy fallback: 3 candidates × 2 future action regimes)
  Each row shows s4+a_candidate followed by imagined future steps (s5+a5, s6+a6, ...).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pygame

from . import body as B
from .tokenize import _BY_KEY, _denormalize, SIG_OFFSET


# Colors
DREAM_BG = (18, 20, 26)
CARD_BG = (28, 32, 42)
CARD_BORDER = (50, 56, 72)
BEST_BORDER = (90, 210, 130)
WINNER_BORDER = (255, 215, 0)
HEADER_BG = (24, 28, 38)
TEXT_WHITE = (230, 235, 245)
TEXT_MUTED = (140, 148, 165)
TEXT_ACCENT = (110, 200, 255)
GROUND_LINE_C = (60, 66, 82)
FONT_AA = False

CAND_COLORS = [
    (90, 190, 255),   # Cand 0: Policy (Cyan)
    (255, 210, 70),   # Cand 1: Replay (Gold)
    (240, 120, 240),  # Cand 2: Noise 30% (Magenta)
]

REGIME_NAMES = ["Kalman", "Policy"]


@dataclass
class DreamStep:
    state_tokens: list[list[float]]  # 13 tokens × 25 floats
    action: list[float] | None        # [tf, df, tb, db, tq]
    step_cost: float = 0.0
    label: str = ""


@dataclass
class DreamTrajectory:
    candidate_idx: int                # 0..2
    regime_idx: int                   # 0..1 (Kalman, Policy)
    regime_name: str                  # "Kalman", "Policy"
    steps: list[DreamStep] = field(default_factory=list)
    total_cost: float = 0.0
    is_best_future: bool = False
    is_winner: bool = False
    eos_pred: list[float] | None = None  # imagined terminal EOS (delta fatigue, delta souffrance), normalized [0, 1]


@dataclass
class DreamRecord:
    step_idx: int                     # Training step index (0..pol_steps-1)
    total_steps: int                  # Total policy steps
    loss: float                       # Policy loss
    facing: int                       # Skeleton facing (+1 or -1)
    context_steps: list[DreamStep]    # [s0+a0, s1+a1, s2+a2, s3+a3, s4]
    candidate_actions: list[list[float]] # candidate actions [n_candidates, 5]
    trajectories: list[DreamTrajectory] # n_candidates * n_regimes trajectories
    best_candidate_idx: int           # Selected candidate index (-1: none accepted)
    best_future_indices: list[int]    # Winning future index per candidate
    candidate_names: list[str] = field(default_factory=lambda: [
        "0: POLICY", "1: REPLAY (DEMO)", "2: NOISE 30%"])
    eos_forced: list[float] | None = None  # hindsight forcing target (delta fatigue, delta souffrance), normalized [0, 1]
    lived_steps: list[DreamStep] | None = None        # full lived sequence s0..s10 with actions and per-salve costs
    predicted_steps: list[DreamStep] | None = None     # real s0..s3 + hindsight a4 + imagined s5..s10
    eos_realized: list[float] | None = None           # realized EOS of the lived sequence, normalized [0, 1]


def decode_state_tokens(st_toks: list[list[float]], facing: int = 1) -> dict:
    """Extract physical posture and signals from 13 state tokens."""
    if hasattr(st_toks, "tolist"):
        st_toks = st_toks.tolist()

    # Token 1: tronc_angle, queue_angle
    t1 = st_toks[1]
    tronc_angle = _denormalize(t1[SIG_OFFSET + 0], _BY_KEY["tronc_angle"])
    queue_angle = _denormalize(t1[SIG_OFFSET + 1], _BY_KEY["queue_angle"])
    torso_angle = -facing * tronc_angle

    # Token 2: front limb
    t2 = st_toks[2]
    angle_f = _denormalize(t2[SIG_OFFSET + 0], _BY_KEY["membre_angle_avant"])
    dist_f = _denormalize(t2[SIG_OFFSET + 1], _BY_KEY["membre_distance_avant"])

    # Token 3: back limb
    t3 = st_toks[3]
    angle_b = _denormalize(t3[SIG_OFFSET + 0], _BY_KEY["membre_angle_arriere"])
    dist_b = _denormalize(t3[SIG_OFFSET + 1], _BY_KEY["membre_distance_arriere"])

    # Token 0: forces
    t0 = st_toks[0]
    force_f = _denormalize(t0[SIG_OFFSET + 0], _BY_KEY["force_actuateur_avant"])
    force_b = _denormalize(t0[SIG_OFFSET + 1], _BY_KEY["force_actuateur_arriere"])

    # Tail absolute angle (neutral is rearward, -facing * pi/2)
    tail_neutral = -facing * (math.pi / 2)
    tail_rel = queue_angle / facing + tail_neutral
    tail_angle = torso_angle + tail_rel

    return {
        "tronc_angle": tronc_angle,
        "torso_angle": torso_angle,
        "tail_angle": tail_angle,
        "tail_rel": tail_rel,
        "queue_angle": queue_angle,
        "angle_f": angle_f,
        "dist_f": dist_f,
        "angle_b": angle_b,
        "dist_b": dist_b,
        "force_f": force_f,
        "force_b": force_b,
    }


def draw_thumbnail(surface: pygame.Surface, rect: pygame.Rect, step: DreamStep,
                   font_small: pygame.font.Font, facing: int = 1,
                   cand_color: tuple[int, int, int] | None = None,
                   is_best: bool = False, is_winner: bool = False,
                   highlight_label: str = "") -> None:
    """Render a single vignette/thumbnail of a state + action target."""
    bx, by, bw, bh = rect.x, rect.y, rect.width, rect.height

    # Card background
    pygame.draw.rect(surface, CARD_BG, rect, border_radius=4)

    # Border
    if is_winner:
        border_c = WINNER_BORDER
        border_w = 2
    elif is_best:
        border_c = BEST_BORDER
        border_w = 2
    else:
        border_c = CARD_BORDER
        border_w = 1
    pygame.draw.rect(surface, border_c, rect, border_w, border_radius=4)

    # Decode posture (purely egocentric frame: front is right, facing is 1)
    info = decode_state_tokens(step.state_tokens, facing=1)
    tronc_angle = info["tronc_angle"]
    tail_angle = info["tail_angle"]

    # Thumbnail center & scale
    cx = bx + bw // 2
    cy = by + bh // 2 - 3
    scale = 0.68  # Fit inside thumbnail

    cos_t, sin_t = math.cos(tronc_angle), math.sin(tronc_angle)

    def rot(lx: float, ly: float) -> tuple[int, int]:
        rx = lx * cos_t + ly * sin_t
        ry = -lx * sin_t + ly * cos_t
        return int(cx + rx * scale), int(cy - ry * scale)

    # Torso triangle
    torso_verts = [(-14, -22), (14, -22), (6, 26)]
    pts_torso = [rot(vx, vy) for vx, vy in torso_verts]
    pygame.draw.polygon(surface, (210, 180, 100), pts_torso)
    pygame.draw.polygon(surface, (240, 240, 245), pts_torso, 1)

    # Head indicator
    ha = rot(6, 26)
    pygame.draw.circle(surface, (246, 210, 120), ha, max(2, int(4 * scale)))

    # Tail
    t_base = rot(0, -16)
    tail_len = B.TAIL_LEN * 0.70
    tail_rel = info.get("tail_rel", -math.pi / 2)
    tail_tip_loc = (tail_len * math.sin(tail_rel), -16 - tail_len * math.cos(tail_rel))
    pt_tail_tip = rot(*tail_tip_loc)
    pygame.draw.line(surface, (190, 150, 80), t_base, pt_tail_tip, 2)

    # Hips and Limbs
    pt_hip_f = rot(7, -20)
    pt_hip_b = rot(-7, -20)

    # Foot positions
    af = info["angle_f"]
    ab = info["angle_b"]
    df = info["dist_f"]
    db = info["dist_b"]

    foot_f_loc = (7 + df * math.sin(af), -20 - df * math.cos(af))
    foot_b_loc = (-7 - db * math.sin(ab), -20 - db * math.cos(ab))
    pt_foot_f = rot(*foot_f_loc)
    pt_foot_b = rot(*foot_b_loc)

    # Draw limbs
    pygame.draw.line(surface, (200, 160, 90), pt_hip_f, pt_foot_f, 2)
    pygame.draw.line(surface, (170, 135, 75), pt_hip_b, pt_foot_b, 2)
    pygame.draw.circle(surface, (240, 240, 245), pt_foot_f, 3)
    pygame.draw.circle(surface, (200, 200, 205), pt_foot_b, 3)

    # Ground line conditioned on touch contact (from token 8)
    t8 = step.state_tokens[8] if len(step.state_tokens) > 8 else None
    contact_f = (t8[SIG_OFFSET + 0] > 0.005) if t8 else False
    contact_b = (t8[SIG_OFFSET + 1] > 0.005) if t8 else False

    gy = None
    if contact_f and contact_b:
        gy = int((pt_foot_f[1] + pt_foot_b[1]) / 2) + 3
    elif contact_f:
        gy = pt_foot_f[1] + 3
    elif contact_b:
        gy = pt_foot_b[1] + 3

    if gy is not None:
        gy = max(by + 6, min(by + bh - 6, gy))
        pygame.draw.line(surface, (70, 85, 105), (bx + 4, gy), (bx + bw - 4, gy), 1)

    # Trunk contact point (token 9): red disc at the collision barycentre,
    # radius proportional to the collision force magnitude. The barycentre
    # (collision_tronc_cx/cy) is in the trunk-local egocentric frame — the
    # same frame as rot() — so it maps directly onto the torso.
    t9 = step.state_tokens[9] if len(step.state_tokens) > 9 else None
    if t9 is not None:
        fx = _denormalize(t9[SIG_OFFSET + 0], _BY_KEY["collision_tronc_x"])
        fy = _denormalize(t9[SIG_OFFSET + 1], _BY_KEY["collision_tronc_y"])
        mag = math.hypot(fx, fy)
        if mag > 10.0:  # low floor to ignore phantom contacts
            ccx = _denormalize(t9[SIG_OFFSET + 2], _BY_KEY["collision_tronc_cx"])
            ccy = _denormalize(t9[SIG_OFFSET + 3], _BY_KEY["collision_tronc_cy"])
            pt_contact = rot(ccx, ccy)
            radius = max(2, int(2 + 5 * min(1.0, mag / 1500.0)))
            pygame.draw.circle(surface, (220, 60, 60), pt_contact, radius)

    # Action consignes (targets) if present.
    # Symbols: plain circle = back limb target · circle + diameter = front
    # limb target · X cross = tail target.
    if step.action is not None and cand_color is not None:
        tf, df_norm, tb, db_norm, tq = step.action
        df_phys = B.LIMB_MIN + df_norm * (B.LIMB_MAX - B.LIMB_MIN)
        db_phys = B.LIMB_MIN + db_norm * (B.LIMB_MAX - B.LIMB_MIN)

        tgt_f_loc = (7 + df_phys * math.sin(tf * math.pi),
                     -20 - df_phys * math.cos(tf * math.pi))
        tgt_b_loc = (-7 - db_phys * math.sin(tb * math.pi),
                     -20 - db_phys * math.cos(tb * math.pi))
        pt_tgt_f = rot(*tgt_f_loc)
        pt_tgt_b = rot(*tgt_b_loc)

        # Front target handle: circle + diagonal diameter
        pygame.draw.circle(surface, cand_color, pt_tgt_f, 4, 1)
        d = 4 * 0.7071  # half-diagonal of the r=4 circle
        pygame.draw.line(surface, cand_color,
                         (pt_tgt_f[0] - d, pt_tgt_f[1] - d),
                         (pt_tgt_f[0] + d, pt_tgt_f[1] + d), 1)
        # Back target handle: plain circle
        pygame.draw.circle(surface, cand_color, pt_tgt_b, 4, 1)

        # Tail target handle (X cross). Tail consigne convention
        # (facing-independent): queue_theta + = tail up over the back,
        # - = down under the belly (at equilibrium queue_angle =
        # -theta_star). Note the state queue_angle rotates the other way:
        # + = toward the front, under the belly.
        tail_tgt_rel = -math.pi / 2 - tq * math.pi
        tgt_tail_loc = (tail_len * math.sin(tail_tgt_rel),
                        -16 - tail_len * math.cos(tail_tgt_rel))
        pt_tgt_tail = rot(*tgt_tail_loc)
        pygame.draw.line(surface, cand_color,
                         (pt_tgt_tail[0] - 3, pt_tgt_tail[1] - 3),
                         (pt_tgt_tail[0] + 3, pt_tgt_tail[1] + 3), 1)
        pygame.draw.line(surface, cand_color,
                         (pt_tgt_tail[0] - 3, pt_tgt_tail[1] + 3),
                         (pt_tgt_tail[0] + 3, pt_tgt_tail[1] - 3), 1)

    # Step label (top-left)
    lbl_text = step.label or highlight_label
    if lbl_text:
        s_lbl = font_small.render(lbl_text, FONT_AA, TEXT_MUTED)
        surface.blit(s_lbl, (bx + 4, by + 2))

    # Cost text (bottom-right)
    if step.step_cost != 0.0:
        c_text = f"{step.step_cost:.2f}"
        s_cost = font_small.render(c_text, FONT_AA, (210, 150, 150))
        surface.blit(s_cost, (bx + bw - s_cost.get_width() - 4, by + bh - 13))


class DreamTheater:
    """Manages the visual Sleep mode interface in Pygame.

    Hindsight (EOS-lookahead) policy steps are recorded in a per-cycle history
    and shown one at a time, storyboard style:
      - LIVED SEQUENCE: the full s0..s10 window as experienced (ground truth),
      - WM COMPLETION: real s0..s3, the hindsight-generated a4 on s4, and the
        imagined s5..s10 it causes (completion toward the forced EOS),
      - DECISION: side by side, what the policy infers on s4 vs what the WM
        recommends (cost, imagined EOS, accepted/rejected).
    A step browser (< PREV / NEXT > / LIVE, or arrow keys) navigates the
    history of the current sleep cycle; LIVE keeps following the newest step.

    Legacy fallback records (stochastic candidates) keep the original
    candidate x regime storyboard with tab filtering and scrolling."""

    MAX_HISTORY = 128

    def __init__(self, width: int = 960, height: int = 600):
        self.width = width
        self.height = height
        self.record: DreamRecord | None = None
        self.history: list[DreamRecord] = []
        self.hist_idx = 0
        self.follow = True   # auto-select the newest step as it arrives
        self.scroll_y = 0
        self.target_scroll_y = 0
        self.tab_idx = 0  # 0: All 6 rows, 1: Cand 0, 2: Cand 1, 3: Cand 2
        self.is_paused = False
        self.step_once = False
        self.tab_names = ["All (6 Rows)", "0: Policy", "1: Replay", "2: Noise 30%"]
        self._btn_rects: dict[str, pygame.Rect] = {}

        # Geometry
        self.header_h = 56
        self.thumb_w = 92
        self.thumb_h = 68
        self.thumb_gap = 6
        self.row_h = 76
        self.row_gap = 6
        self.seq_thumb_w = 64      # small thumbnails for full-sequence rows
        self.seq_thumb_h = 48
        self.seq_thumb_gap = 4

    def update(self, record: DreamRecord) -> None:
        self.record = record
        if self.history and self.history[-1] is record:
            return                      # re-displaying the same step (dream viewer)
        self.history.append(record)
        if len(self.history) > self.MAX_HISTORY:
            del self.history[0]
        if self.follow:
            self.hist_idx = len(self.history) - 1
        else:
            self.hist_idx = min(self.hist_idx, len(self.history) - 1)
        if record.lived_steps is not None:
            self.tab_names = ["All"] + list(record.candidate_names)
        else:
            self.tab_names = ["All (12 Rows)", "0: Policy", "1: Replay", "2: Noise 15%", "3: Noise 30%"]

    def _browse(self, delta: int) -> None:
        """Move the history selector by ``delta`` steps; landing on the newest
        step re-enables live following."""
        if not self.history:
            return
        self.hist_idx = max(0, min(len(self.history) - 1, self.hist_idx + delta))
        self.follow = (self.hist_idx == len(self.history) - 1)

    def clear(self) -> None:
        """Reset displayed storyboard to prepare for a new sleep cycle."""
        self.record = None
        self.history.clear()
        self.hist_idx = 0
        self.follow = True
        self.scroll_y = 0
        self.target_scroll_y = 0
        self.tab_idx = 0
        self.step_once = False

    def toggle_pause(self) -> bool:
        self.is_paused = not self.is_paused
        self.step_once = False
        return self.is_paused

    def handle_event(self, ev: pygame.event.Event) -> bool:
        """Handle mouse wheel, keys, browser buttons, and tab selection.
        Returns True if handled."""
        if ev.type == pygame.MOUSEWHEEL:
            self.target_scroll_y = min(0, self.target_scroll_y + ev.y * 40)
            return True
        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_SPACE:
                self.toggle_pause()
                return True
            elif ev.key == pygame.K_n and self.is_paused:
                self.step_once = True
                return True
            elif ev.key in (pygame.K_LEFT, pygame.K_RIGHT):
                self._browse(-1 if ev.key == pygame.K_LEFT else 1)
                return True
            elif ev.key == pygame.K_TAB:
                self.tab_idx = (self.tab_idx + 1) % len(self.tab_names)
                self.target_scroll_y = 0
                return True
            elif ev.key in (pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                val = ev.key - pygame.K_0
                if val < len(self.tab_names):
                    self.tab_idx = val
                    self.target_scroll_y = 0
                    return True
            elif ev.key == pygame.K_UP:
                self.target_scroll_y = min(0, self.target_scroll_y + 60)
                return True
            elif ev.key == pygame.K_DOWN:
                self.target_scroll_y -= 60
                return True
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            mx, my = ev.pos
            # Step browser buttons first
            for key, rect in self._btn_rects.items():
                if rect.collidepoint(ev.pos):
                    if key == "prev":
                        self._browse(-1)
                    elif key == "next":
                        self._browse(1)
                    elif key == "live" and self.history:
                        self.hist_idx = len(self.history) - 1
                        self.follow = True
                    return True
            # Tab clicks (legacy candidate filtering)
            if 26 <= my <= 52:
                tx = 16
                for i, name in enumerate(self.tab_names):
                    tw = 110
                    if tx <= mx <= tx + tw:
                        self.tab_idx = i
                        self.target_scroll_y = 0
                        return True
                    tx += tw + 8
        return False

    def draw(self, screen: pygame.Surface, font: pygame.font.Font,
             font_small: pygame.font.Font, status: str = "",
             brain=None) -> None:
        """Render the complete Dream Theater storyboard: the browser-selected
        policy step of the current sleep cycle, or the World Model training
        screen while no policy step is available yet."""
        screen.fill(DREAM_BG)

        if not self.history:
            self._draw_wm_training(screen, font, font_small, status, brain)
            return

        rec = self.history[self.hist_idx]

        # 1. Top Header Bar
        pygame.draw.rect(screen, HEADER_BG, (0, 0, self.width, self.header_h))
        pygame.draw.line(screen, CARD_BORDER, (0, self.header_h), (self.width, self.header_h), 1)

        # Title, status and loss
        pause_txt = " [PAUSED - [n] for next step]" if self.is_paused else " [Running — Space to pause]"
        title_str = f"DREAM THEATER  |  Step {rec.step_idx + 1}/{rec.total_steps}  |  Loss: {rec.loss:.4f}{pause_txt}"
        s_title = font.render(title_str, FONT_AA, WINNER_BORDER if self.is_paused else TEXT_WHITE)
        screen.blit(s_title, (16, 6))

        # Hindsight forcing target of the selected step (EOS-lookahead policy training)
        if rec.eos_forced is not None:
            dfat = 2.0 * rec.eos_forced[0] - 1.0
            dsuf = 2.0 * rec.eos_forced[1] - 1.0
            eos_txt = f"Hindsight EOS forced: dfat {dfat:+.2f} dsuffering {dsuf:+.2f}"
            s_eos = font_small.render(eos_txt, FONT_AA, TEXT_ACCENT)
            screen.blit(s_eos, (16 + s_title.get_width() + 24, 9))

        if rec.lived_steps is not None:
            self._draw_hindsight(screen, rec, font, font_small)
        else:
            self._draw_legacy(screen, rec, font, font_small)

    def _draw_wm_training(self, screen: pygame.Surface, font: pygame.font.Font,
                          font_small: pygame.font.Font, status: str,
                          brain) -> None:
        """Sleep screen shown before the first policy step of the cycle."""
        # World Model training screen (no "Phase 1")
        pygame.draw.rect(screen, HEADER_BG, (0, 0, self.width, self.header_h))
        pygame.draw.line(screen, CARD_BORDER, (0, self.header_h), (self.width, self.header_h), 1)
        title_str = "World Model Training"
        s_title = font.render(title_str, FONT_AA, TEXT_WHITE)
        screen.blit(s_title, (16, 16))

        cw, ch = 560, 236
        cx = (self.width - cw) // 2
        cy = (self.height - ch) // 2
        card_rect = pygame.Rect(cx, cy, cw, ch)
        pygame.draw.rect(screen, CARD_BG, card_rect, border_radius=8)
        pygame.draw.rect(screen, CARD_BORDER, card_rect, 1, border_radius=8)

        msg = status if status else "World Model optimization in progress..."
        txt = font.render(msg, FONT_AA, TEXT_WHITE)
        screen.blit(txt, (cx + (cw - txt.get_width()) // 2, cy + 14))

        import re
        m = re.search(r"(\d+)/(\d+)", msg)
        if m:
            step, total = int(m.group(1)), int(m.group(2))
            pct = min(1.0, max(0.0, step / max(1, total)))
            pb_w, pb_h = cw - 60, 6
            pb_x = cx + 30
            pb_y = cy + 36
            pygame.draw.rect(screen, (30, 36, 50), (pb_x, pb_y, pb_w, pb_h), border_radius=3)
            pygame.draw.rect(screen, (70, 140, 230), (pb_x, pb_y, int(pb_w * pct), pb_h), border_radius=3)

        # Replicate console print lines in the GUI
        print_lines: list[tuple[str, tuple[int, int, int]]] = []
        if brain is not None:
            cycle = getattr(brain, "sleep_cycle_stats", {})
            c_before = cycle.get("coreset_before", getattr(brain.buffer, "coreset_size", 0))
            n_wake = cycle.get("wake", 0)
            n_prune = cycle.get("pruned", 0)
            add_raw = cycle.get("addendum_initial", n_wake + n_prune)
            d_intra = cycle.get("dedup_intra", 0)
            d_cross = cycle.get("dedup_coreset", 0)
            n_refresh = cycle.get("refreshed", 0)
            f_dropped = cycle.get("filter_dropped", None)
            c_after = cycle.get("coreset_after", None)

            c_after_str = str(c_after) if c_after is not None else f"{brain.buffer.coreset_size} (in progress)"
            f_drop_str = f"-{f_dropped}" if f_dropped is not None else "-- (in progress)"

            print_lines.append((f"  - Coreset before sleep           : {c_before}", TEXT_WHITE))
            print_lines.append((f"  - New experiences                : +{n_wake}", TEXT_ACCENT))
            print_lines.append((f"  - Memories revisited (forgetting): +{n_prune}", TEXT_ACCENT))
            print_lines.append((f"  - Raw addendum                   : {add_raw}", TEXT_WHITE))
            print_lines.append((f"  - Intra-addendum duplicates      : -{d_intra}", TEXT_MUTED))
            print_lines.append((f"  - Addendum/coreset duplicates    : -{d_cross} ({n_refresh} refreshed)", TEXT_MUTED))
            print_lines.append((f"  - Familiar sequences dropped     : {f_drop_str}", TEXT_MUTED))
            print_lines.append((f"  - Coreset after sleep            : {c_after_str}", WINNER_BORDER if c_after is not None else TEXT_WHITE))
        else:
            print_lines.append(("  - Waiting for sleep cycle to start...", TEXT_MUTED))

        for idx, (s_line, s_col) in enumerate(print_lines):
            s_rend = font_small.render(s_line, FONT_AA, s_col)
            screen.blit(s_rend, (cx + 36, cy + 52 + idx * 21))

    def _draw_legacy(self, screen: pygame.Surface, rec: DreamRecord,
                     font: pygame.font.Font, font_small: pygame.font.Font) -> None:
        """Legacy stochastic-candidate storyboard: 3 candidates x 2 imagined
        regimes, tab-filterable, scrollable."""
        # Smooth scroll lerp
        self.scroll_y += (self.target_scroll_y - self.scroll_y) * 0.3

        # Tabs
        tx = 16
        for i, name in enumerate(self.tab_names):
            tw = 110
            rect_tab = pygame.Rect(tx, 28, tw, 22)
            is_active = (i == self.tab_idx)
            tab_bg = (50, 56, 75) if is_active else (30, 34, 46)
            tab_border = WINNER_BORDER if is_active else CARD_BORDER
            pygame.draw.rect(screen, tab_bg, rect_tab, border_radius=3)
            pygame.draw.rect(screen, tab_border, rect_tab, 1, border_radius=3)
            c_name = CAND_COLORS[(i - 1) % len(CAND_COLORS)] if i > 0 else TEXT_WHITE
            s_tab = font_small.render(name, FONT_AA, c_name if is_active else TEXT_MUTED)
            screen.blit(s_tab, (tx + (tw - s_tab.get_width()) // 2, 31))
            tx += tw + 8

        # Help hint on top-right
        hint_txt = "Tab / 0-4: Filter | Arrows: Steps | Wheel: Scroll | [n]: Next step | Space: Resume" if self.is_paused else "Tab / 0-4: Filter | Arrows: Steps | Wheel: Scroll | Space: Pause"
        hint = font_small.render(hint_txt, FONT_AA, TEXT_MUTED)
        screen.blit(hint, (self.width - hint.get_width() - 16, 31))

        # 2. Rows content area
        content_y = self.header_h + 10 + int(self.scroll_y)

        # --- ROW 0: Real Context (s0 -> s4) ---
        row_rect = pygame.Rect(12, content_y, self.width - 24, self.row_h)
        pygame.draw.rect(screen, (24, 28, 36), row_rect, border_radius=4)
        pygame.draw.rect(screen, (60, 68, 88), row_rect, 1, border_radius=4)

        # Context row label
        s_cxt = font_small.render("PAST REAL CONTEXT (s0 → s4)", FONT_AA, TEXT_ACCENT)
        screen.blit(s_cxt, (20, content_y + 4))

        # Context thumbnails
        start_x = 180
        for idx, step in enumerate(rec.context_steps):
            th_rect = pygame.Rect(start_x + idx * (self.thumb_w + self.thumb_gap),
                                  content_y + 4, self.thumb_w, self.thumb_h)
            lbl = f"s{idx}" if idx < len(rec.context_steps) - 1 else "s4 (decision)"
            cand_c = CAND_COLORS[0] if step.action is not None else None
            draw_thumbnail(screen, th_rect, step, font_small, facing=rec.facing,
                           cand_color=cand_c, highlight_label=lbl)

        content_y += self.row_h + self.row_gap + 8

        # --- Imagined Trajectory Rows ---
        # Filter trajectories based on tab
        visible_trajs: list[DreamTrajectory] = []
        for traj in rec.trajectories:
            if self.tab_idx == 0 or traj.candidate_idx == (self.tab_idx - 1):
                visible_trajs.append(traj)

        last_cand = -1
        for traj in visible_trajs:
            c_idx = traj.candidate_idx
            cand_color = CAND_COLORS[c_idx]

            # Candidate group header
            if c_idx != last_cand:
                last_cand = c_idx
                is_win_cand = (c_idx == rec.best_candidate_idx)
                header_text = f"CANDIDATE {rec.candidate_names[c_idx]}"
                if is_win_cand:
                    header_text += "  ★ WINNER (SELECTED FOR LEARNING)"
                s_grp = font.render(header_text, FONT_AA, WINNER_BORDER if is_win_cand else cand_color)
                screen.blit(s_grp, (16, content_y))
                content_y += 24

            # Trajectory row box
            row_rect = pygame.Rect(12, content_y, self.width - 24, self.row_h)
            is_win = traj.is_winner
            is_best = traj.is_best_future

            row_bg = (34, 46, 38) if is_win else ((28, 38, 32) if is_best else CARD_BG)
            row_border = WINNER_BORDER if is_win else (BEST_BORDER if is_best else CARD_BORDER)
            pygame.draw.rect(screen, row_bg, row_rect, border_radius=4)
            pygame.draw.rect(screen, row_border, row_rect, 2 if (is_win or is_best) else 1, border_radius=4)

            # Left row info: Regime name + Cost
            r_name = traj.regime_name
            s_reg = font_small.render(r_name, FONT_AA, TEXT_WHITE)
            screen.blit(s_reg, (20, content_y + 12))

            cost_str = f"Cost: {traj.total_cost:.2f}"
            s_cost = font_small.render(cost_str, FONT_AA, WINNER_BORDER if is_win else (BEST_BORDER if is_best else TEXT_MUTED))
            screen.blit(s_cost, (20, content_y + 30))

            if is_win:
                s_win = font_small.render("🏆 WINNER", FONT_AA, WINNER_BORDER)
                screen.blit(s_win, (20, content_y + 48))
            elif traj.eos_pred is not None:
                # Imagined terminal outcome of this rollout (physical deltas)
                dfat = 2.0 * traj.eos_pred[0] - 1.0
                dsuf = 2.0 * traj.eos_pred[1] - 1.0
                s_eos = font_small.render(f"EOS dfat {dfat:+.2f} dsuf {dsuf:+.2f}",
                                          FONT_AA, TEXT_MUTED)
                screen.blit(s_eos, (20, content_y + 48))
            elif is_best:
                s_best = font_small.render("★ Best Regime", FONT_AA, BEST_BORDER)
                screen.blit(s_best, (20, content_y + 48))

            # Render thumbnails along the row: [s4+a4] [s5+a5] [s6+a6] ...
            start_x = 180
            for s_idx, st in enumerate(traj.steps):
                th_rect = pygame.Rect(start_x + s_idx * (self.thumb_w + self.thumb_gap),
                                      content_y + 4, self.thumb_w, self.thumb_h)
                lbl = st.label or ("s4+cand" if s_idx == 0 else f"s{4 + s_idx}")
                draw_thumbnail(screen, th_rect, st, font_small, facing=rec.facing,
                               cand_color=cand_color, is_best=is_best, is_winner=is_win,
                               highlight_label=lbl)

            content_y += self.row_h + self.row_gap

        # Min scroll clamping
        min_scroll = min(0, self.height - content_y - 20)
        self.target_scroll_y = max(min_scroll, min(0, self.target_scroll_y))

    # --- hindsight (EOS-lookahead) storyboard ---------------------------------
    def _draw_hindsight(self, screen: pygame.Surface, rec: DreamRecord,
                        font: pygame.font.Font, font_small: pygame.font.Font) -> None:
        """Storyboard of one hindsight policy step: the lived sequence, the WM
        completion toward the forced EOS, and the policy-vs-WM decision."""
        # the full-sequence rows fit on screen: no scrolling in this view
        self.scroll_y = self.target_scroll_y = 0
        self._draw_browser(screen, rec, font_small)

        y = self.header_h + 10

        # --- LIVED SEQUENCE row: the ground truth s0 -> s10 ---
        label = "LIVED SEQUENCE (s0-s10)"
        if rec.eos_realized is not None:
            label += f"  EOS dfat {2.0 * rec.eos_realized[0] - 1.0:+.2f} dsuf {2.0 * rec.eos_realized[1] - 1.0:+.2f}"
        y = self._draw_seq_row(screen, rec, rec.lived_steps, label, y, font_small,
                               TEXT_WHITE, (200, 200, 210))

        # --- WM COMPLETION row: hindsight-generated a4 + imagined s5 -> s10 ---
        label = "WM COMPLETION (hindsight EOS)"
        traj_gen = rec.trajectories[1]
        if traj_gen.eos_pred is not None:
            label += f"  EOS dfat {2.0 * traj_gen.eos_pred[0] - 1.0:+.2f} dsuf {2.0 * traj_gen.eos_pred[1] - 1.0:+.2f}"
        y = self._draw_seq_row(screen, rec, rec.predicted_steps, label, y, font_small,
                               TEXT_ACCENT, CAND_COLORS[1])

        # --- DECISION row: what the policy infers vs what the WM recommends ---
        y = self._draw_decision_row(screen, rec, font, font_small, y)

    def _draw_browser(self, screen: pygame.Surface, rec: DreamRecord,
                      font_small: pygame.font.Font) -> None:
        """Step browser bar: < PREV / NEXT > / LIVE plus the step indicator —
        the selector for browsing what happened during policy training."""
        n = len(self.history)
        specs = [
            ("prev", "< PREV", self.hist_idx > 0),
            ("next", "NEXT >", self.hist_idx < n - 1),
            ("live", "LIVE", self.follow),
        ]
        x = 16
        for key, txt, enabled in specs:
            rect = pygame.Rect(x, 28, 84, 22)
            bg = (50, 56, 75) if enabled else (26, 30, 40)
            border = WINNER_BORDER if enabled else CARD_BORDER
            pygame.draw.rect(screen, bg, rect, border_radius=3)
            pygame.draw.rect(screen, border, rect, 1, border_radius=3)
            s = font_small.render(txt, FONT_AA, TEXT_WHITE if enabled else TEXT_MUTED)
            screen.blit(s, (x + (84 - s.get_width()) // 2, 31))
            self._btn_rects[key] = rect
            x += 84 + 8
        s_idx = font_small.render(f"step {self.hist_idx + 1}/{n}", FONT_AA, TEXT_WHITE)
        screen.blit(s_idx, (x + 4, 31))

        # Acceptance of the selected step (hindsight supervision metric)
        acc = f"accepted: {'yes' if traj_accepted(rec) else 'no'}"
        s_acc = font_small.render(acc, FONT_AA,
                                  BEST_BORDER if traj_accepted(rec) else TEXT_MUTED)
        screen.blit(s_acc, (x + 4 + s_idx.get_width() + 24, 31))

        # Help hint on top-right
        hint_txt = "Arrows: Browse steps | [n]: Next step | Space: Resume" if self.is_paused \
            else "Arrows: Browse steps | Space: Pause"
        hint = font_small.render(hint_txt, FONT_AA, TEXT_MUTED)
        screen.blit(hint, (self.width - hint.get_width() - 16, 31))

    def _draw_seq_row(self, screen: pygame.Surface, rec: DreamRecord,
                      steps: list[DreamStep], label: str, y: int,
                      font_small: pygame.font.Font,
                      label_color, action_color) -> int:
        """One full-sequence row: label line, then small thumbnails wrapped
        below it. Returns the y below the row."""
        s_lbl = font_small.render(label, FONT_AA, label_color)
        screen.blit(s_lbl, (16, y))
        yy = y + 16
        x0 = 16
        avail = max(self.seq_thumb_w, self.width - 24 - x0)
        step_w = self.seq_thumb_w + self.seq_thumb_gap
        x = x0
        for st in steps:
            if x + self.seq_thumb_w > x0 + avail:
                x = x0
                yy += self.seq_thumb_h + self.seq_thumb_gap
            rect = pygame.Rect(x, yy, self.seq_thumb_w, self.seq_thumb_h)
            draw_thumbnail(screen, rect, st, font_small, facing=rec.facing,
                           cand_color=action_color)
            x += step_w
        return yy + self.seq_thumb_h + 10

    def _draw_decision_row(self, screen: pygame.Surface, rec: DreamRecord,
                           font: pygame.font.Font, font_small: pygame.font.Font,
                           y: int) -> int:
        """Side-by-side decision on s4: what the policy infers vs what the WM
        recommends, with cost, imagined EOS and acceptance."""
        s_lbl = font_small.render("DECISION (POLICY vs WM HINDSIGHT)", FONT_AA, TEXT_ACCENT)
        screen.blit(s_lbl, (16, y))
        y += 16
        s4 = rec.context_steps[-1]
        names = ["POLICY", "WM HINDSIGHT"]
        x = 16
        for c_idx in range(2):
            traj = rec.trajectories[c_idx]
            step = DreamStep(state_tokens=s4.state_tokens,
                             action=rec.candidate_actions[c_idx])
            rect = pygame.Rect(x, y, self.thumb_w, self.thumb_h)
            draw_thumbnail(screen, rect, step, font_small, facing=rec.facing,
                           cand_color=CAND_COLORS[c_idx], is_winner=traj.is_winner)
            ax = x + self.thumb_w + 10
            lines = [names[c_idx], f"Cost {traj.total_cost:.2f}"]
            if traj.eos_pred is not None:
                lines.append(f"EOS dfat {2.0 * traj.eos_pred[0] - 1.0:+.2f}"
                             f" dsuf {2.0 * traj.eos_pred[1] - 1.0:+.2f}")
            if c_idx == 1:
                lines.append("ACCEPTED" if traj.is_winner else "REJECTED")
            col = WINNER_BORDER if traj.is_winner else CAND_COLORS[c_idx]
            for i, txt in enumerate(lines):
                s = font_small.render(txt, FONT_AA, col if i == 0 else TEXT_MUTED)
                screen.blit(s, (ax, y + 4 + i * 16))
            x = ax + 150
        return y + self.thumb_h + 12


def traj_accepted(rec: DreamRecord) -> bool:
    """Whether the WM's hindsight action was accepted for learning in this step."""
    return bool(rec.trajectories) and rec.trajectories[-1].is_winner
