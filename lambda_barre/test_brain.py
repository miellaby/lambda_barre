"""End-to-end test for the 2-model brain (no pygame window).

Exercises the full pipeline: encode a dense salve, journal transitions, run a
sleep cycle that trains both the world model and the policy, and check that
the world-model loss decreases on structured data and that the policy produces
valid consignes.

Run with the project on PYTHONPATH:
    PYTHONPATH=. python -m pytest lambda_barre/test_brain.py -q
or directly:
    PYTHONPATH=. python -m lambda_barre.test_brain
"""
from __future__ import annotations

import random

from . import tokenize as T
from .brain import Brain, ExperienceBuffer
from .models import Policy, WorldModel, THETA_RANGE


def _run_sleep(b, **kw):
    """Consume a sleep generator and return the final stats dict."""
    for label, value in b.sleep(**kw):
        if label == "done":
            return value
    return None


def _make_salve(state_signals: list[float],
                action_signals: list[float]) -> list[list[float]]:
    """Build a 16-token dense salve from 53 sensory + 5 motor [0,1] floats."""
    return T.make_salve(state_signals, action_signals)


def test_policy_outputs_are_valid_consignes():
    import torch
    p = Policy()
    sv = torch.rand(4, p.N_STATE)
    a = p(sv)
    assert a.shape == (4, 5)
    cons = a.tolist()
    for row in cons:
        tl, dl, tr, dr, tq = row
        assert -1.0 - 1e-3 <= tl <= 1.0 + 1e-3 and -1.0 - 1e-3 <= tr <= 1.0 + 1e-3 and -1.0 - 1e-3 <= tq <= 1.0 + 1e-3
        assert 0.0 - 1e-3 <= dl <= 1.0 + 1e-3
        assert 0.0 - 1e-3 <= dr <= 1.0 + 1e-3


def test_world_model_forward_and_predict_shapes():
    import torch
    wm = WorldModel()
    L = T.SALVE_TOKENS + T.SALVE_TOKENS + T.STATE_TOKENS  # 16+16+13 = 45
    x = torch.rand(3, L, T.DENSE_DIM)
    out = wm(x)
    assert out.shape == (3, L, T.N_SIGNAL)
    assert not torch.isnan(out).any().item()
    # predict_next / predict_next_state: ctx = [B, 16, 25] -> [B, 13, 25]
    ctx = torch.rand(3, T.SALVE_TOKENS, T.DENSE_DIM)
    nxt_s = wm.predict_next_state(ctx)
    assert nxt_s.shape == (3, T.STATE_TOKENS, T.DENSE_DIM)
    sigs_s = nxt_s[:, :, 9:]
    assert bool((sigs_s >= 0).all().item()) and bool((sigs_s <= 1).all().item())

    # predict_next_action: ctx_with_state = [B, 29, 25] -> [B, 3, 25]
    ctx_with_s = torch.cat([ctx, nxt_s], dim=1)
    nxt_a = wm.predict_next_action(ctx_with_s)
    assert nxt_a.shape == (3, T.ACTION_TOKENS, T.DENSE_DIM)
    sigs_a = nxt_a[:, :, 9:]
    assert bool((sigs_a >= 0).all().item()) and bool((sigs_a <= 1).all().item())

    # predict_next_salve: ctx = [B, 16, 25] -> state [B, 13, 25], action [B, 3, 25]
    s, a = wm.predict_next_salve(ctx)
    assert s.shape == (3, T.STATE_TOKENS, T.DENSE_DIM)
    assert a.shape == (3, T.ACTION_TOKENS, T.DENSE_DIM)


def test_salve_cost_sign():
    # salve_cost sums the 6 innate signals (effort, douleur, courbature,
    # instabilite, vertige, confort). confort is signed (negative = reward);
    # the 5 others are unsigned costs.
    # Flat state-signal layout: token 10 (Coûts) holds effort, douleur,
    # courbature, instabilite, vertige at indices 45..49; token 11 (Récompense)
    # holds confort at index 50; token 12 (Intéroception) holds fatigue, souffrance at 51..52.
    # all-zero: confort=0 -> -1.0 (max reward) -> cost = -1.0 < 0
    salve = _make_salve([0.0] * 53, [0.5] * 5)
    assert T.salve_cost(salve) < 0.0
    # douleur=1.0 (->1.0), confort=0.5 (->0.0) -> cost = 4.0 > 0
    ss = [0.0] * 53
    ss[46] = 1.0      # douleur slot
    ss[50] = 0.5      # confort slot
    salve = _make_salve(ss, [0.5] * 5)
    assert T.salve_cost(salve) > 0.0


def test_sleep_trains_both_models_and_world_loss_decreases():
    random.seed(1)
    b = Brain(seed=1)
    # Structured world: next state ≈ current state (near-static), with low cost
    # (confort high = reward, costs 0). This gives the world model something
    # learnable: predict the next state ≈ the current state.
    for _ in range(160):
        ss = [random.random() for _ in range(53)]
        ss[52] = 0.0  # confort=0 -> -1.0 (max reward)
        salve = _make_salve(ss, [0.5, 0.5, 0.5, 0.5, 0.5])
        b.record(salve)
    assert len(b.buffer) >= 1  # at least one complete 10-step sequence
    s1 = _run_sleep(b, wm_epochs=4, wm_batch=32, pol_steps=8, pol_batch=12, clear_buffer=False)
    s2 = _run_sleep(b, wm_epochs=4, wm_batch=32, pol_steps=8, pol_batch=12)
    # losses are finite
    for k in ("wm_loss", "pol_loss"):
        assert s2[k] == s2[k], f"{k} is NaN"   # NaN check
    # world-model loss should drop on this learnable (near-identity) data
    assert s2["wm_loss"] < s1["wm_loss"], (s1["wm_loss"], s2["wm_loss"])


def test_act_returns_five_consignes():
    random.seed(2)
    b = Brain(seed=2)
    salve = _make_salve([random.random() for _ in range(53)],
                        [0.5, 0.5, 0.5, 0.5, 0.5])
    a = b.act(salve)
    assert len(a) == 5


def test_brain_drives_real_sim_and_sleeps():
    """End-to-end: the brain drives the real pymunk body, the real sensors +
    encoder produce dense salves, transitions are journaled, and a sleep cycle
    trains both models off that real experience. No display is needed."""
    from . import body as B, world as W, proprio as S, extero as E, intero as I
    from .tokenize import DenseEncoder
    space = W.make_space()
    skel = B.build_skeleton(space)
    B.apply_consignes(skel)
    proprio = S.Proprio(space, skel)
    reward = S.Reward(skel)
    cursor = E.Cursor(skel)
    vision = E.Vision(skel, space)
    touch = E.Touch(space, skel)
    intero = I.Intero()
    encoder = DenseEncoder()
    brain = Brain(seed=3)
    accum = 0.0
    DT = 1.0 / 6.0
    mouse = (400, 300)
    for _ in range(240):           # 4 s of simulation
        touch.reset_contacts()
        for _s in range(3):
            W.step(space, skel, 1 / 180)
        signals = proprio.update(skel, 1 / 60)
        ts = touch.update(skel, 1 / 60)
        rs = reward.update(skel, signals, ts, 1 / 60)
        isg = intero.update(rs, 1 / 60)
        cs = cursor.update(skel, mouse, 1 / 60)
        vs = vision.update(skel, 1 / 60)
        accum += 1 / 60
        if accum >= DT:
            accum -= DT
            salve = encoder.encode(signals, ts, cs, vs, isg, rs, skel)
            brain.record(salve)
            tl, dl, tr, dr, tq = brain.act(salve)
            tf_phys = tl * THETA_RANGE
            tb_phys = tr * THETA_RANGE
            df_phys = B.LIMB_MIN + dl * (B.LIMB_MAX - B.LIMB_MIN)
            db_phys = B.LIMB_MIN + dr * (B.LIMB_MAX - B.LIMB_MIN)
            tq_phys = tq * THETA_RANGE

            if skel.facing == 1:
                skel.limb_r.theta_star = -tf_phys
                skel.limb_r.d_star = df_phys
                skel.limb_l.theta_star = +tb_phys
                skel.limb_l.d_star = db_phys
            else:
                skel.limb_l.theta_star = +tf_phys
                skel.limb_l.d_star = df_phys
                skel.limb_r.theta_star = -tb_phys
                skel.limb_r.d_star = db_phys

            skel.tail_act.theta_star = tq_phys
            B.apply_consignes(skel)
    assert len(brain.buffer) >= 1  # at least one complete sequence from 4s sim
    stats = _run_sleep(brain, wm_epochs=2, wm_batch=24, pol_steps=4, pol_batch=12)
    for k in ("wm_loss", "pol_loss"):
        assert stats[k] == stats[k]   # finite
    a = brain.act(salve)
    assert len(a) == 5


def test_experience_buffer_linear_recording_and_sliding_windows():
    buf = ExperienceBuffer(seq_len=5, capacity=20, seed=42)
    assert len(buf) == 0
    assert buf.sample(4) == []

    # Push 4 items (< seq_len 5)
    for i in range(4):
        buf.push([[float(i)] * 25] * 16)
    assert len(buf) == 0

    # 5th item reaches seq_len -> exactly 1 window
    buf.push([[4.0] * 25] * 16)
    assert len(buf) == 1

    # Push 5 more -> total 10 items in segment -> 10 - 5 + 1 = 6 windows
    for i in range(5, 10):
        buf.push([[float(i)] * 25] * 16)
    assert len(buf) == 6

    samples = buf.sample(10)
    assert len(samples) == 6  # min(batch, total)
    for seq in samples:
        assert len(seq) == 5
        # Check contiguity
        vals = [s[0][0] for s in seq]
        for j in range(len(vals) - 1):
            assert vals[j + 1] == vals[j] + 1.0


def test_experience_buffer_boundary_and_discontinuity_isolation():
    buf = ExperienceBuffer(seq_len=5, capacity=50, seed=42)

    # Segment 1: values 0..9 (10 items, 6 windows)
    for i in range(10):
        buf.push([[float(i)] * 25] * 16)
    buf.boundary()
    assert len(buf) == 6

    # Incomplete segment: values 50..52 (3 items < 5)
    for i in range(50, 53):
        buf.push([[float(i)] * 25] * 16)
    buf.boundary()  # Must be dropped!
    assert len(buf) == 6

    # Segment 2: values 100..107 (8 items, 4 windows)
    for i in range(100, 108):
        buf.push([[float(i)] * 25] * 16)

    # Total extractable windows = 6 + 4 = 10
    assert len(buf) == 10
    assert buf.num_segments == 2

    # Draw samples and verify none contains dropped items or crosses boundary
    samples = buf.sample(20)
    assert len(samples) == 10
    for seq in samples:
        vals = [s[0][0] for s in seq]
        assert len(vals) == 5
        # Must belong entirely to seg 1 [0..9] OR seg 2 [100..107]
        is_seg1 = all(0.0 <= v <= 9.0 for v in vals)
        is_seg2 = all(100.0 <= v <= 107.0 for v in vals)
        assert is_seg1 or is_seg2, f"Sequence crossed boundary or contained dropped items: {vals}"


def test_experience_buffer_capacity_eviction():
    buf = ExperienceBuffer(seq_len=5, capacity=8, seed=42)
    # Push 20 items -> 20 - 5 + 1 = 16 windows, should be capped at capacity=8
    for i in range(20):
        buf.push([[float(i)] * 25] * 16)
    assert len(buf) == 8

    # Oldest items should have been evicted; newest items should remain
    samples = buf.sample(8)
    for seq in samples:
        vals = [s[0][0] for s in seq]
        for v in vals:
            assert v >= 8.0  # since last 12 items (8..19) contain 8 windows


def test_experience_buffer_clear():
    buf = ExperienceBuffer(seq_len=5, capacity=20, seed=42)
    for i in range(15):
        buf.push([[float(i)] * 25] * 16)
    buf.boundary()
    for i in range(10):
        buf.push([[float(i)] * 25] * 16)
    assert len(buf) > 0
    assert buf.num_segments == 2

    buf.clear()
    assert len(buf) == 0
    assert buf.num_segments == 0
    assert buf.sample(5) == []


def test_brain_boundary_and_history_isolation():
    b = Brain(seed=42)
    # Push salves into brain (moving transitions)
    for i in range(15):
        salve = _make_salve([0.05 * i] * 53, [0.5] * 5)
        b.record(salve)
    assert len(b.buffer) == 15 - (b.buffer.seq_len) + 1

    # Signal boundary (e.g. facing flip or reset)
    b.boundary()
    assert len(b.buffer) == 15 - (b.buffer.seq_len) + 1  # preserved in completed segments
    assert len(b._wake_history) == 0                     # wake context cleared
    assert b._cached_latent is None


def test_experience_buffer_two_tier_consolidation_and_persistence():
    buf = ExperienceBuffer(seq_len=5, capacity=50, pool_capacity=100, seed=42)
    # Session 1: push 15 items into segment 1, seal, then 10 items into segment 2
    for i in range(15):
        buf.push([[float(i)] * 25] * 16)
    buf.boundary()
    for i in range(15, 25):
        buf.push([[float(i)] * 25] * 16)

    assert buf.pool_size == 0
    # Seg 1: 15 - 5 + 1 = 11 windows. Seg 2: 10 - 5 + 1 = 6 windows. Total = 17 windows
    assert buf.journal_len == 17
    assert len(buf) == 17

    # Consolidate with stride 2
    # Seg 1: offsets 0, 2, 4, 6, 8, 10 -> 6 windows.
    # Seg 2: offsets 0, 2, 4, 5 (last window) -> 4 windows. Total added = 10
    added = buf.consolidate(stride=2)
    assert added == 10
    assert buf.pool_size == 10
    assert buf.journal_len == 0  # Journal cleared upon consolidation
    assert len(buf) == 10

    # Samples now come from the persistent pool
    samples = buf.sample(5)
    assert len(samples) == 5
    for s in samples:
        assert len(s) == 5

    # Session 2: push 8 new items during wake
    for i in range(100, 108):
        buf.push([[float(i)] * 25] * 16)
    assert buf.pool_size == 10
    assert buf.journal_len == 8 - 5 + 1  # 4 windows in new session
    assert len(buf) == 14

    # Clear journal only (as sleep waking does)
    buf.clear_journal()
    assert buf.pool_size == 10  # persistent pool remains intact!
    assert buf.journal_len == 0
    assert len(buf) == 10

    # Total clear
    buf.clear()
    assert buf.pool_size == 0
    assert buf.journal_len == 0
    assert len(buf) == 0


def test_explore_action_batch_bounds_and_sigma():
    import torch
    b = Brain(seed=42)
    base = torch.tensor([
        [0.0, 0.5, 0.0, 0.5, 0.0],
        [0.9, 0.95, -0.9, 0.05, 0.95],
    ], device=b.device)

    # Large sigma to trigger bounds
    explored = b._explore_action_batch(base, sigma=5.0)
    assert explored.shape == (2, 5)

    # Signed dims: 0, 2, 4 must be in [-1.0, 1.0]
    for col in (0, 2, 4):
        assert (explored[:, col] >= -1.0 - 1e-5).all()
        assert (explored[:, col] <= 1.0 + 1e-5).all()

    # Unsigned dims: 1, 3 must be in [0.0, 1.0]
    for col in (1, 3):
        assert (explored[:, col] >= 0.0 - 1e-5).all()
        assert (explored[:, col] <= 1.0 + 1e-5).all()


def test_train_policy_candidate_selection():
    import torch
    from .brain import _EXPLORE_MARGIN_PCT
    # Candidate ranking logic: alternatives (1..3) must beat baseline (0) by at least margin
    candidate_costs = torch.tensor([
        [5.000, 5.100, 4.9995, 5.050],  # 4.9995 is within noise (< 5% gain) -> picks baseline (0)
        [5.000, 5.200, 5.700, 4.500],   # 4.500 has 10% gain (> 5%) -> picks candidate 3
        [5.500, 4.000, 4.300, 4.200],   # demonstrated is 4.000 (> 5% gain) -> picks candidate 1
        [3.000, 4.000, 4.300, 4.200],   # policy is 3.000 -> picks candidate 0
    ])
    cost_baseline = candidate_costs[:, 0:1]
    margin = _EXPLORE_MARGIN_PCT * cost_baseline.abs().clamp(min=1.0)
    adjusted_costs = candidate_costs.clone()
    adjusted_costs[:, 1:] += margin
    best_idx = adjusted_costs.argmin(dim=1)
    assert best_idx.tolist() == [0, 3, 1, 0]


def test_sleep_preserves_wake_history_context():
    b = Brain(seed=42)
    # Record and tick several transitions
    for i in range(15):
        salve = _make_salve([0.1 * (i % 5)] * 53, [0.5] * 5)
        b.record(salve)
        b.wake_tick(salve)

    wake_len_before = len(b._wake_history)
    assert wake_len_before > 0
    assert b._cached_latent is not None

    # Run sleep
    _run_sleep(b, wm_epochs=1, wm_batch=8, pol_steps=1, pol_batch=4)

    # Wake history and cached latent must be preserved across sleep
    assert len(b._wake_history) == wake_len_before
    assert b._cached_latent is not None


def test_reset_restores_limb_distances():
    from . import body as B, world as W
    space = W.make_space()
    skel = B.build_skeleton(space)
    # Default distances should be 20.0
    assert skel.limb_l.d_star == 20.0
    assert skel.limb_r.d_star == 20.0
    assert skel.spawn_d_l == 20.0
    assert skel.spawn_d_r == 20.0

    # Modify distances
    skel.limb_l.d_star = 31.5
    skel.limb_r.d_star = 12.3
    skel.limb_l.theta_star = 0.99
    skel.limb_r.theta_star = -0.55
    B.apply_consignes(skel)

    assert skel.limb_l.d_star == 31.5
    assert skel.limb_r.d_star == 12.3

    # Reset skeleton
    B.reset(skel)

    # Distances and thetas must be restored to spawn values
    assert skel.limb_l.d_star == 20.0
    assert skel.limb_r.d_star == 20.0
    assert skel.limb_l.theta_star == skel.spawn_theta_l
    assert skel.limb_r.theta_star == skel.spawn_theta_r


def test_three_futures_bellman_optimism():
    import torch
    b = Brain(seed=42)
    # 1. Test decode_action_batch roundtrip
    action_in = torch.tensor([[0.3, 0.7, -0.4, 0.1, -0.8]], device=b.device)
    toks = b._encode_action_batch(action_in)
    decoded = b._decode_action_batch(toks)
    assert torch.allclose(action_in, decoded, atol=1e-5)

    # 2. Test Bellman optimism logic: min across 3 futures
    c_kalman = torch.tensor([12.0, 30.0, 15.0])
    c_wm = torch.tensor([25.0, 10.0, 20.0])
    c_pol = torch.tensor([50.0, 45.0, 8.0])  # clumsy policy has high cost on 0 and 1
    c_optimistic = torch.minimum(torch.minimum(c_kalman, c_wm), c_pol)
    assert c_optimistic.tolist() == [12.0, 10.0, 8.0]

    # 3. Test train_policy execution with 3-future evaluation
    for i in range(20):
        salve = _make_salve([0.1 * (i % 5)] * 53, [0.5] * 5)
        b.record(salve)
    steps_out = list(b.train_policy(steps=2, batch=4, n_imagine=3))
    assert len(steps_out) == 2
    for phase, loss in steps_out:
        assert phase == "pol"
        assert loss == loss and loss >= 0.0  # valid finite loss


def test_dream_record_and_dream_theater_rendering():
    import pygame
    from lambda_barre.dream import DreamTheater, DreamRecord
    pygame.init()

    b = Brain(seed=42)
    for i in range(20):
        salve = _make_salve([0.05 * (i % 5)] * 53, [0.2 * (i % 4)] * 5)
        b.record(salve)
    list(b.train_policy(steps=1, batch=4, n_imagine=4))

    assert b.last_dream_record is not None
    rec = b.last_dream_record
    assert isinstance(rec, DreamRecord)
    assert len(rec.context_steps) == 5  # s0..s3 with actions, s4 decision state
    assert len(rec.candidate_actions) == 4
    assert len(rec.trajectories) == 12  # 4 cands × 3 regimes
    for traj in rec.trajectories:
        assert len(traj.steps) == 1 + 4  # s4+cand + 4 imagined steps

    # Test DreamTheater UI
    theater = DreamTheater(960, 600)
    theater.update(rec)
    assert not theater.is_paused
    theater.toggle_pause()
    assert theater.is_paused
    assert not theater.step_once

    # Test single-step key (N) when paused
    ev_n = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n)
    assert theater.handle_event(ev_n)
    assert theater.step_once

    # Test keyboard navigation
    ev_tab = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB)
    assert theater.handle_event(ev_tab)
    assert theater.tab_idx == 1

    ev_wheel = pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-2)
    assert theater.handle_event(ev_wheel)

    # Test drawing on Surface
    surf = pygame.Surface((960, 600))
    font = pygame.font.SysFont("monospace", 16)
    font_small = pygame.font.SysFont("monospace", 10)
    theater.draw(surf, font, font_small, status="sleeping...")

    # Test filter by candidate tabs
    for tab in range(5):
        theater.tab_idx = tab
        theater.draw(surf, font, font_small)


def test_brain_record_stationary_pause_and_resume():
    b = Brain(seed=42)
    assert b.is_recording

    # 1. Push 11 moving salves so sequence is sufficiently long (>= seq_len 11)
    for i in range(11):
        salve = _make_salve([0.05 * i] * 53, [0.1 * i] * 5)
        recorded = b.record(salve)
        assert recorded is True
        assert b.is_recording is True
    assert len(b.buffer._current) == 11

    # 2. Push 5 identical static salves (matching the 11th salve)
    static_salve = _make_salve([0.5] * 53, [1.0] * 5)
    for _ in range(5):
        recorded = b.record(static_salve)
        assert recorded is True
        assert b.is_recording is True
    assert len(b.buffer._current) == 16

    # 3. Push 6th static salve: reaches 6 static ticks with len >= 11 -> stops recording
    recorded = b.record(static_salve)
    assert recorded is True
    assert b.is_recording is False
    assert len(b.buffer._current) == 17

    # 4. Push further static salves: must NOT be recorded
    for _ in range(5):
        recorded = b.record(static_salve)
        assert recorded is False
        assert b.is_recording is False
        assert len(b.buffer._current) == 17  # size unchanged

    # Context and current segment must NOT be discarded/boundary'd
    assert len(b.buffer._segments) == 0
    assert len(b.buffer._current) == 17

    # 5. Tokens move again: recording must immediately resume into same full context
    moved_salve = _make_salve([0.9] * 53, [0.5] * 5)
    recorded = b.record(moved_salve)
    assert recorded is True
    assert b.is_recording is True
    assert len(b.buffer._current) == 18
    assert len(b.buffer._segments) == 0


def test_brain_record_too_short_sequence_not_paused():
    b = Brain(seed=42)
    # Sequence starts with 3 moving salves (< seq_len 11)
    for i in range(3):
        b.record(_make_salve([0.1 * i] * 53, [0.5] * 5))
    assert len(b.buffer._current) == 3

    # Push 6 identical salves (static for 6 ticks, but len reaches only 9 < 11)
    static_salve = _make_salve([0.2] * 53, [0.5] * 5)
    for _ in range(6):
        recorded = b.record(static_salve)
        assert recorded is True
        # Must NOT pause because the sequence in progress is not long enough yet
        assert b.is_recording is True
    assert len(b.buffer._current) == 9

    # Push 2 more to reach seq_len = 11 (static_ticks is now 8 >= 6)
    b.record(static_salve)
    assert len(b.buffer._current) == 10
    assert b.is_recording is True

    b.record(static_salve)
    assert len(b.buffer._current) == 11
    # Now sequence is sufficiently long (11 >= 11) and static >= 6 -> pauses!
    assert b.is_recording is False

    # Next static salve is not recorded
    assert b.record(static_salve) is False
    assert len(b.buffer._current) == 11


def test_brain_record_ignores_metabolic_drift():
    b = Brain(seed=42)
    # Sequence of 11 moving salves (0.05 * 10 = 0.5)
    for i in range(11):
        b.record(_make_salve([0.05 * i] * 53, [0.5] * 5))

    # Static in animal, env, consignes, but courbature (slot 47) drifts every tick
    for i in range(6):
        ss = [0.5] * 53
        ss[47] = 0.01 * (i + 1)  # courbature drift
        b.record(_make_salve(ss, [0.5] * 5))

    # Should pause after 6 ticks because metabolic drift is ignored
    assert b.is_recording is False


def test_coreset_addendum_prune_commit():
    buf = ExperienceBuffer(seq_len=5, capacity=200, pool_capacity=200, seed=42)
    # Populate Coreset with 40 sequences directly
    for i in range(40):
        buf._coreset.append([[[float(i)] * 25] * 16 for _ in range(5)])

    assert buf.coreset_size == 40
    assert buf.addendum_size == 0
    assert len(buf) == 40

    # Prune 33% (default 33% of 40 = 13 sequences)
    pruned = buf.prune_coreset()
    assert pruned == 13
    assert buf.coreset_size == 27
    assert buf.addendum_size == 13
    assert len(buf) == 40

    # Filter addendum (keep only 1 sequence)
    buf.filter_addendum([buf._addendum[0]])
    assert buf.addendum_size == 1

    # Commit addendum
    committed = buf.commit_addendum()
    assert committed == 1
    assert buf.coreset_size == 28
    assert buf.addendum_size == 0
    assert len(buf) == 28


def test_addendum_surprise_evaluation():
    b = Brain(seed=42)
    # Build a constant sequence
    c_salve = _make_salve([0.2] * 53, [0.5] * 5)
    seq_easy = [c_salve] * 11

    # Build an erratic sequence with high cost (douleur = 1.0)
    erratic_salve = _make_salve([0.9] * 53, [-0.8] * 5)
    seq_hard = [c_salve] * 5 + [erratic_salve] * 6

    surprises = b.evaluate_sequences_surprise([seq_easy, seq_hard])
    assert len(surprises) == 2
    assert all(s >= 0.0 for s in surprises)


def test_naive_sequence_distances_and_clustering():
    from .brain import pairwise_sequence_distances, complete_linkage_clustering, select_medoids
    s1 = _make_salve([0.1] * 53, [0.2] * 5)
    s2 = _make_salve([0.101] * 53, [0.2] * 5)  # very close to s1
    s3 = _make_salve([0.8] * 53, [-0.5] * 5)  # distant

    seq1 = [s1] * 11
    seq2 = [s2] * 11
    seq3 = [s3] * 11

    D = pairwise_sequence_distances([seq1, seq2, seq3])
    assert D.shape == (3, 3)
    assert D[0, 0] == 0.0
    assert D[0, 1] < 0.02   # seq1 and seq2 are near-identical
    assert D[0, 2] > 0.30   # seq1 and seq3 are far apart

    clusters = complete_linkage_clustering(D, eps=0.04)
    assert len(clusters) == 2  # {seq1, seq2} and {seq3}
    medoids = select_medoids(clusters, D)
    assert len(medoids) == 2


def test_experience_buffer_intra_addendum_deduplication():
    buf = ExperienceBuffer(seq_len=5, seed=42)
    s_dup = _make_salve([0.3] * 53, [0.1] * 5)
    s_unique = _make_salve([0.9] * 53, [-0.7] * 5)

    seq_dup = [s_dup] * 5
    seq_uniq = [s_unique] * 5

    # 10 identical sequences + 1 unique sequence
    for _ in range(10):
        buf._addendum.append(seq_dup)
        buf._addendum_meta.append("wake")
    buf._addendum.append(seq_uniq)
    buf._addendum_meta.append("wake")

    assert len(buf._addendum) == 11
    n_dropped = buf.deduplicate_addendum(eps=0.04)
    assert n_dropped == 9
    assert len(buf._addendum) == 2


def test_experience_buffer_cross_deduplication_and_memory_decay():
    buf = ExperienceBuffer(seq_len=5, pool_capacity=3, seed=42)
    s1 = _make_salve([0.1] * 53, [0.1] * 5)
    s2 = _make_salve([0.5] * 53, [0.5] * 5)
    s3 = _make_salve([0.9] * 53, [-0.9] * 5)

    seq1 = [s1] * 5
    seq2 = [s2] * 5
    seq3 = [s3] * 5

    # 1. Cold start: add seq1 and seq2 to Coreset
    buf._addendum = [seq1, seq2]
    res1 = buf.consolidate_addendum_into_coreset(eps=0.04, decay=0.0)
    assert res1["added"] == 2
    assert buf.coreset_size == 2
    assert buf._coreset_vivacity == [1.0, 1.0]

    # 2. Addendum contains a duplicate of seq1 and a novel seq3
    s1_noisy = _make_salve([0.1005] * 53, [0.1] * 5)
    seq1_noisy = [s1_noisy] * 5

    # First decay Coreset by 0.20 -> vivacities become 0.80
    buf._coreset_vivacity = [0.80, 0.80]

    buf._addendum = [seq1_noisy, seq3]
    # Consolidate with decay=0.01
    res2 = buf.consolidate_addendum_into_coreset(eps=0.04, decay=0.01)

    assert res2["refreshed"] == 1  # seq1 was refreshed
    assert res2["added"] == 1      # seq3 was added
    assert buf.coreset_size == 3

    # seq1 vivacity was 1.0 - 0.01 = 0.99
    # seq2 vivacity was 0.80 - 0.01 = 0.79
    # seq3 vivacity was 1.0 - 0.01 = 0.99
    assert abs(buf._coreset_vivacity[0] - 0.99) < 1e-3
    assert abs(buf._coreset_vivacity[1] - 0.79) < 1e-3
    assert abs(buf._coreset_vivacity[2] - 0.99) < 1e-3

    # 3. Test capacity eviction: capacity is 3, add 1 more novel sequence
    s4 = _make_salve([0.35] * 53, [0.0] * 5)
    seq4 = [s4] * 5
    buf._addendum = [seq4]
    res3 = buf.consolidate_addendum_into_coreset(eps=0.04, decay=0.0)

    assert buf.coreset_size == 3
    assert res3["evicted"] == 1
    # seq2 (lowest vivacity 0.79) was evicted!
    assert seq2 not in buf._coreset


def test_wm_theater_rollout_and_rendering():
    """Verify World Model Theater rollout generation and headless rendering."""
    import pygame
    from lambda_barre.brain import Brain, ExperienceBuffer
    from lambda_barre.wm_theater import WMTheater, rollout_sequence

    brain = Brain()
    buf = ExperienceBuffer(seq_len=11, seed=42)

    # Create 2 mock sequences of 11 salves (16 tokens each)
    mock_seqs = []
    for s_idx in range(2):
        seq = []
        for i in range(11):
            salve = []
            for t_idx in range(16):
                tok = [0.0] * 25
                if t_idx == 10:  # Cost token
                    tok[9:14] = [0.1 * (i + 1), 0.05, 0.02, 0.01, 0.0]
                salve.append(tok)
            seq.append(salve)
        mock_seqs.append(seq)

    buf._coreset = mock_seqs
    buf._coreset_vivacity = [1.0, 0.9]

    # Test rollout_sequence logic
    rec = rollout_sequence(brain, mock_seqs[0], 0, 2, 1.0)
    assert rec.seq_idx == 0
    assert len(rec.steps) == 11
    # First 5 steps (s0..s4) are context only
    for s_i in range(5):
        assert rec.steps[s_i].pred_step is None
        assert rec.steps[s_i].pred_costs is None
        assert len(rec.steps[s_i].real_costs) == 5
    # Steps 5..10 are predicted
    for s_i in range(5, 11):
        assert rec.steps[s_i].pred_step is not None
        assert rec.steps[s_i].pred_costs is not None
        assert len(rec.steps[s_i].pred_costs) == 5
        assert rec.steps[s_i].posture_rmse is not None
        assert rec.steps[s_i].cost_rmse is not None

    # Test WMTheater GUI rendering in headless mode
    theater = WMTheater(brain, buf, width=1140, height=700)
    assert theater.num_seqs == 2
    assert theater.seq_idx == 0

    pygame.init()
    surf = pygame.Surface((1140, 700))
    font = pygame.font.SysFont("monospace", 13, bold=True)
    font_small = pygame.font.SysFont("monospace", 11)
    font_tiny = pygame.font.SysFont("monospace", 9)

    theater.draw(surf, font, font_small, font_tiny)

    theater.next_seq()
    assert theater.seq_idx == 1
    theater.draw(surf, font, font_small, font_tiny)

    theater.prev_seq()
    assert theater.seq_idx == 0

    # Event handling
    ev_next = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT)
    theater.handle_event(ev_next)
    assert theater.seq_idx == 1


def test_saliency_weighted_distances_and_consolidation():
    import torch
    from .brain import pairwise_sequence_distances, cross_sequence_distances, Brain, ExperienceBuffer
    b = Brain(device="cpu")

    # 1. Cold start / empty buffer returns ones
    w_empty = b.compute_saliency_weights()
    assert w_empty.shape == (2816,)
    assert torch.allclose(w_empty, torch.ones(2816))

    # 2. Distance equality when weights are ones
    s1 = _make_salve([0.1] * 53, [0.2] * 5)
    s2 = _make_salve([0.4] * 53, [-0.3] * 5)
    seq1 = [s1] * 11
    seq2 = [s2] * 11

    D_none = pairwise_sequence_distances([seq1, seq2], weights=None)
    D_ones = pairwise_sequence_distances([seq1, seq2], weights=torch.ones(2816))
    assert torch.allclose(D_none, D_ones)

    cross_none = cross_sequence_distances([seq1], [seq2], weights=None)
    cross_ones = cross_sequence_distances([seq1], [seq2], weights=torch.ones(2816))
    assert torch.allclose(cross_none, cross_ones)

    # 3. Sensitivity weighting: populate buffer and compute weights
    for _ in range(5):
        for _ in range(12):
            b.record(_make_salve([0.2] * 53, [0.1] * 5))
        b.buffer.boundary()
    b.buffer.extract_addendum()
    b.buffer.commit_addendum()

    weights = b.compute_saliency_weights(batch_size=4)
    assert weights.shape == (2816,)
    assert (weights >= 0.0).all()
    assert abs(weights.mean().item() - 1.0) < 1e-4

    # 4. ExperienceBuffer deduplication with weights
    buf = ExperienceBuffer(seq_len=11, seed=42)
    buf._addendum = [seq1, seq1, seq2]
    n_drop = buf.deduplicate_addendum(eps=0.04, weights=weights)
    assert n_drop == 1
    assert len(buf._addendum) == 2


if __name__ == "__main__":
    test_saliency_weighted_distances_and_consolidation()
    test_policy_outputs_are_valid_consignes()
    test_world_model_forward_and_predict_shapes()
    test_salve_cost_sign()
    test_act_returns_five_consignes()
    test_naive_sequence_distances_and_clustering()
    test_experience_buffer_intra_addendum_deduplication()
    test_experience_buffer_cross_deduplication_and_memory_decay()
    test_sleep_trains_both_models_and_world_loss_decreases()
    test_brain_drives_real_sim_and_sleeps()
    test_experience_buffer_linear_recording_and_sliding_windows()
    test_experience_buffer_boundary_and_discontinuity_isolation()
    test_experience_buffer_capacity_eviction()
    test_experience_buffer_clear()
    test_coreset_addendum_prune_commit()
    test_addendum_surprise_evaluation()
    test_brain_boundary_and_history_isolation()
    test_brain_record_stationary_pause_and_resume()
    test_brain_record_too_short_sequence_not_paused()
    test_brain_record_ignores_metabolic_drift()
    test_experience_buffer_two_tier_consolidation_and_persistence()
    test_explore_action_batch_bounds_and_sigma()
    test_train_policy_candidate_selection()
    test_sleep_preserves_wake_history_context()
    test_reset_restores_limb_distances()
    test_three_futures_bellman_optimism()
    test_dream_record_and_dream_theater_rendering()
    test_wm_theater_rollout_and_rendering()
    print("all brain tests passed")


