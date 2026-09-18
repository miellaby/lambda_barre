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
    sv = torch.rand(4, 143)
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
    # Flat state-signal layout: token 11 (Coûts) holds effort, douleur,
    # courbature, instabilite, vertige at indices 47..51; token 12 (Récompense)
    # holds confort at index 52.
    # all-zero: confort=0 -> -1.0 (max reward) -> cost = -1.0 < 0
    salve = _make_salve([0.0] * 53, [0.5] * 5)
    assert T.salve_cost(salve) < 0.0
    # douleur=1.0 (->1.0), confort=0.5 (->0.0) -> cost = 4.0 > 0
    ss = [0.0] * 53
    ss[48] = 1.0      # douleur slot
    ss[52] = 0.5      # confort slot
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
    # Push salves into brain
    for _ in range(15):
        salve = _make_salve([0.1] * 53, [0.5] * 5)
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


def test_train_policy_minimal_improvement_margin():
    import torch
    # Verify candidate ranking logic with 2% margin:
    # Row 0: small 1% noise fluctuation -> should stay on candidate 0 (policy)
    # Row 1: real improvement of 5% -> should switch to candidate 2
    # Row 2: demonstrated baseline is best -> should pick candidate 1
    candidate_costs = torch.tensor([
        [5.00, 5.10, 4.96, 4.97],  # 4.96 is only 0.8% better than 5.00 (below 2% = 0.10)
        [5.00, 5.20, 4.70, 4.80],  # 4.70 is 6% better than 5.00 (exceeds 2% margin)
        [5.50, 4.00, 4.30, 4.20],  # demonstrated is 4.00, best of all
    ])
    cost_baseline = torch.minimum(candidate_costs[:, 0], candidate_costs[:, 1])
    margin = 0.02 * cost_baseline.abs().clamp(min=1.0)
    effective_costs = candidate_costs.clone()
    effective_costs[:, 2:] += margin.unsqueeze(1)

    best_idx = effective_costs.argmin(dim=1)
    assert best_idx.tolist() == [0, 2, 1]


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


if __name__ == "__main__":
    test_policy_outputs_are_valid_consignes()
    test_world_model_forward_and_predict_shapes()
    test_salve_cost_sign()
    test_act_returns_five_consignes()
    test_sleep_trains_both_models_and_world_loss_decreases()
    test_brain_drives_real_sim_and_sleeps()
    test_experience_buffer_linear_recording_and_sliding_windows()
    test_experience_buffer_boundary_and_discontinuity_isolation()
    test_experience_buffer_capacity_eviction()
    test_experience_buffer_clear()
    test_brain_boundary_and_history_isolation()
    test_experience_buffer_two_tier_consolidation_and_persistence()
    test_explore_action_batch_bounds_and_sigma()
    test_train_policy_minimal_improvement_margin()
    test_sleep_preserves_wake_history_context()
    print("all brain tests passed")
