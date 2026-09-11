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
from .brain import Brain
from .models import Policy, WorldModel


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
        assert -1.58 < tl < 1.58 and -1.58 < tr < 1.58 and -1.58 < tq < 1.58
        assert 10.0 - 1e-3 <= dl <= 32.0 + 1e-3
        assert 10.0 - 1e-3 <= dr <= 32.0 + 1e-3


def test_world_model_forward_and_predict_shapes():
    import torch
    wm = WorldModel()
    L = T.SALVE_TOKENS + T.SALVE_TOKENS + T.STATE_TOKENS  # 16+16+13 = 45
    x = torch.rand(3, L, T.DENSE_DIM)
    out = wm(x)
    assert out.shape == (3, L, T.N_SIGNAL)
    assert not torch.isnan(out).any().item()
    # predict_next: ctx = [B, 16, 25] (one state+action salve) -> [B, 13, 25]
    ctx = torch.rand(3, T.SALVE_TOKENS, T.DENSE_DIM)
    nxt = wm.predict_next(ctx)
    assert nxt.shape == (3, T.STATE_TOKENS, T.DENSE_DIM)
    sigs = nxt[:, :, 9:]
    assert bool((sigs >= 0).all().item()) and bool((sigs <= 1).all().item())


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
        nxt = list(salve)  # next state equals current state
        b.record(salve, nxt)
    assert len(b.buffer) >= 1  # at least one complete 10-step sequence
    s1 = _run_sleep(b, wm_epochs=4, wm_batch=32, pol_steps=8, pol_batch=12)
    s2 = _run_sleep(b, wm_epochs=4, wm_batch=32, pol_steps=8, pol_batch=12)
    # losses are finite
    for k in ("wm_loss", "pol_loss", "return"):
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
    from . import body as B, world as W, sensors as S, extero as E, intero as I
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
    prev = None
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
            if prev is not None:
                brain.record(prev, salve)
            tl, dl, tr, dr, tq = brain.act(salve)
            skel.limb_l.theta_star = tl; skel.limb_l.d_star = dl
            skel.limb_r.theta_star = tr; skel.limb_r.d_star = dr
            skel.tail_act.theta_star = tq
            prev = salve
    assert len(brain.buffer) >= 1  # at least one complete sequence from 4s sim
    stats = _run_sleep(brain, wm_epochs=2, wm_batch=24, pol_steps=4, pol_batch=12)
    for k in ("wm_loss", "pol_loss"):
        assert stats[k] == stats[k]   # finite
    a = brain.act(prev)
    assert len(a) == 5


if __name__ == "__main__":
    test_policy_outputs_are_valid_consignes()
    test_world_model_forward_and_predict_shapes()
    test_salve_cost_sign()
    test_act_returns_five_consignes()
    test_sleep_trains_both_models_and_world_loss_decreases()
    test_brain_drives_real_sim_and_sleeps()
    print("all brain tests passed")
