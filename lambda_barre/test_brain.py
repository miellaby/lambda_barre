"""End-to-end test for the 2-model brain (no pygame window).

Exercises the full pipeline: encode a salve, journal transitions, run a sleep
cycle that trains both the world model and the policy, and check that the
world-model loss decreases on structured data and that the policy produces
valid consignes.

Run with the project on PYTHONPATH:
    PYTHONPATH=. python -m pytest lambda_barre/test_brain.py -q
or directly:
    PYTHONPATH=. python lambda_barre/test_brain.py
"""
from __future__ import annotations

import random

from . import tokenize as T
from .brain import Brain
from .models import Policy, WorldModel


def _make_salve(scalars: list[int], action_vals: list[int]):
    """Build a 67-token salve from 55 scalar values and 5 action scalar values."""
    svals = T.state_vals_from_scalars(scalars)
    avals = [0] + action_vals  # ACTION sep (value 0) + 5 scalars
    return list(zip(T.SALVE_IDS, svals + avals))


def test_policy_outputs_are_valid_consignes():
    import torch
    p = Policy()
    sv = torch.rand(4, 55)
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
    seq = T.STATE_IDS + T.ACTION_IDS + T.STATE_IDS  # 128
    ids = torch.tensor([seq] * 3)
    vals = torch.randint(0, 256, (3, 128))
    out = wm(ids, vals)
    assert out.shape == (3, 128, 256)
    nxt = wm.predict_next(ids[:, :67], vals[:, :67])
    assert nxt.shape == (3, T.STATE_LEN)


def test_salve_cost_sign():
    # reward_pos is signed: token value 0 -> -1.0 (reward, negative cost),
    # token value 255 -> +1.0 (penalty). reward_neg is unsigned cost.
    scalars = [0] * 55
    sv = T.state_vals_from_scalars(scalars)
    # reward (reward_pos=0 -> -1.0), no cost -> negative total cost
    sv[T._REWARD_POS_OFF] = 0
    sv[T._REWARD_NEG_OFF] = 0
    salve = list(zip(T.SALVE_IDS, sv + [0, 0, 0, 0, 0, 0]))
    assert T.salve_cost(salve) < 0.0
    # punish (reward_neg=255 -> +1.0), no reward -> positive total cost
    sv[T._REWARD_POS_OFF] = 128   # neutral (dequant ~0)
    sv[T._REWARD_NEG_OFF] = 255
    salve = list(zip(T.SALVE_IDS, sv + [0, 0, 0, 0, 0, 0]))
    assert T.salve_cost(salve) > 0.0


def test_sleep_trains_both_models_and_world_loss_decreases():
    random.seed(1)
    b = Brain(seed=1)
    # Structured world: next state ≈ current state (near-static), with low cost
    # (reward_pos high, reward_neg 0). This gives the world model something
    # learnable: predict the next state ≈ the current state.
    for _ in range(160):
        scalars = [random.randint(0, 255) for _ in range(55)]
        svals = T.state_vals_from_scalars(scalars)
        svals[T._REWARD_POS_OFF] = 255
        svals[T._REWARD_NEG_OFF] = 0
        salve = list(zip(T.SALVE_IDS, svals + [0, 128, 128, 128, 128, 128]))
        nxt = list(salve)  # next state equals current state
        b.record(salve, nxt)
    assert len(b.buffer) == 160
    s1 = b.sleep(wm_epochs=4, wm_batch=32, pol_steps=8, pol_batch=12, horizon=2)
    s2 = b.sleep(wm_epochs=4, wm_batch=32, pol_steps=8, pol_batch=12, horizon=2)
    # losses are finite
    for k in ("wm_loss", "pol_loss", "return"):
        assert s2[k] == s2[k], f"{k} is NaN"   # NaN check
    # world-model loss should drop on this learnable (near-identity) data
    assert s2["wm_loss"] < s1["wm_loss"], (s1["wm_loss"], s2["wm_loss"])


def test_act_returns_five_consignes():
    random.seed(2)
    b = Brain(seed=2)
    salve = _make_salve([random.randint(0, 255) for _ in range(55)],
                        [128, 128, 128, 128, 128])
    a = b.act(salve)
    assert len(a) == 5


def test_brain_drives_real_sim_and_sleeps():
    """End-to-end: the brain drives the real pymunk body, the real sensors +
    encoder produce salves, transitions are journaled, and a sleep cycle
    trains both models off that real experience. No display is needed."""
    from . import body as B, world as W, sensors as S, extero as E, intero as I
    from .tokenize import TokenEncoder
    space = W.make_space()
    skel = B.build_skeleton(space)
    B.apply_consignes(skel)
    proprio = S.Proprio(space, skel)
    reward = S.Reward(skel)
    cursor = E.Cursor(skel)
    vision = E.Vision(skel, space)
    touch = E.Touch(space, skel)
    intero = I.Intero()
    encoder = TokenEncoder()
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
    assert len(brain.buffer) > 10
    stats = brain.sleep(wm_epochs=2, wm_batch=24, pol_steps=4, pol_batch=12, horizon=2)
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
