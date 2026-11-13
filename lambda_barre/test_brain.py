"""Tests du pipeline refonte (REFONTE.md) : séquence enregistrée réordonnée
(contexte | dénouement | conséquence inversée | action de décision), un seul
masque causal, loss à poids 0 sur le contexte, inférence par échantillonnage du
dénouement + élection f(probable, favorable), policy pondérée par p.detach()."""
from __future__ import annotations

import random

import pytest
import torch

from lambda_barre.brain import (Brain, ExperienceBuffer, build_sequence,
                                build_target_tables, compute_seq_layout,
                                compute_salve_positions, relief_score,
                                sequence_to_window)
from lambda_barre.tokenize import (ACTION_TOKENS, DENSE_DIM, N_SIGNAL,
                                   SALVE_TOKENS, SIG_OFFSET, STATE_TOKENS,
                                   make_salve)

SEQ_STEPS = 10
N_CTX = 4


def _rand_salve(rng):
    return make_salve([rng.random() for _ in range(53)],
                      [rng.random() for _ in range(5)])


def _rand_window(rng, n=SEQ_STEPS + 1):
    return [_rand_salve(rng) for _ in range(n)]


@pytest.fixture
def rng():
    return random.Random(1234)


@pytest.fixture
def brain():
    torch.manual_seed(0)
    return Brain(seq_steps=SEQ_STEPS, n_ctx=N_CTX,
                 buffer_capacity=200, buffer_pool_capacity=200)


# --- layout -------------------------------------------------------------------
def test_layout_positions():
    layout = compute_seq_layout(SEQ_STEPS, N_CTX)
    assert layout["ctx_len"] == N_CTX * SALVE_TOKENS + STATE_TOKENS      # 77
    assert layout["eos_pos"] == layout["ctx_len"]                         # dénouement juste après le contexte
    assert layout["conseq_len"] == (SEQ_STEPS - N_CTX) * STATE_TOKENS     # 78
    assert layout["action_pos"] == layout["conseq_start"] + layout["conseq_len"]
    assert layout["seq_len"] == layout["action_pos"] + ACTION_TOKENS      # 159


def test_salve_positions():
    sp = compute_salve_positions(SEQ_STEPS, N_CTX)
    layout = compute_seq_layout(SEQ_STEPS, N_CTX)
    assert len(sp) == layout["seq_len"]
    assert sp[layout["eos_pos"]] == SEQ_STEPS
    # conséquence inversée : positions décroissantes seq_steps..n_ctx+1
    conseq = sp[layout["conseq_start"]:layout["action_pos"]]
    assert conseq[0] == SEQ_STEPS and conseq[-1] == N_CTX + 1
    assert all(conseq[i * STATE_TOKENS] == SEQ_STEPS - i for i in range(SEQ_STEPS - N_CTX))
    # action de décision à la position n_ctx
    assert sp[layout["action_pos"]:] == [N_CTX] * ACTION_TOKENS


def test_target_tables_prompt_context():
    valid, weight = build_target_tables(SEQ_STEPS, N_CTX)
    layout = compute_seq_layout(SEQ_STEPS, N_CTX)
    seq_len = layout["seq_len"]
    assert valid.shape == (seq_len - 1, N_SIGNAL)
    # contexte = prompt : poids 0 sur toutes les lignes prédisant un token du contexte
    assert weight[:layout["ctx_len"] - 1].sum() == 0.0
    # à partir du dénouement (row ctx_len-1 prédit le token eos_pos) : poids 1
    # sur les slots valides, 0 sur le padding
    tail_w = weight[layout["ctx_len"] - 1:]
    tail_v = valid[layout["ctx_len"] - 1:]
    assert (tail_w[tail_v] == 1.0).all()
    assert (tail_w[~tail_v] == 0.0).all()
    # le dénouement a 6 slots valides
    assert valid[layout["eos_pos"] - 1, :6].all()
    assert not valid[layout["eos_pos"] - 1, 6:].any()


# --- séquence enregistrée -------------------------------------------------------
def test_build_sequence_order(rng):
    window = _rand_window(rng)
    seq = build_sequence(window, n_ctx=N_CTX)
    layout = compute_seq_layout(SEQ_STEPS, N_CTX)
    assert len(seq) == layout["seq_len"]
    assert all(len(tok) == DENSE_DIM for tok in seq)
    # contexte : salves complètes 0..n_ctx-1 puis s_{n_ctx} seul
    for i in range(N_CTX):
        assert seq[i * SALVE_TOKENS:(i + 1) * SALVE_TOKENS] == window[i]
    assert seq[N_CTX * SALVE_TOKENS:layout["ctx_len"]] == window[N_CTX][:STATE_TOKENS]
    # conséquence inversée : s_{seq_steps} .. s_{n_ctx+1}, états seuls
    for j, k in enumerate(range(SEQ_STEPS, N_CTX, -1)):
        assert seq[layout["conseq_start"] + j * STATE_TOKENS:
                   layout["conseq_start"] + (j + 1) * STATE_TOKENS] == window[k][:STATE_TOKENS]
    # action de décision = a_{n_ctx}, derniers tokens de la séquence
    assert seq[layout["action_pos"]:] == window[N_CTX][STATE_TOKENS:]


def test_build_sequence_eos_token(rng):
    window = _rand_window(rng)
    seq = build_sequence(window, n_ctx=N_CTX)
    layout = compute_seq_layout(SEQ_STEPS, N_CTX)
    eos_tok = seq[layout["eos_pos"]]
    # préfixe intero + canal dédié, 6 slots dans [0,1], reste à 0
    assert eos_tok[0] == 0.0
    assert all(0.0 <= v <= 1.0 for v in eos_tok[SIG_OFFSET:SIG_OFFSET + 6])
    assert all(v == 0.0 for v in eos_tok[SIG_OFFSET + 6:])


def test_sequence_window_roundtrip(rng):
    window = _rand_window(rng)
    seq = build_sequence(window, n_ctx=N_CTX)
    back = sequence_to_window(seq, n_ctx=N_CTX, seq_steps=SEQ_STEPS)
    assert len(back) == SEQ_STEPS + 1
    # le contexte et l'action de décision sont reconstruits à l'identique
    for i in range(N_CTX):
        assert back[i] == window[i]
    assert back[N_CTX][:STATE_TOKENS] == window[N_CTX][:STATE_TOKENS]
    assert back[N_CTX][STATE_TOKENS:] == window[N_CTX][STATE_TOKENS:]
    # les états de la conséquence inversée sont réinversés
    for k in range(N_CTX + 1, SEQ_STEPS + 1):
        assert back[k][:STATE_TOKENS] == window[k][:STATE_TOKENS]


# --- buffer ---------------------------------------------------------------------
def test_extract_addendum_builds_recorded_sequences(brain, rng):
    for _ in range(SEQ_STEPS + 12):
        brain.record_tick(_rand_salve(rng))
    n = brain.buffer.extract_addendum()
    assert n == 12   # 22 salves - 11 (seq_len) + 1 fenêtres glissantes
    assert brain.buffer.addendum_size == 12
    seq = brain.buffer._addendum[0]
    assert len(seq) == brain.seq_len


def test_buffer_save_load_roundtrip(brain, rng, tmp_path):
    for _ in range(SEQ_STEPS + 5):
        brain.record_tick(_rand_salve(rng))
    brain.buffer.extract_addendum()
    brain.buffer.commit_addendum()
    p = str(tmp_path / "buf.pt")
    brain.buffer.save(p)
    buf2 = ExperienceBuffer(seq_len=SEQ_STEPS + 1, n_ctx=N_CTX)
    assert buf2.load(p)
    assert buf2.coreset_size == brain.buffer.coreset_size
    assert len(buf2._coreset[0]) == brain.seq_len


def test_buffer_load_rejects_legacy_layout(brain, tmp_path):
    # un checkpoint legacy stocke des fenêtres de salves temporelles (11 salves
    # de 16 tokens), pas des séquences plates de seq_len tokens
    p = str(tmp_path / "legacy.pt")
    torch.save({"coreset": [[_rand_salve(random.Random(1)) for _ in range(11)]],
                "seq_len": SEQ_STEPS + 1, "n_ctx": N_CTX}, p)
    buf2 = ExperienceBuffer(seq_len=SEQ_STEPS + 1, n_ctx=N_CTX)
    assert not buf2.load(p)
    assert buf2.coreset_size == 0


# --- WM training -----------------------------------------------------------------
def test_batch_tensors_shape_and_intero_mask(brain, rng):
    seqs = [build_sequence(_rand_window(rng), n_ctx=N_CTX) for _ in range(4)]
    vals = brain._batch_tensors(seqs)
    assert vals.shape == (4, brain.seq_len, DENSE_DIM)
    # slots intero des salves état masqués (contexte + conséquence inversée)
    intero_pos = [k * SALVE_TOKENS + 12 for k in range(N_CTX + 1)]
    intero_pos += [brain.conseq_start + j * STATE_TOKENS + 12 for j in range(brain.n_imagine)]
    assert (vals[:, intero_pos, SIG_OFFSET:] == 0.0).all()


def test_wm_single_mask_is_causal(brain):
    from lambda_barre.models import build_salve_mask
    assert torch.equal(brain.mask, build_salve_mask(brain.seq_len, brain.device))


def test_wm_train_step_decreases_loss(brain, rng):
    seqs = [build_sequence(_rand_window(rng), n_ctx=N_CTX) for _ in range(8)]
    first = brain._train_wm_batch(seqs)
    last = first
    for _ in range(30):
        last = brain._train_wm_batch(seqs)
    assert last < first


# --- inférence -------------------------------------------------------------------
def test_predict_denouement_shape(brain, rng):
    seqs = [build_sequence(_rand_window(rng), n_ctx=N_CTX) for _ in range(4)]
    vals = brain._batch_tensors(seqs)
    mu = brain.predict_denouement(vals[:, :brain.ctx_len])
    assert mu.shape == (4, 6)
    assert ((mu >= 0.0) & (mu <= 1.0)).all()


def test_sample_denouements_shapes_and_election(brain, rng):
    seqs = [build_sequence(_rand_window(rng), n_ctx=N_CTX) for _ in range(4)]
    vals = brain._batch_tensors(seqs)
    ctx = vals[:, :brain.ctx_len]
    eos_elu, p = brain.sample_denouements(ctx, k=16, temperature=0.1, top_p=0.9)
    assert eos_elu.shape == (4, 6) and p.shape == (4,)
    assert ((eos_elu >= 0.0) & (eos_elu <= 1.0)).all()
    assert ((p > 0.0) & (p <= 1.0)).all()


def test_sample_denouements_keeps_satisfying_observed(brain, rng):
    seqs = [build_sequence(_rand_window(rng), n_ctx=N_CTX) for _ in range(2)]
    vals = brain._batch_tensors(seqs)
    ctx = vals[:, :brain.ctx_len]
    eos_obs = vals[:, brain.eos_pos, SIG_OFFSET:SIG_OFFSET + 6]
    brain.eos_satisfying = -1.0   # tout dénouement observé est "satisfaisant"
    eos_elu, p = brain.sample_denouements(ctx, eos_observed=eos_obs)
    assert torch.allclose(eos_elu, eos_obs)
    assert (p == 1.0).all()


def test_generate_from_denouement(brain, rng):
    seqs = [build_sequence(_rand_window(rng), n_ctx=N_CTX) for _ in range(2)]
    vals = brain._batch_tensors(seqs)
    ctx = vals[:, :brain.ctx_len]
    eos_elu, _ = brain.sample_denouements(ctx)
    gen = brain.generate_from_denouement(ctx, eos_elu)
    assert gen["full"].shape == (2, brain.seq_len, DENSE_DIM)
    assert gen["conseq"].shape == (2, brain.conseq_len, DENSE_DIM)
    assert gen["action"].shape == (2, ACTION_TOKENS, DENSE_DIM)
    # l'action de décision est le dernier token de la séquence générée
    assert torch.equal(gen["action"], gen["full"][:, brain.action_pos:])


# --- policy ----------------------------------------------------------------------
def test_policy_step_ponderee_par_p(brain, rng):
    for _ in range(SEQ_STEPS + 8):
        brain.record_tick(_rand_salve(rng))
    brain.buffer.extract_addendum()
    losses = [v for _, v in brain.train_policy(steps=2, batch=4)]
    assert len(losses) == 2
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    assert brain.last_dream_record is not None
    assert brain.last_dream_record.eos_forced is not None


# --- sleep cycle -------------------------------------------------------------------
def test_sleep_cycle_smoke(brain, rng):
    for _ in range(SEQ_STEPS + 10):
        brain.record_tick(_rand_salve(rng))
    labels = []
    for lbl, v in brain.sleep_cycle(wm_coreset_epoch=1, wm_addendum_epoch=1,
                                    wm_batch=8, pol_steps=1, pol_batch=4):
        labels.append(lbl)
    assert "wm_addendum" in labels and "pol" in labels and labels[-1] == "done"
    assert brain.buffer.coreset_size > 0
