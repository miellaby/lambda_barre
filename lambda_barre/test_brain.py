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
    assert layout["eos_pos"] == layout["ctx_len"]                         # token de dénouement juste après le contexte
    assert layout["conseq_start"] == layout["eos_pos"]                    # 1er token de s_{seq_steps} inversée
    assert layout["conseq_len"] == (SEQ_STEPS - N_CTX - 1) * SALVE_TOKENS + STATE_TOKENS  # 93
    assert layout["action_pos"] == layout["conseq_start"] + layout["conseq_len"]
    assert layout["seq_len"] == layout["action_pos"] + ACTION_TOKENS      # 173


def test_salve_positions():
    sp = compute_salve_positions(SEQ_STEPS, N_CTX)
    layout = compute_seq_layout(SEQ_STEPS, N_CTX)
    assert len(sp) == layout["seq_len"]
    assert sp[layout["eos_pos"]] == SEQ_STEPS + 1     # PE dédié du dénouement (hors fenêtre)
    # conséquence rétro-causale : unités de 16 tokens, k = seq_steps..n_ctx+1
    conseq = sp[layout["conseq_start"]:layout["action_pos"]]
    n_im = SEQ_STEPS - N_CTX
    for j in range(n_im - 1):
        k = SEQ_STEPS - j
        t0 = 1 if j == 0 else 0     # unité 0 : le token 0 est le dénouement (PE dédié)
        assert all(conseq[j * SALVE_TOKENS + t] == k for t in range(t0, STATE_TOKENS))
        assert all(conseq[j * SALVE_TOKENS + STATE_TOKENS + t] == k - 1
                   for t in range(ACTION_TOKENS))
    # dernière unité : s_{n_ctx+1} seule (sans action)
    j = n_im - 1
    assert all(conseq[j * SALVE_TOKENS + t] == N_CTX + 1 for t in range(STATE_TOKENS))
    # action de décision à la position n_ctx
    assert sp[layout["action_pos"]:] == [N_CTX] * ACTION_TOKENS


def test_target_tables_prompt_context():
    valid, weight = build_target_tables(SEQ_STEPS, N_CTX)
    layout = compute_seq_layout(SEQ_STEPS, N_CTX)
    seq_len = layout["seq_len"]
    assert valid.shape == (seq_len - 1, N_SIGNAL)
    # contexte = prompt : poids 0 sur toutes les lignes prédisant un token du contexte
    assert weight[:layout["ctx_len"] - 1].sum() == 0.0
    # génération dans l'ordre eos, rétro-conséquence, action : 1/3 chacun
    # (chaque groupe vaut 1.0 en poids total) ; padding à 0
    tail_w = weight[layout["ctx_len"] - 1:]
    tail_v = valid[layout["ctx_len"] - 1:]
    eos_w = weight[layout["eos_pos"] - 1:layout["eos_pos"]]
    traj_w = weight[layout["eos_pos"]:layout["action_pos"] - 1]
    dec_w = weight[layout["action_pos"] - 1:]
    assert torch.isclose(eos_w.sum(), torch.tensor(1.0))
    assert torch.isclose(traj_w.sum(), torch.tensor(1.0))
    assert torch.isclose(dec_w.sum(), torch.tensor(1.0))
    assert (tail_w[~tail_v] == 0.0).all()
    assert (tail_w[tail_v] > 0).all()
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
    # conséquence rétro-causale : unités de 16 tokens [s_k inversée (13), a_{k-1} (3)]
    for j, k in enumerate(range(SEQ_STEPS, N_CTX, -1)):
        base = layout["conseq_start"] + j * SALVE_TOKENS
        if k == SEQ_STEPS:
            # token de dénouement en premier (remplace le token 12 de s_{seq_steps})
            assert seq[base][SIG_OFFSET + 6:] == [0.0] * (N_SIGNAL - 6)
            assert seq[base + 1:base + STATE_TOKENS] == window[k][:STATE_TOKENS - 1][::-1]
        else:
            assert seq[base:base + STATE_TOKENS] == window[k][:STATE_TOKENS][::-1]
        if j < SEQ_STEPS - N_CTX - 1:
            assert seq[base + STATE_TOKENS:base + SALVE_TOKENS] == window[k - 1][STATE_TOKENS:]
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
    # la conséquence rétro-causale est réinversée (états + actions a_k)
    for k in range(N_CTX + 1, SEQ_STEPS):
        assert back[k] == window[k]
    # dernière salve : token 12 remplacé par le dénouement (intero remis à 0),
    # a_{seq_steps} non enregistrée, à zéro
    for t in range(STATE_TOKENS):
        if t == 12:
            assert back[SEQ_STEPS][t][SIG_OFFSET:] == [0.0] * N_SIGNAL
        else:
            assert back[SEQ_STEPS][t] == window[SEQ_STEPS][t]
    assert back[SEQ_STEPS][STATE_TOKENS:] == [[0.0] * DENSE_DIM] * ACTION_TOKENS


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
    # slots intero des salves état masqués (contexte + conséquence rétro-causale,
    # offset 0 de chaque unité = token 12 inversé) — sauf eos_pos (dénouement)
    intero_pos = [k * SALVE_TOKENS + 12 for k in range(N_CTX + 1)]
    intero_pos += [brain.conseq_start + j * SALVE_TOKENS for j in range(1, brain.n_imagine)]
    assert (vals[:, intero_pos, SIG_OFFSET:] == 0.0).all()
    # le token de dénouement reste visible en entrée
    assert (vals[:, brain.eos_pos, SIG_OFFSET:SIG_OFFSET + 6] != 0.0).any()


def test_wm_single_mask_is_causal(brain):
    from lambda_barre.models import build_salve_mask
    m = build_salve_mask(brain.seq_len, brain.device)
    assert m.dtype == torch.bool
    assert torch.equal(brain.mask, m)
    # causalité : au-dessus de la diagonale tout est masqué, rien en dessous
    upper = torch.triu(torch.ones_like(m, dtype=torch.bool), diagonal=1)
    assert m[upper].all()
    assert not m[~upper].any()


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
    eos_elu, p = brain.sample_denouements(ctx, temperature=0.5, top_n=4)
    assert eos_elu.shape == (4, 6) and p.shape == (4,)
    assert ((eos_elu >= 0.0) & (eos_elu <= 1.0)).all()
    assert ((p > 0.0) & (p <= 1.0)).all()
    # top_n=1 -> l'élu est le mode le plus probable (= predict_denouement),
    # et p_elu = probabilité exacte de ce mode (vaut 1.0 à température -> 0)
    mu_best = brain.predict_denouement(ctx)
    eos1, p1 = brain.sample_denouements(ctx, temperature=1e-3, top_n=1)
    assert torch.allclose(eos1, mu_best, atol=1e-5)
    assert torch.allclose(p1, torch.ones(4, device=brain.device), atol=1e-4)


def test_sample_denouements_observed_wins_only_if_better_relief(brain, rng):
    from lambda_barre.brain import relief_score
    seqs = [build_sequence(_rand_window(rng), n_ctx=N_CTX) for _ in range(2)]
    vals = brain._batch_tensors(seqs)
    ctx = vals[:, :brain.ctx_len]
    # observé au relief maximal (1.0 partout -> relief = 1.0) : gagne
    eos_obs = torch.ones(2, 6, device=brain.device)
    eos_elu, p = brain.sample_denouements(ctx, eos_observed=eos_obs)
    assert torch.allclose(eos_elu, eos_obs)
    # observé au relief minimal (0.0 partout -> relief = -1) : perd contre l'élu
    eos_elu2, p2 = brain.sample_denouements(ctx, eos_observed=torch.zeros(2, 6, device=brain.device))
    assert not torch.allclose(eos_elu2, torch.zeros(2, 6, device=brain.device))
    assert (p2 < 1.0).all()


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


def test_kv_cache_matches_full_forward_with_positions(brain, rng):
    # le KV-cache (positions explicites + infusion EOS >= eos_pos) doit être
    # identique au forward complet, y compris à travers la frontière eos_pos
    from lambda_barre.models import KVCache, build_salve_mask
    seq = build_sequence(_rand_window(rng), n_ctx=N_CTX)
    vals = brain._batch_tensors([seq])
    eos_cond = vals[:, brain.eos_pos, SIG_OFFSET:SIG_OFFSET + 6]
    L = brain.eos_pos + 20
    x = vals[:, :L]
    full = brain.world(x, attn_mask=build_salve_mask(L, brain.device), is_causal=True,
                       salve_positions=brain.salve_positions[:L],
                       eos_cond=eos_cond, eos_pos=brain.eos_pos)
    cache = KVCache(len(brain.world.transformer.layers), brain.world.d_model, 1,
                    brain.device)
    split = brain.eos_pos + 1
    out1 = brain.world.cache_forward(x[:, :split], cache,
                                     positions=brain.salve_positions[:split],
                                     eos_cond=eos_cond, eos_pos=brain.eos_pos)
    out2 = brain.world.cache_forward(x[:, split:], cache,
                                     positions=brain.salve_positions[split:L],
                                     eos_cond=eos_cond, eos_pos=brain.eos_pos)
    cat = torch.cat([out1, out2], dim=1)
    assert torch.allclose(cat, full, atol=1e-5)


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


def test_train_wm_phase_fixed_seq_yields_when_source_empty(brain, rng):
    # source vide mais fixed_seq fournie : la phase doit yield-er par batch
    # (sinon next() bloque indéfiniment — bug de la touche W du wm-manager)
    seq = build_sequence(_rand_window(rng), n_ctx=N_CTX)
    gen = brain.train_wm_phase(3, 8, target_loss=None, source="coreset",
                               fixed_seq=lambda: seq)
    out = [next(gen) for _ in range(3)]
    assert len(out) == 3
    assert all(lbl == "wm" for lbl, _, _ in out)
    gen.close()


def test_train_wm_phase_empty_source_no_fixed_returns(brain):
    gen = brain.train_wm_phase(3, 8, source="coreset")
    assert list(gen) == []


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


# --- lookahead (saut de la dernière salve de la fenêtre) -------------------------
def test_lookahead_positions():
    LA = 8
    layout = compute_seq_layout(SEQ_STEPS, N_CTX)
    sp = compute_salve_positions(SEQ_STEPS, N_CTX, lookahead=LA)
    assert sp[layout["eos_pos"]] == N_CTX + LA + 1                    # PE dédié du dénouement
    assert sp[layout["conseq_start"] + 1] == N_CTX + LA               # états de la dernière salve (saut)
    assert sp[layout["conseq_start"] + STATE_TOKENS] == SEQ_STEPS - 1  # son action garde sa vraie position
    assert sp[layout["conseq_start"] + SALVE_TOKENS + 1] == SEQ_STEPS - 1  # unité 1 = s_{seq_steps-1}
    # défaut = pas de saut (comportement historique)
    sp0 = compute_salve_positions(SEQ_STEPS, N_CTX)
    assert sp0[layout["eos_pos"]] == SEQ_STEPS + 1
    assert sp0[layout["conseq_start"] + 1] == SEQ_STEPS


def test_wm_reproduces_learned_sequences(rng):
    # Pas de test sur la loss : on entraîne jusqu'à convergence sur n séquences
    # quasi-statiques, puis on vérifie que le WM reproduit EXACTEMENT les tokens
    # appris (dénouement + conséquence + décision) — si la loss est mal calculée
    # ou décalée, ce test le voit.
    def static_window():
        base = [rng.random() for _ in range(53)]
        base_a = [rng.random() for _ in range(5)]
        return [make_salve([v + rng.uniform(-0.01, 0.01) for v in base],
                           [v + rng.uniform(-0.01, 0.01) for v in base_a])
                for _ in range(SEQ_STEPS + 1)]

    torch.manual_seed(0)
    b = Brain(seq_steps=SEQ_STEPS, n_ctx=N_CTX, d_model=128, nhead=8, layers=1,
              dim_ff=256, pol_hidden=64, lr_wm=1e-3)
    seqs = [build_sequence(static_window(), n_ctx=N_CTX) for _ in range(3)]
    layout = compute_seq_layout(SEQ_STEPS, N_CTX)
    b.world.train()
    for _ in range(10000):
        b._train_wm_batch(seqs)
    b.world.eval()
    with torch.no_grad():
        vals = b._batch_tensors(seqs)
        eos_cond = vals[:, b.eos_pos, SIG_OFFSET:SIG_OFFSET + 6]
        pred = b.world(vals, attn_mask=b.mask, is_causal=True,
                       salve_positions=b.salve_positions,
                       eos_cond=eos_cond, eos_pos=b.eos_pos)
    p = slice(layout["ctx_len"] - 1, layout["seq_len"] - 1)   # partie apprise
    err = (pred[:, p, :] - vals[:, 1:, SIG_OFFSET:][:, p, :]).abs()
    valid = b.target_valid[p].unsqueeze(0).expand_as(err)
    assert err[valid].max().item() < 0.1


def test_lookahead_window_extraction(rng):
    LA = 8
    buf = ExperienceBuffer(seq_len=SEQ_STEPS + 1, n_ctx=N_CTX, lookahead=LA,
                           capacity=50, pool_capacity=50, seed=0)
    seg = []
    for i in range(SEQ_STEPS + LA + 5):
        s = _rand_salve(rng)
        s[0][SIG_OFFSET] = float(i)          # marqueur de salve
        seg.append(s)
    buf._segments.append(seg)
    assert buf.extract_addendum() > 0
    layout = compute_seq_layout(SEQ_STEPS, N_CTX)
    cs = layout["conseq_start"]
    seq0 = buf._addendum[0]
    # unité 0 = dernière salve de la fenêtre = s_{n_ctx+lookahead} (saut +lookahead)
    assert seq0[cs + STATE_TOKENS - 1][SIG_OFFSET] == float(N_CTX + LA)
    # unité 1 = s_{seq_steps-1} (partie consécutive)
    assert seq0[cs + SALVE_TOKENS + STATE_TOKENS - 1][SIG_OFFSET] == float(SEQ_STEPS - 1)


def test_confort_encoding_and_eos_sign(brain):
    from lambda_barre.tokenize import (_BY_KEY, _REWARD_IDX, _denormalize,
                                       _normalize, salve_cost)
    spec = _BY_KEY["confort"]
    assert spec.first_val == 0.0 and spec.last_val == -1.0
    assert abs(_normalize(0.0, spec) - 0.0) < 1e-6
    assert abs(_normalize(-0.3, spec) - 0.3) < 1e-6
    assert abs(_normalize(-1.0, spec) - 1.0) < 1e-6
    assert abs(_denormalize(0.3, spec) - (-0.3)) < 1e-6

    # salve_cost == _salve_cost_batch avec confort actif
    s_comfy = make_salve([0.0] * 53, [0.5] * 5)
    s_comfy[_REWARD_IDX][SIG_OFFSET] = 0.3   # confort = -0.3
    s_flat = make_salve([0.0] * 53, [0.5] * 5)
    t_comfy = torch.tensor([s_comfy], dtype=torch.float32, device=brain.device)
    assert abs(salve_cost(s_comfy) - (-0.3)) < 1e-5
    assert abs(float(brain._salve_cost_batch(t_comfy)[0]) - salve_cost(s_comfy)) < 1e-5

    # relèvement (0.0 -> -0.3) donne slot EOS confort > 0.5 ; perte (-0.3 -> 0.0) donne < 0.5
    win_gain = [s_flat] * (N_CTX + 1) + [s_comfy] * (SEQ_STEPS - N_CTX)
    win_loss = [s_comfy] * (N_CTX + 1) + [s_flat] * (SEQ_STEPS - N_CTX)
    seq_gain = build_sequence(win_gain, n_ctx=N_CTX)
    seq_loss = build_sequence(win_loss, n_ctx=N_CTX)
    eos_pos = compute_seq_layout(SEQ_STEPS, N_CTX)["eos_pos"]
    assert seq_gain[eos_pos][SIG_OFFSET + 5] > 0.5
    assert seq_loss[eos_pos][SIG_OFFSET + 5] < 0.5


def test_relief_score_tanh_symmetry_and_dominance():
    neutral = torch.tensor([0.5] * 6)
    assert abs(float(relief_score(neutral))) < 1e-6
    pain_gain = torch.tensor([0.5, 1.0, 0.5, 0.5, 0.5, 0.5])
    pain_loss = torch.tensor([0.5, 0.0, 0.5, 0.5, 0.5, 0.5])
    assert abs(float(relief_score(pain_gain)) + float(relief_score(pain_loss))) < 1e-6
    assert float(relief_score(pain_gain)) > 0.7
    # douleur max + petit gain d'effort reste fortement négatif
    mixed = torch.tensor([0.75, 0.0, 0.5, 0.5, 0.5, 0.5])
    assert float(relief_score(mixed)) < -0.5
    # 5 coûts au pire + confort élevé reste très négatif
    catastrophic = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.95])
    assert float(relief_score(catastrophic)) < -0.9


def test_mdn_learns_multimodal_75_25_and_exact_p(rng):
    torch.manual_seed(0)
    b = Brain(seq_steps=SEQ_STEPS, n_ctx=N_CTX, d_model=64, nhead=4, layers=1,
              dim_ff=128, pol_hidden=32, lr_wm=3e-3, wm_input_noise=0.0,
              eos_n_modes=4, eos_nll_w=1.0)
    base_win = _rand_window(rng)
    eos_pos = b.eos_pos
    seq_neutral = build_sequence(base_win, n_ctx=N_CTX)
    seq_fall = [list(t) for t in seq_neutral]
    # même contexte exact, deux dénouements distincts : 75% neutre / 25% chute
    eos_a = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    eos_b = [0.2, 0.05, 0.5, 0.1, 0.1, 0.2]
    seq_neutral[eos_pos] = seq_neutral[eos_pos][:SIG_OFFSET] + eos_a + [0.0] * (N_SIGNAL - 6)
    seq_fall[eos_pos] = seq_fall[eos_pos][:SIG_OFFSET] + eos_b + [0.0] * (N_SIGNAL - 6)
    batch = [seq_neutral] * 15 + [seq_fall] * 5
    b.world.train()
    for _ in range(300):
        b._train_wm_batch(batch)
    ctx = b._batch_tensors([seq_neutral])[:, :b.ctx_len]
    mu_best = b.predict_denouement(ctx)[0]
    eos_elu, p_elu = b.sample_denouements(ctx, top_p=0.95, min_p=0.05)
    # le plus probable est le neutre (~0.5), et sa proba apprise est ~0.75 (pas 1/K !)
    assert (mu_best - torch.tensor(eos_a, device=b.device)).abs().max().item() < 0.1
    assert (eos_elu[0] - torch.tensor(eos_a, device=b.device)).abs().max().item() < 0.1
    assert 0.55 < float(p_elu[0]) < 0.90


def test_inference_preserves_training_flag_with_dropout(rng):
    b = Brain(seq_steps=SEQ_STEPS, n_ctx=N_CTX, dropout=0.1)
    seq = build_sequence(_rand_window(rng), n_ctx=N_CTX)
    ctx = b._batch_tensors([seq])[:, :b.ctx_len]
    b.world.train()
    mu = b.predict_denouement(ctx)
    assert b.world.training
    eos_elu, _ = b.sample_denouements(ctx)
    assert b.world.training
    gen = b.generate_from_denouement(ctx, eos_elu)
    assert gen["full"].shape == (1, b.seq_len, DENSE_DIM)
    assert b.world.training


def test_experience_buffer_window_span_with_lookahead(rng):
    # seq_steps=12, n_ctx=6, lookahead=10 -> window_span=17 (pas seq_len=13)
    buf = ExperienceBuffer(seq_len=13, n_ctx=6, lookahead=10,
                           capacity=50, pool_capacity=50, seed=0)
    assert buf.window_span == 17
    for _ in range(15):
        buf.push(_rand_salve(rng))
    assert buf.journal_len == 0
    assert buf.sample(4) == []
    buf.boundary()
    assert len(buf._segments) == 0    # segment de 15 < 17 non scellé
    for _ in range(17):
        buf.push(_rand_salve(rng))
    assert buf.journal_len == 1
    assert len(buf.sample(4)) == 1
    assert buf.extract_addendum() == 1


def test_dream_record_uses_generated_consequence(brain, rng):
    for _ in range(SEQ_STEPS + 5):
        brain.record_tick(_rand_salve(rng))
    brain.buffer.extract_addendum()
    list(brain.train_policy(steps=1, batch=2))
    rec = brain.last_dream_record
    assert rec is not None
    # les états imaginés dans predicted_steps proviennent de generate_from_denouement,
    # pas de la copie directe des vrais états vécus
    imagined_s = rec.predicted_steps[brain.n_ctx].state_tokens
    real_s = rec.lived_steps[brain.n_ctx + 1].state_tokens
    assert imagined_s != real_s


def test_train_wm_phase_waits_out_adam_slingshot(brain, rng, monkeypatch):
    seq = build_sequence(_rand_window(rng), n_ctx=N_CTX)
    brain.buffer._coreset.append(seq)
    brain.buffer._coreset_meta.append(ExperienceBuffer._default_meta("coreset"))
    # Simule une loss qui descend, puis un slingshot à l'epoch 4 (juste avant
    # quota=5) qui ne retombe et ne se stabilise qu'aux epochs 7, 8, 9.
    losses = iter([0.10, 0.05, 0.02, 0.08, 0.05, 0.03, 0.02, 0.018, 0.017, 0.016])
    monkeypatch.setattr(brain, "_train_wm_batch", lambda batch: next(losses))
    epochs_seen = [ep for _, ep, _ in brain.train_wm_phase(
        epochs=5, batch=8, target_loss=0.06, source="coreset",
        slingshot_ratio=1.30, slingshot_settle=3, slingshot_max_extra=20)]
    # Ne s'arrête PAS à l'epoch 5 (0.05 <= target 0.06 mais en plein slingshot),
    # attend la retombée (ep 7) + 3 epochs calmes (ep 7, 8, 9).
    assert epochs_seen == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert not brain.sleep_wm_slingshot


def test_input_noise_mask_preserves_prefix_padding_intero_and_eos(brain):
    from lambda_barre.tokenize import _INTERO_IDX, _N_SIGNALS
    m = brain.input_noise_mask[0]                       # [seq_len, DENSE_DIM]
    # préfixe structurel intact partout
    assert (m[:, :SIG_OFFSET] == 0.0).all()
    # token dénouement intact
    assert (m[brain.eos_pos] == 0.0).all()
    # tokens intero masqués (contexte + conséquence) intacts
    intero_pos = [k * SALVE_TOKENS + _INTERO_IDX for k in range(brain.n_ctx + 1)]
    intero_pos += [brain.conseq_start + j * SALVE_TOKENS for j in range(1, brain.n_imagine)]
    assert (m[intero_pos] == 0.0).all()
    # padding intact sur REWARD (1 slot valide) et ACT_TAIL (1 slot valide)
    assert m[11, SIG_OFFSET].item() == 1.0
    assert (m[11, SIG_OFFSET + 1:] == 0.0).all()
    assert m[brain.action_pos + 2, SIG_OFFSET].item() == 1.0
    assert (m[brain.action_pos + 2, SIG_OFFSET + 1:] == 0.0).all()
    # VISION (16 slots valides) entièrement bruité sur ses 16 slots signaux
    assert (m[4, SIG_OFFSET:SIG_OFFSET + _N_SIGNALS[4]] == 1.0).all()


def test_eos_cond_infusion_zero_leak_on_context(brain, rng):
    seq = build_sequence(_rand_window(rng), n_ctx=N_CTX)
    vals = brain._batch_tensors([seq])
    eos_lo = torch.zeros(1, 6, device=brain.device)
    eos_hi = torch.ones(1, 6, device=brain.device)
    brain.world.eval()
    with torch.no_grad():
        p_lo, mdn_lo = brain.world.forward_with_eos(
            vals, attn_mask=brain.mask, is_causal=True,
            salve_positions=brain.salve_positions, eos_row=brain.eos_pos - 1,
            eos_cond=eos_lo, eos_pos=brain.eos_pos)
        p_hi, mdn_hi = brain.world.forward_with_eos(
            vals, attn_mask=brain.mask, is_causal=True,
            salve_positions=brain.salve_positions, eos_row=brain.eos_pos - 1,
            eos_cond=eos_hi, eos_pos=brain.eos_pos)
    # Zéro fuite causale sur le contexte (0..eos_pos-1) et sur les sorties MDN !
    assert torch.equal(p_lo[:, :brain.eos_pos], p_hi[:, :brain.eos_pos])
    for t_lo, t_hi in zip(mdn_lo, mdn_hi):
        assert torch.equal(t_lo, t_hi)
    # Effet direct immédiat sur la conséquence et l'action (q >= eos_pos)
    assert (p_lo[:, brain.eos_pos:] - p_hi[:, brain.eos_pos:]).abs().max().item() > 1e-3


def test_dedup_includes_consequence_and_drops_true_duplicates(rng):
    buf = ExperienceBuffer(seq_len=SEQ_STEPS + 1, n_ctx=N_CTX,
                           dedup_eps=0.02, capacity=50, pool_capacity=50, seed=0)
    layout = compute_seq_layout(SEQ_STEPS, N_CTX)
    base_win = _rand_window(rng)
    seq_a = build_sequence(base_win, n_ctx=N_CTX)
    # Quasi-doublon complet (perturbation minuscule 1e-3 partout)
    seq_dup = [list(tok) for tok in seq_a]
    seq_dup[0][SIG_OFFSET] = min(1.0, max(0.0, seq_dup[0][SIG_OFFSET] + 1e-3))
    # Même contexte + même EOS + même action de décision, MAIS conséquence différente !
    seq_diff_conseq = [list(tok) for tok in seq_a]
    for q in range(layout["eos_pos"] + 1, layout["action_pos"]):
        for k in range(SIG_OFFSET, DENSE_DIM):
            if seq_diff_conseq[q][k] != 0.0:
                seq_diff_conseq[q][k] = 1.0 - seq_diff_conseq[q][k]
    buf._addendum = [seq_a, seq_dup, seq_diff_conseq]
    buf._addendum_meta = [ExperienceBuffer._default_meta("wake") for _ in range(3)]

    d_intra = buf.deduplicate_addendum()
    assert d_intra == 1
    assert len(buf._addendum) == 2                   # seq_a et seq_diff_conseq gardées
    assert buf._addendum_meta[0]["count"] == 2

    # Consolidation puis arrivée d'un doublon du coreset -> rafraîchit la vivacité
    buf.commit_addendum()
    buf._coreset_vivacity[0] = 0.2
    buf._coreset_meta[0]["vivacity"] = 0.2
    buf._addendum = [[list(tok) for tok in seq_dup]]
    buf._addendum_meta = [ExperienceBuffer._default_meta("wake")]
    d_core = buf.deduplicate_against_coreset()
    assert d_core == 1
    assert len(buf._addendum) == 0
    assert buf._coreset_vivacity[0] == 1.0
    assert buf._coreset_meta[0]["count"] == 3


def test_density_eviction_protects_rare_old_memories(rng):
    buf = ExperienceBuffer(seq_len=SEQ_STEPS + 1, n_ctx=N_CTX,
                           dedup_eps=0.0, vivacity_decay=0.95,
                           capacity=50, pool_capacity=3, seed=0)
    # 1 souvenir ancien et rare (loin de tout, vivacité très faible)
    win_rare = [make_salve([0.9] * 53, [0.9] * 5) for _ in range(SEQ_STEPS + 1)]
    seq_rare = build_sequence(win_rare, n_ctx=N_CTX)
    # 4 souvenirs très proches dans un cluster dense
    win_dense = [make_salve([0.1] * 53, [0.1] * 5) for _ in range(SEQ_STEPS + 1)]
    seq_dense = build_sequence(win_dense, n_ctx=N_CTX)
    cluster = []
    for k in range(4):
        s = [list(tok) for tok in seq_dense]
        s[0][SIG_OFFSET] = 0.1 + 0.002 * k
        cluster.append(s)

    buf._coreset = [seq_rare] + cluster[:2]
    buf._coreset_vivacity = [0.05, 0.5, 0.8]
    buf._coreset_meta = [ExperienceBuffer._default_meta("coreset") for _ in range(3)]
    buf._addendum = cluster[2:]
    buf._addendum_meta = [ExperienceBuffer._default_meta("wake") for _ in range(2)]

    buf.commit_addendum()
    assert buf.coreset_size == 3
    assert buf.last_evicted == 2
    # Le souvenir rare ancien (malgré vivacité 0.05) n'est PAS évincé !
    assert seq_rare in buf._coreset


def test_addendum_phase_uses_mixed_coreset_replay(brain, rng, monkeypatch):
    core_seqs = [build_sequence(_rand_window(rng), n_ctx=N_CTX) for _ in range(6)]
    add_seqs = [build_sequence(_rand_window(rng), n_ctx=N_CTX) for _ in range(4)]
    brain.buffer._coreset = core_seqs
    brain.buffer._coreset_vivacity = [1.0] * len(core_seqs)
    brain.buffer._coreset_meta = [ExperienceBuffer._default_meta("coreset") for _ in core_seqs]
    brain.buffer._addendum = add_seqs
    brain.buffer._addendum_meta = [ExperienceBuffer._default_meta("wake") for _ in add_seqs]

    seen_batches = []
    def spy_train(batch_seqs):
        seen_batches.append(batch_seqs)
        return 0.01

    monkeypatch.setattr(brain, "_train_wm_batch", spy_train)
    list(brain.train_wm_phase(epochs=1, batch=4, target_loss=None, source="addendum"))
    # batch=4 -> step_batch=2 addendum + 2 coreset = 4 (mix 50/50), 2 batchs par epoch
    assert len(seen_batches) == 2
    core_ids = {id(s) for s in core_seqs}
    add_ids = {id(s) for s in add_seqs}
    for b in seen_batches:
        assert len(b) == 4
        assert sum(id(s) in add_ids for s in b) == 2
        assert sum(id(s) in core_ids for s in b) == 2
