# λ̄ Design — Refonte WM à dénouement (branche `refonte`)

Référence de conception de la refonte (ADR-0002, spec REFONTE.md). L'ancien
pipeline à deux passes causale/lookahead est archivé sous le tag `LEGACY-AI`.

## Séquence enregistrée

Une séquence est une **liste plate de `seq_len` tokens** ordonnés :

```
contexte | dénouement | conséquence inversée | action de décision
```

- **contexte** : `[s0,a0, ..., a_{n_ctx-1}, s_{n_ctx}]` — `ctx_len` tokens ;
- **dénouement** : 1 token EOS à `eos_pos = ctx_len`, ground truth calculé à
  l'extraction (`_eos_slots_of_window`) : progrès par canal
  `f = P − poids_chemin·A` sur les 6 canaux de coût, 0.5 = neutre ;
- **conséquence inversée** : états seuls `s_{seq_steps} .. s_{n_ctx+1}`, du plus
  tardif au plus précoce — `conseq_len = n_imagine × 13` tokens ;
- **action de décision** : `a_{n_ctx}`, 3 tokens, **dernier token** de la
  séquence (`action_pos`).

Géométrie : `compute_seq_layout` ; positions temporelles (PE sinusoïdal keyed by
salve index) : `compute_salve_positions` (conséquence inversée en positions
décroissantes). Le journal de veille reste un flux de salves temporelles ;
`build_sequence` réordonne une fenêtre de `seq_steps+1` salves à l'extraction
(`ExperienceBuffer.extract_addendum`).

## Entraînement du WM (`Brain._train_wm_batch`)

Un seul masque (strictement causal sur `seq_len`), un seul forward par batch.
La loss est une MSE masquée (`build_target_tables`) : **poids 0 sur le
contexte** (prompt, non appris), **poids 1 sur tous les tokens à partir du
dénouement**. Dans le même forward, le dénouement est prédit depuis le
contexte, et la conséquence inversée puis l'action de décision sont prédites
conditionnellement au dénouement. Aucun dénouement contrefactuel injecté.
Toutes les sorties sont des régressions continues [0,1] (MSE), y compris le
dénouement (vecteur 6 slots).

`train_wm_phase` : vraies epochs (dataset balayé en batchs sans omission ni
remise), `learn_count` +1 par epoch, `surprise` = loss du batch. L'optimisation
du dataset (dedup, filtrage surprise, oubli actif, tri plateau, pondération
d'impact) est **mise de côté** pour l'instant.

## Inférence (`Brain.sample_denouements` / `generate_from_denouement`)

1. `predict_denouement(ctx)` : moyenne conditionnelle µ (6 slots) lue à la
   dernière ligne du forward sur le contexte.
2. Échantillonnage de K vecteurs `v ~ N(µ, T²)` (température `eos_temperature`),
   top-p sur les tirages ; chaque tirage noté
   `f = log p(v) + eos_w_fav · relief(v)` ; **le meilleur est élu** (pas
   nécessairement le plus probable).
3. Règle de l'observé : si `relief(eos_observé) >= eos_satisfying`, on reste sur
   l'observé (p = 1) ; sinon on élit parmi les échantillonnés (extrapolation
   cross-contexte).
4. `generate_from_denouement(ctx, eos_elu)` : génération auto-régressive
   depuis `[contexte, dénouement élu]` (masque causal, positions explicites),
   déroule la conséquence inversée jusqu'à l'action de décision (dernier token).

## Policy (`Brain.train_policy`)

- Latent d'entrée = couche médiane du WM sur la **partie contexte** seule
  (même latent qu'en éveil).
- Cible = **action de décision générée** par le WM sous le dénouement élu.
- Pondération linéaire par la probabilité du dénouement élu :
  `w = p`, `loss = Σ_i mse_i · p_i.detach() / Σ_i p_i.detach()`.

## Sleep cycle (`Brain.sleep_cycle`)

Générateur consommé frame par frame par l'UI :

1. `extract_addendum` — fenêtres du journal → séquences enregistrées dans
   l'Addendum (yield `wm_*` ensuite) ;
2. phase WM Coreset (`yield "wm_coreset", loss`), puis phase WM Addendum
   (`yield "wm_addendum", loss`) ;
3. policy (`yield "pol", loss`), record Dream Theater minimal ;
4. `commit_addendum` — Addendum → Coreset (éviction FIFO de capacité) ;
5. `yield "done", stats`.

`wm_loss_ratio = loss_après / loss_avant` ; `wm_plateau = ratio > 0.95` (HUD).

## Checkpoints

`wm_ckpt.pt` / `pol_ckpt.pt` / `buf_ckpt.pt`. Le changement de layout de
séquence invalide les buffers legacy : `ExperienceBuffer.load` les rejette ;
lancer avec `--reset`.
