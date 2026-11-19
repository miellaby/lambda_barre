# λ̄ Design — Refonte WM à dénouement (branche `refonte`)

Référence de conception de la refonte (ADR-0002, spec REFONTE.md). L'ancien
pipeline à deux passes causale/lookahead est archivé sous le tag `LEGACY-AI`.

## Séquence enregistrée

Une séquence est une **liste plate de `seq_len` tokens** ordonnés :

```
contexte | dénouement | conséquence inversée | action de décision
```

La **fenêtre temporelle** extraite du journal fait `seq_steps+1` salves avec un
**saut** (`lookahead`) : `seq_steps` salves consécutives `s0..s_{seq_steps-1}`,
puis la dernière salve à `+lookahead` ticks de la décision
(`s_{n_ctx+lookahead}`) — on apprend le futur proche, le dénouement est lointain.

- **contexte** : `[s0,a0, ..., a_{n_ctx-1}, s_{n_ctx}]` — `ctx_len` tokens ;
- **dénouement** : 1 token EOS à `eos_pos = ctx_len`, ground truth calculé à
  l'extraction (`_eos_slots_of_window`) : intégrale par canal de l'écart au
  coût de la décision pondérée par la vraie position temporelle,
  `r[j] = Σ_{k>d} eos_decay^(pos_k−d)·(c[k][j]−c[d][j])·p_k` avec `pos_k = k`
  sauf dernière salve à `pos = n_ctx+lookahead`, et `p_k = lookahead` pour le
  dernier état (prolongation fictive du coût final),
  `slot[j] = 0.5 − 0.5·tanh(r[j])`, 0.5 = neutre ;
- **conséquence inversée** : états seuls `s_{seq_steps} .. s_{n_ctx+1}`, du plus
  tardif au plus précoce — `conseq_len = n_imagine × 13` tokens ;
- **action de décision** : `a_{n_ctx}`, 3 tokens, **dernier token** de la
  séquence (`action_pos`).

Géométrie : `compute_seq_layout` ; positions temporelles (PE sinusoïdal keyed by
salve index) : `compute_salve_positions(seq_steps, n_ctx, lookahead)` — la
dernière salve de la fenêtre (unité 0 de la conséquence) a la vraie position
`n_ctx + lookahead`, l'action `a_{k-1}` garde `k-1`, et le dénouement a un PE
dédié `n_ctx + lookahead + 1` (hors fenêtre). Le journal de veille reste un flux
de salves temporelles ; `build_sequence` réordonne une fenêtre de
`seq_steps+1` salves (avec saut) à l'extraction
(`ExperienceBuffer.extract_addendum` / `_window_at`).

## Entraînement du WM (`Brain._train_wm_batch`)

Un seul masque (strictement causal bool sur `seq_len`), un seul forward par
batch — sans scale multiplicatif d'entrée (`ctx_attn_boost` supprimé), remplacé
par une **infusion directe du dénouement** (`eos_cond_proj : Linear(6, d_model)`)
additionnée à l'embedding de chaque token `q >= eos_pos` (conséquence inversée
et action de décision) sans jamais toucher au contexte `0..eos_pos-1` (zéro
fuite sur le MDN et cohérence stricte avec l'éveil / KV-cache).
La loss combine :

- une MSE masquée (`build_target_tables`) : **poids 0 sur le contexte**
  (prompt, non appris), **poids 1/3 par groupe** (dénouement, trajectoire,
  action de décision), pondérée par token ;
- une tête **MDN** (*Mixture Density Network*, `eos_n_modes` = 8 composantes
  gaussiennes diagonales par mode, plancher `eos_sigma_min` = 0.02) lue à la
  row `eos_pos - 1`, entraînée par **NLL exacte** non pondérée par l'impact
  (préserve les vraies fréquences calibrées $P(\text{EOS}\mid\text{ctx})$,
  décalée par son minimum théorique en `sigma_min` et normalisée par slot, donc
  toujours $\ge 0$, poids `eos_nll_w`) — une vraie distribution **jointe
  continue et multimodale** sur les 6 slots, sans codebook ni k-means.

Dans le même forward, le dénouement est prédit depuis le contexte, et la
conséquence inversée puis l'action de décision sont prédites conditionnellement
au dénouement (teacher-forcing). Aucun dénouement contrefactuel injecté.

`train_wm_phase` : vraies epochs (dataset shufflé et balayé en batchs sans
omission ni remise — préserve la pondération intra-batch par impact et la
calibration du MDN), `learn_count` +1 par epoch, `surprise` = loss du batch, et
garde-fou anti-arrêt en plein slingshot d'Adam. En phase `wm_addendum`, lorsque
le coreset est non vide, chaque batch fait un **replay mixte 50/50** (moitié
addendum, moitié tirée du coreset) pour éviter l'oubli catastrophique avant
`train_policy`.

## Optimisation simple du dataset (`ExperienceBuffer`)

- **Déduplication géométrique complète** (`deduplicate_addendum` et
  `deduplicate_against_coreset`, seuil RMSE `dedup_eps = 0.02`) calculée sur
  **tous les slots signaux valides de la séquence complète** (contexte +
  dénouement + conséquence inversée + action de décision, hors `intero` masqué
  et padding). Deux séquences ayant une conséquence différente ne sont jamais
  fusionnées ; un vrai doublon rafraîchit `vivacity = 1.0` et incrémente `count`
  du représentant conservé.
- **Pas de filtrage destructif par surprise** ni de remise en jeu d'un tiers du
  coreset (préserve les souvenirs sains maîtrisés et les paires
  contrefactuelles).
- **Éviction à capacité par densité** (`commit_addendum`) : décroissance douce
  de la vivacité (`vivacity_decay = 0.95` par cycle), puis si `coreset >
  pool_capacity`, éviction dans les **paires les plus proches** (distance RMSE
  sur la séquence complète) en gardant le membre de plus haute priorité
  `max(|relief|, impact_floor) · (0.5 + 0.5·vivacity)` — compresse les routines
  denses tout en protégeant les comportements rares anciens.

## Inférence (`Brain.sample_denouements` / `generate_from_denouement`)

- `_denouement_dist(ctx)` : paramètres MDN `(logits_pi, mu, sigma)` lus à la
  dernière row du forward sur le contexte (`world.eval()` restauré après
  appel) ; `predict_denouement(ctx)` = centre $\mu$ du mode le **plus probable**
  (`argmax(logits_pi)`).
- Élection exacte sur les $K$ modes du MDN :
  `probs = softmax(logits_pi / eos_temperature)`, noyau **nucleus exact**
  (`(cum - sorted_p) < eos_top_p`, sans off-by-one ; `eos_top_n` optionnel) et
  **plancher `eos_min_p`** (le mode argmax reste toujours éligible) ;
  **élection du meilleur `relief_score(mu)`** dans ce noyau, avec
  `p_elu = probs[elu]`.
  `relief_score(slots) = tanh(Σ (W_j / W_max) · (2·slot_j - 1))` (additif,
  symétrique autour de `0.5 = 0`, non dilué sur un canal isolé).
- Règle de l'observé : si `eos_observed` est fourni, il remplace l'élu
  uniquement quand son relief dépasse strictement celui de l'élu (`p_elu = 1`).
- `generate_from_denouement(ctx, eos_elu)` : génération auto-régressive
  depuis `[contexte, dénouement élu]` (masque causal, KV-cache, positions
  explicites), déroule la conséquence inversée jusqu'à l'action de décision
  (dernier token).

## Policy (`Brain.train_policy`)

- Latent d'entrée = couche médiane du WM sur la **partie contexte** seule,
  normalisé par `F.layer_norm` sans paramètres (identique en éveil et en
  sommeil).
- Cible = **action de décision générée** par le WM sous le dénouement élu.
- Pondération par confiance et soulagement positif :
  `w = p_elu.detach() * relu(relief_elu)`, `loss = Σ_i mse_i · w_i / Σ_i w_i`.

## Sleep cycle (`Brain.sleep_cycle`)

Générateur consommé frame par frame par l'UI :

- `extract_addendum` + déduplication géométrique intra-addendum et contre le
  coreset (`yield "prune", ...`) ;
- phase WM Coreset (`yield "wm_coreset", loss`), puis phase WM Addendum en
  replay mixte 50/50 avec le Coreset (`yield "wm_addendum", loss`) ;
- policy (`yield "pol", loss`), record Dream Theater avec conséquence imaginée ;
- `commit_addendum` — Addendum → Coreset (décroissance vivacité + éviction par
  densité si capacité dépassée) ;
- `yield "done", stats`.

`wm_loss_ratio = loss_après / loss_avant` ; `wm_plateau = ratio > 0.95` (HUD).

## Checkpoints

`wm_ckpt.pt` / `pol_ckpt.pt` / `buf_ckpt.pt`. Le changement de layout de
séquence invalide les buffers legacy : `ExperienceBuffer.load` les rejette ;
lancer avec `--reset`.
