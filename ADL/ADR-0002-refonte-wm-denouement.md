# ADR-0002 — Refonte du World Model à dénouement (entraînement rétrotemporel)

- Statut : adopté
- Date : 2026-11-13
- Remplace : le pipeline à deux passes causale/lookahead (décrit dans les
  sessions ≤ 2026-11-13 et DESIGN.md legacy) — l'ancien état est tagué
  `LEGACY-AI`.

## Contexte

L'ancien WM apprenait deux choses en parallèle : la dynamique causale P(s(t+1)|s(t),a(t)) et l'inversion lookahead P(a4 | contexte, EOS). Le masque lookahead était un deuxième masque, la passe lookahead un deuxième forward, et le dénouement (EOS) était un token terminal calculé au batch time. Le diagnostic montrait que la loss causale descendait sans refléter la logique du monde.

## Décision

### Séquence enregistrée

Une séquence enregistrée est une **liste plate de `seq_len` tokens** dans
l'ordre : `contexte | conséquence rétro-causale | action de décision`.

- **contexte** : `[s_0,a_0, ..., a_{n_ctx-1}, s_{n_ctx}]` (`ctx_len` tokens,
  ordre normal ; `s_{n_ctx}` seul, sans action) ;
- **conséquence rétro-causale** (à partir de `conseq_start = eos_pos =
  ctx_len`) : unités de 16 tokens pour `k = seq_steps .. n_ctx+1`, du plus
  tardif au plus précoce — `s_k` avec ses 13 tokens d'état **inversés**
  (token 12 en premier), puis `a_{k-1}` (3 tokens, ordre interne normal).
  En remontant, prédire `s_k` mobilise `a_k` et `s_{k+1}` : d'où
  l'entrelacement `s26, a25, s25, a24, ..., s5, a4` ;
- **token de dénouement** : le token 12 de la première unité (`s_{seq_steps}`),
  à `eos_pos = ctx_len` — **premier token après le contexte**, 6 slots,
  valeur observée réelle (ground truth) calculée à l'extraction depuis la
  fenêtre temporelle (le journal de veille reste un flux de salves
  temporelles). C'est le token de dénouement qui est prédit juste après le
  contexte à l'inférence et qu'on échantillonne pour élire un dénouement ;
- **action de décision** : `a_{n_ctx}` (3 derniers tokens de la séquence,
  `action_pos`).

Le dénouement est donc porté par une salve existante (la dernière salve
temporelle = première unité de la conséquence rétro-causale), comme avant la
refonte ; il n'y a pas de token isolé en dehors d'une salve. `a_{seq_steps}`
n'est pas enregistrée.

Le dénouement reste un **vecteur continu** de 6 slots [0,1] (progrès par canal
`f = P − poids_chemin·A`) : pas de classes, pas de quantification. Les sorties
du WM sont des régressions continues partout.

### Entraînement du WM

Un seul masque (strictement causal sur `seq_len`), un seul forward par batch.
La loss (MSE masquée) porte sur **tous les tokens à partir du dénouement** ;
le contexte sert de prompt (poids 0), comme un transformer de traduction.
Dans le même forward, le dénouement est prédit depuis le contexte, et la
conséquence inversée puis l'action de décision sont prédites conditionnellement
au dénouement (teacher-forcing causal + positions explicites). Aucun
dénouement contrefactuel n'est injecté à l'entraînement.

### Inférence

1. Depuis le contexte, le WM prédit la moyenne conditionnelle µ du dénouement
   (vecteur 6 slots).
2. Échantillonnage de K vecteurs `v ~ N(µ, T²)` (température T), top-K/top-p
   sur les tirages (densité gaussienne) — pas d'argmax (qui donnerait la
   moyenne conditionnelle).
3. Chaque tirage est noté `f(probable, favorable) = log p(v) + w_fav ·
   relief(v)` ; le meilleur est élu, pas nécessairement le plus probable.
4. Génération auto-régressive depuis `[contexte, dénouement élu]` : la
   conséquence inversée est déroulée jusqu'à l'action de décision.
5. Règle de l'observé : si un dénouement observé depuis ce contexte atteint le
   seuil `eos_satisfying`, on reste sur l'observé (p = 1) ; l'élection
   d'un dénouement meilleur que l'observé (extrapolation cross-contexte) n'est
   utilisée que sinon.

### Policy

Le latent d'entrée de la policy est produit par le WM sur la **partie
contexte** seule. La policy imite l'**action de décision générée** par le WM ;
la descente de gradient est pondérée linéairement par la probabilité du
dénouement élu : `w = p`, `loss *= p.detach()` par séquence du batch.

## Conséquences

- `brain.py` réécrit en cleanroom (branche `refonte`) : `compute_seq_layout`,
  `compute_salve_positions`, `build_target_tables`, `build_sequence`,
  `ExperienceBuffer` (journal temporel → séquences plates à l'extraction),
  `Brain` (un seul masque, `predict_denouement` / `sample_denouements` /
  `generate_from_denouement`, policy pondérée par p).
- `models.py` inchangé (le forward accepte déjà `salve_positions`).
- Checkpoints invalides après le changement de layout : lancer avec `--reset` ;
  `ExperienceBuffer.load` rejette les fichiers legacy.
- `wm_theater.py`, `wm_manager.py`, `dream_theater.py` à refaire à partir de
  leurs fonctionnalités sur le nouveau layout.
- Optimisation du dataset (métadonnées riches, dedup, filtrage surprise,
  oubli actif, tri plateau, pondération d'impact) remise à plus tard.
- Nouveaux termes canoniques (à ajouter à ADR-0001) : **conséquence inversée**,
  **action de décision**, **dénouement élu**, **élection f(probable,
  favorable)**.
