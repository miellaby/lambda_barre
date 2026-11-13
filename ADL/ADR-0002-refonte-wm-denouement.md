# ADR-0002 — Refonte du World Model à dénouement (entraînement rétrotemporel)

- Statut : adopté
- Date : 2026-11-13
- Remplace : le pipeline à deux passes causale/lookahead (décrit dans les
  sessions ≤ 2026-11-13 et DESIGN.md legacy) — l'ancien état est tagué
  `LEGACY-AI`.

## Contexte

L'ancien WM apprenait deux choses en parallèle : la dynamique causale
P(s(t+1)|s(t),a(t)) et l'inversion lookahead P(a4 | contexte, EOS). Le masque
lookahead était un deuxième masque, la passe lookahead un deuxième forward, et
le dénouement (EOS) était un token terminal calculé au batch time. Le diagnostic
de la session 2026-11-13 montrait que la loss causale descendait sans refléter
la logique du monde. La spec REFONTE.md acte un entraînement rétrotemporel
unique.

## Décision

### Séquence enregistrée

Une séquence enregistrée est une **liste plate de `seq_len` tokens** dans
l'ordre : `contexte | dénouement | conséquence inversée | action de décision`.

- **contexte** : `[s0,a0, ..., a_{n_ctx-1}, s_{n_ctx}]` (`ctx_len` tokens) ;
- **dénouement** : 1 token EOS à `eos_pos = ctx_len`, valeur observée réelle
  (ground truth), calculé à l'extraction depuis la fenêtre temporelle (le
  journal de veille reste un flux de salves temporelles) ;
- **conséquence inversée** : états post-décision seuls, du plus tardif au plus
  précoce, `s_{seq_steps} .. s_{n_ctx+1}` (`conseq_len = n_imagine × 13`) ;
- **action de décision** : `a_{n_ctx}` (3 tokens), **dernier token** de la
  séquence (`action_pos`).

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
