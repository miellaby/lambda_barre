# TODO — Refonte du World Model (REFONTE.md)

Spec source : REFONTE.md. Cleanroom du noyau (brain.py) lancé sur la branche
`refonte` ; l'ancien état est tagué LEGACY-AI. La doc legacy (README.md,
DESIGN.md, sessions, ADL) décrit l'ancien fonctionnement à deux passes
causale/lookahead : à relire avec recul.

## Fait (cleanroom v1)

- [x] Séquence enregistrée = liste plate réordonnée `contexte | dénouement | conséquence inversée | action de décision` (`build_sequence`, calcul du dénouement à l'extraction depuis le journal temporel)
- [x] Un seul masque (strictement causal), un seul forward/loss par batch : poids 0 sur le contexte (prompt), 1.0 sur tous les tokens à partir du dénouement
- [x] Même forward : dénouement prédit depuis le contexte ; conséquence inversée + action de décision conditionnées au dénouement (teacher-forcing causal, positions explicites)
- [x] Sorties continues [0,1] en MSE partout ; le dénouement reste un vecteur continu de 6 slots (pas de classes)
- [x] Inférence : échantillonnage gaussien autour de la moyenne prédite (température T, top-K/top-p), élection `argmax(log p + w_fav·relief)`, génération auto-régressive depuis `[contexte, dénouement élu]` jusqu'à l'action de décision
- [x] Règle de l'observé : si le dénouement observé atteint `eos_satisfying`, on reste dessus (p = 1)
- [x] Policy : latent = WM sur la partie contexte ; imitation de l'action de décision générée ; `loss` pondérée par `p.detach()` par séquence
- [x] UI principale (`main.py`) adaptée, app headless OK ; `test_brain.py` réécrit (18 tests)
- [x] Checkpoints legacy invalides : le buffer rejette l'ancien layout au load (run with `--reset`)

## À faire

- [ ] wm_theater / wm_manager / dream_theater : refaire à partir de leurs fonctionnalités sur le nouveau layout (comparaison ground-truth vs rollout WM, training epoch-by-epoch sur séquence sélectionnée)
- [ ] Optimisation du dataset (remise à plus tard par décision) : métadonnées riches, dedup, filtrage par surprise, pruning / oubli actif, tri plateau, pondération d'impact de la loss WM
- [ ] DESIGN.md, README.md, ADR-0002 (refonte), session log
- [ ] Nettoyer main.py des branches mortes (prune/meta/filter) quand l'optimisation dataset reviendra
- [ ] Perf génération : 81 forwards full-sequence par rollout — KV-cache à réintroduire avec positions explicites si besoin
