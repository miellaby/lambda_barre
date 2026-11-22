# Cycle de sommeil (`Brain.sleep_cycle`, `brain.py`)

## Déroulé du cycle

- Passer le cerveau en mode sommeil et mémoriser la loss WM précédente (`wm_loss_before_sleep`).
- Extraire les fenêtres du journal de veille dans l'addendum (`extract_addendum`, réordonnancement contexte / dénouement / conséquence inversée / action de décision) puis vider le journal.
- Choisir le mode de déduplication : distance cosinus sur le latent WM de la séquence complète si le WM a convergé (`last_wm_loss <= wm_target_loss`), sinon distance RMSE géométrique pondérée par token.
- Dédupliquer l'addendum en interne (`deduplicate_addendum`) : ne jamais fusionner si `max |Δa_dec| > 0.05` ou `|Δrelief| > 0.08` ; parmi de vrais doublons, conserver le meilleur `relief`, remettre `vivacity = 1.0` et cumuler `count`.
- Dédupliquer l'addendum contre le coreset (`deduplicate_against_coreset`) : même garde-fou `a_dec` / `relief` ; chaque doublon écarté rafraîchit `vivacity = 1.0` et cumule `count` sur le souvenir du coreset.
- Calculer immédiatement les statistiques déterministes du cycle (`addendum_kept`, `evicted`, `coreset_after`) et émettre l'événement `prune`.
- Si le coreset et l'addendum sont tous les deux vides :
  - Repasser en mode veille et émettre `done` avec des pertes `NaN`.
- Si le coreset est non vide et `wm_coreset_epoch > 0` :
  - Exécuter la phase d'entraînement WM sur le coreset (quota `wm_coreset_epoch`, événement `wm_coreset`).
- Si l'addendum est non vide et `wm_addendum_epoch > 0` :
  - Exécuter la phase d'entraînement WM sur l'addendum (quota `wm_addendum_epoch`, événement `wm_addendum`, avec replay mixte 50/50 depuis le coreset s'il est non vide).
- Calculer `last_wm_loss` (MSE moyenne), `wm_loss_ratio` et l'indicateur de plateau (`wm_loss_ratio > 0.95`).
- Si le buffer est non vide et `pol_steps > 0` (phase politique) :
  - Passer le World Model en mode évaluation (`eval`) et la politique en mode entraînement (`train`).
  - Pour chaque pas jusqu'à `pol_steps` :
    - Échantillonner un batch de séquences depuis le buffer (`coreset` + `addendum`).
    - Extraire le vecteur latent intermédiaire normalisé (`layer_norm`) depuis la passe causale du WM sur le contexte seul et les 51 scalaires sensoriels de l'état de décision.
    - Élire le dénouement optimal (`sample_denouements` : distribution MDN depuis le code VQ, noyau top-p et plancher `eos_min_p`, meilleur score de soulagement, règle de l'observé avec marge `rel_obs >= rel_elu - 0.10`) et calculer l'espérance `V_ctx`.
    - Utiliser l'action réelle enregistrée `a_real` comme cible `a*` sur les séquences gagnantes (`use_obs`), et ne générer la conséquence inversée puis `a*` par le WM en KV-cache que sur les séquences où un meilleur dénouement a été élu (`~use_obs`, plus l'index 0 pour Dream Theater).
    - Prédire l'action de la politique et calculer la MSE uniforme en espace `[0, 1]` vers `a*`, pondérée par confiance et avantage AWR `p_elu · relu(max(relief(élu) − V_ctx, relief(élu)))`.
    - Rétropropager, clipper le gradient à `1.0` et exécuter un pas d'optimiseur Adam (`opt_pol`).
    - Construire l'enregistrement Dream Theater (`_build_dream_record`) et émettre `pol`.
    - Si l'utilisateur demande à sauter la phase : interrompre immédiatement la phase.
- Consolider l'addendum dans le coreset (`commit_addendum`) :
  - Appliquer la décroissance de vivacité (`vivacity_decay`) sur tous les souvenirs existants du coreset.
  - Ajouter les séquences conservées de l'addendum au coreset avec `vivacity = 1.0`.
  - Si la taille du coreset dépasse `pool_capacity` :
    - Calculer les distances entre séquences complètes (latent WM si convergé, sinon RMSE géométrique).
    - Tant que le coreset dépasse `pool_capacity` : évincer dans la paire restante la plus proche le souvenir de moindre priorité `max(|relief|, impact_floor) · (0.5 + 0.5 · vivacity)` et transférer son `count` au voisin survivant.
- Réinitialiser l'état d'enregistrement de veille, repasser en mode veille et émettre `done` avec les statistiques finales du cycle (et sauvegarder les checkpoints côté application).

## Phase d'entraînement WM (`train_wm_phase` / `_train_wm_batch`, commun coreset et addendum)

- Passer le World Model en mode entraînement (`train`).
- Tant que le quota d'époques n'est pas atteint, ou que la MSE moyenne de l'époque dépasse `wm_target_loss`, ou qu'Adam est en slingshot (dans la limite `slingshot_max_extra`) :
  - Mélanger aléatoirement les indices de la source (`coreset` ou `addendum`).
  - Pour chaque batch sans remise de la source :
    - Si la source est `addendum` et que le coreset est non vide : compléter le batch à 50/50 avec des séquences tirées du coreset (replay mixte contre l'oubli catastrophique).
    - Ajouter un léger bruit gaussien (`wm_input_noise`) sur les slots signaux valides des entrées (hors préfixe, padding, `intero` et dénouement).
    - Effectuer un forward causal avec infusion du dénouement réel (`eos_cond_proj` sur `q >= eos_pos`), goulot VQ (`LayerNorm` + quantification), tête MDN (`eos_mdn_mlp`) et tête auxiliaire `(code VQ, EOS) → action de décision`.
    - Calculer la loss totale : MSE masquée pondérée par l'impact `max(|relief|, wm_impact_floor)` + `eos_nll_w` × (NLL MDN pondérée par `log1p(count)` + `eos_vq_beta` × engagement VQ) + `eos_vq_act_w` × MSE action auxiliaire.
    - Rétropropager la loss totale, clipper le gradient à `1.0`, exécuter un pas d'optimiseur Adam (`opt_wm`) puis mettre à jour le codebook VQ par EMA (avec réinitialisation des codes morts).
    - Enregistrer la MSE masquée pure comme score de surprise des séquences du batch et émettre l'événement de la phase (`wm_coreset` ou `wm_addendum`).
    - Si l'utilisateur demande à sauter la phase : interrompre immédiatement la phase.
  - Incrémenter `learn_count` sur chaque séquence de la source et mettre à jour le détecteur de slingshot sur la MSE moyenne de l'époque.
