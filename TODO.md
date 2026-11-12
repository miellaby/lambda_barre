- bugfix: le mask fait que la passe lookahead tient compte du contexte?
il faut que l'action-décision voit l'EOS, mais aussi le contexte bien évidemment.
- calculer la progression de la loss WM. préserver la loss WM
  avant sommeil, calculer et stocker le ratio avec la nouvelle loss WM.
  Si le ratio apres/avant > 0.95 = plateau WM
- rajouter des métadonnées dans les séquences enregistrés:
  * relief_score score de soulagement par l'EOS enregistré
  * compteur d'apprentissage (+1 à chaque apprentissage causal dans un batch WM)
  * vivacité (déja fait)
  * causal_score similarité cosin de l'EOS causal et enregistré
  * lookahead_score similarité cosin de la décision lookahead et enregistrée.
  * dissonance: distance cosin entre l'action enregistrée et l'action politique
  * surprise: loss au dernier entrainement causal
  * latent: vecteur latent (capturé de façon similaire au latent  
    intermédiaire d'entrée de politique) à la génération du token EOS
  * score de soulagement politique (voir ci-après)
- au début du sommeil, pour chaque séquence coreset/addendum. Faire les
  passes causale et lookahead.
  Et faire tourner la politique sur le contexte enregistré
  puis le WM causal sur la suite de l'action politque.
  Puis stocker dans les métadonnées:
  * latent: vecteur capturé dans la passe causale au moment de générer
    le token EOS.
  * dissonance
  * causal_score
  * lookahead_score
  * score de soulagement politique
- Combiner la distance entre séquences pondérées par la salience
  avec une distance cosin entre les latents des séquences.
  Combiner (additionner/max) les 2 distances pour le clustering, le seuillage...
- si plateau WM atteint uniquement, faire une version triée
 *temporaire* du coreset, triée par min-flou sur les valeurs
  normées et pondérées de:
      * déterminisme = (lookahed_score + causal_score) / 2
      * consonnance = l'inverse de la dissonance
      * vivacité
  Les moins déterministes et les moins vivaces en premier.
- Si le coreset dépasse la capacité de m séquences, 
  supprimer du coreset les m première séquences du coreset trié
  (supprimer aussi dans le coreset). Ne plus faire d'écratage du
  coreset à d'autres endroits.
- apprentissage WM. faire de vrais epoch WM pour coreset et addendum:
  * 1 epoch = tout le dataset réparti en batchs sans omission ni remise
    de séquences. Batcher dans l'ordre du coreset trié.
  * L'epoch passe à 5 dans les configs.
- l'oubli conditionné par la surprise ne concerne que les nouvelles séquences
  de l'addendum (compteur apprentissage == 0 avant le premier entrainement)
  à la premiere epoch: la séquence peut être supprimée aussitot (pas dans les epoch suivantes).
- apps wm_manager/wm_theater: reporter les changements. pour l'entrainement
  rajouter la séquence sélectionnée dans tous les batchs.
- décalage action/conséquence à l'enregistrement en mode brain-off
- si brain off, enregistrer chaque paire action-conséquence en mode $(a_n\ ;\ s_{n-k})$ avec $k$ paramétrable (k>=0) défaut 2. Enregistrement Brain ON équivaut à k forcé à 0
- mettre à jour BACKLOG.md
- faire des propositions d'ajoute dans ADR-0001-nomenclature.md
- mettre à jour IDEAS.md et DESIGN.md
- renommer Brain.sleep() en sleep_cycle() (traduction de "cycle de sommeil"),
  mettre à jour les appelants (main.py, tests, docs) — FAIT 2026-11-12
- à creuser: poids du terme latent dans la distance combinée dedup
  (latent_weight=1.0 par défaut, addition vs max non tranché)
- à creuser: amplitude du bruit sur le critère de tri (sorted_coreset_indices noise=0.05)
- à creuser: vivacity_pow (p) actuellement 1.0, tester > 1

