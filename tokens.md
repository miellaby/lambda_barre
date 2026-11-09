# Tokens de λ̄

Tokens denses produits par DenseEncoder (tokenize.py). Lissage amont : encodeur.md.

## Token dense

25 flottants, sans embedding apprise :

- type (1) : 0 = état, 1 = action
- modalité (4) : action `_M_ACTION` (0,0,0,0), intéroception `_M_INTERO` (1,0,0,0), proprioception `_M_PROPRIO` (0,1,0,0), extéroception `_M_EXTERO` (0,0,1,0), renforcement `_M_RENF` (1,0,0,1)
- canal (4) : sous-système dans la modalité
- signaux (16) : scalaires normalisés [0,1], zero-padding au-delà, exclus du masque de loss

## Salve

16 tokens par tick WM (wm_hz = 5 Hz) : 13 tokens d'état + 3 tokens d'action (5 valeurs effectives). Préfixe = type+modalité+canal en binaire (9 bits), puis label du canal :

```text
== proprioception ==
  tok  0  [001000001]  FORCE        3 sig  force_actuateur_avant/arriere, couple_queue
  tok  1  [001000010]  POSTURE      4 sig  tronc_angle, queue_angle, accel_tete_avant/haut
  tok  2  [001000100]  LIMB_F       2 sig  membre_angle_avant, membre_distance_avant
  tok  3  [001001000]  LIMB_B       2 sig  membre_angle_arriere, membre_distance_arriere
== extéroception ==
  tok  4  [000101111]  VISION      16 sig  vis_c1..16
  tok  5  [000100001]  BALL         4 sig  ball_dir, ball_prox, ball_vr, ball_va
  tok  6  [000100010]  FLOW         3 sig  flux_surface, flux_x, flux_y
  tok  7  [000100100]  SOUND        5 sig  son_0..4
  tok  8  [000101000]  TOUCH_FOOT   2 sig  contact_sol_avant/arrière
  tok  9  [000101010]  TOUCH_TRUNK  4 sig  collision_tronc_x/y/cx/cy
== renforcement ==
  tok 10  [010010000]  COSTS        5 sig  effort, douleur, courbature, instabilite, vertige
  tok 11  [010011000]  REWARD       1 sig  confort
== intéroception ==
  tok 12  [010000001]  METAB        2 sig  fatigue, souffrance
  tok 12b [010000010]  EOS          6 sig  dernier token, delta renforcement normalisé (tok 10 et 11)
== action ==
  tok 13  [100000001]  ACT_F        2 sig  membre_avant_theta/d
  tok 14  [100000010]  ACT_B        2 sig  membre_arrière_theta/d
  tok 15  [100000100]  ACT_TAIL     1 sig  queue_theta
```

## Séquence

Fenêtre glissante de seq_steps + 1 = 27 salves (seq_steps = 26). n_ctx = 4 : contexte de décision s0..s4 (s4 = état de la prise de décision, qui conditionne a4), fenêtre de conséquence s5..s26 (22 salves). À l'entraînement, le token 12 de la dernière salve est reconstruit en token EOS.

## Token EOS

Token d'état, modalité intéroception, canal (0,0,1,0), 6 slots : delta brut de chaque signal de coût (5 pénalités + récompense) entre fenêtre de conséquence et contexte de décision, normalisé (d+1)/2. 0.5 = neutre. Ni amplifié, ni seuillé.

- contexte de décision : s4 seul (one-hot)
- fenêtre de conséquence : poids linéaires de 1 à 2, le dernier état pèse 2× le premier

relief_score(séquence) = relief_score(eos(séquence)) = somme pondérée des 6 slots, poids [1,4,2,2,3,1]. impact = abs(relief_score).

## Forçage hindsight

Additif par slot, eps ~ U(0.015, 0.0375) (eos_force_eps_lo/hi), plancher de neutralité à 0.5.

## Modèles

- world model : entrée 25→d_model 128, séquence de 429 tokens (27 salves), masque causal par salve, tête 128→16.
- policy : tick 10 Hz (pol_hz), entrée 51 scalaires (tokens 0..11, hors bilan métabolique) + latent 128 = 179, MLP 179→512→512→512→5 (hidden = 512).
