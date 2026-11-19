# Tokens de λ̄

Tokens denses produits par DenseEncoder (tokenize.py). Lissage amont : encodeur.md.

## Token dense

29 flottants, sans embedding apprise :

- type (1) : 0 = état, 1 = action
- modalité (6) : one-hot, 1 bit allumé — intéroception `_M_INTERO` (100000), proprioception `_M_PROPRIO` (010000), extéroception `_M_EXTERO` (001000), renforcement `_M_RENF` (000100), action `_M_ACTION` (000010), dénouement `_M_EOS` (000001)
- canal (6) : one-hot dans la modalité (ids réutilisés entre modalités ; la modalité les sépare)
- signaux (16) : scalaires normalisés [0,1], zero-padding au-delà, exclus du masque de loss

## Salve

16 tokens par tick WM : 13 tokens d'état + 3 tokens d'action (5 valeurs effectives). Préfixe = type(1) + modalité(6 one-hot) + canal(6 one-hot) = 13 bits, puis label du canal :

```text
== proprioception (010000) ==
  tok  0  [0 010000 100000]  FORCE        3 sig  force_actuateur_avant/arriere, couple_queue
  tok  1  [0 010000 010000]  POSTURE      4 sig  tronc_angle, queue_angle, accel_tete_avant/haut
  tok  2  [0 010000 001000]  LIMB_F       2 sig  membre_angle_avant, membre_distance_avant
  tok  3  [0 010000 000100]  LIMB_B       2 sig  membre_angle_arriere, membre_distance_arriere
== extéroception (001000) ==
  tok  4  [0 001000 100000]  VISION      16 sig  vis_c1..16
  tok  5  [0 001000 010000]  BALL         4 sig  ball_dir, ball_prox, ball_vr, ball_va
  tok  6  [0 001000 001000]  FLOW         3 sig  flux_surface, flux_x, flux_y
  tok  7  [0 001000 000100]  SOUND        5 sig  son_0..4
  tok  8  [0 001000 000010]  TOUCH_FOOT   2 sig  contact_sol_avant/arrière
  tok  9  [0 001000 000001]  TOUCH_TRUNK  4 sig  collision_tronc_x/y/cx/cy
== renforcement (000100) ==
  tok 10  [0 000100 100000]  COSTS        5 sig  effort, douleur, courbature, instabilite, vertige
  tok 11  [0 000100 010000]  REWARD       1 sig  confort
== intéroception (100000) ==
  tok 12  [0 100000 100000]  METAB        2 sig  fatigue, souffrance
== dénouement (000001) ==
  EOS     [0 000001 100000]  EOS          6 sig  voir « Token EOS » ci-dessous
== action (000010) ==
  tok 13  [1 000010 100000]  ACT_F        2 sig  membre_avant_theta/d
  tok 14  [1 000010 010000]  ACT_B        2 sig  membre_arrière_theta/d
  tok 15  [1 000010 001000]  ACT_TAIL     1 sig  queue_theta
```

## Séquence

Fenêtre glissante de seq_steps + 1 = 13 salves (seq_steps = 12), n_ctx = 6, **lookahead = 10** : la fenêtre temporelle = s0..s11 consécutifs + la dernière salve s16 (saut de +lookahead=10 après la décision s6) — futur proche appris, dénouement lointain.

Séquence enregistrée (layout `compute_seq_layout`) : `contexte | dénouement | conséquence rétro-causale | action de décision` — le token 12 de la dernière salve (s16) est remplacé par le **token EOS** à `eos_pos = ctx_len`. La conséquence est enregistrée de la dernière salve vers la première (unités de 16 tokens : s_k inversée puis a_{k−1}), l'action de décision a_6 en dernier.

PE sinusoïdal non appris, keyed by **indice de salve** (tous les tokens d'une salve partagent la position) ; la dernière salve a la vraie position n_ctx+lookahead = 16, la conséquence rétro-causale garde ses positions décroissantes (16, 11→7), l'action de décision est à n_ctx. Le token EOS a son **PE dédié n_ctx + lookahead + 1 = 17** (hors fenêtre, token de synthèse).

## Token EOS

Token d'état (type 0), **modalité dédiée `_M_EOS`** (bit 6), canal 0, 6 slots. Par canal j (5 pénalités + inconfort = −confort) :

```
r[j]    = Σ_{k>d} eos_decay^(pos_k − d) · (c[k][j] − c[d][j]) · p_k   (d = n_ctx, L = seq_steps+1)
pos_k   = k, sauf dernière salve (k = L−1) à pos = n_ctx + lookahead = 16 (saut de la fenêtre)
p_k     = lookahead (=10) si k = L−1 (prolongation fictive du coût final), sinon 1
eos_decay = 0.8706  (dernier terme nu = 0.25, quart du début : 0.25^(1/lookahead) ; poids effectif du dernier coût = 0.25 × lookahead = 2.5)
slot[j] = 0.5 − 0.5·tanh(r[j])     0.5 = neutre ; amélioration (r<0) → slot > 0.5
```

Intégrale de l'écart au coût de la décision sur toute la conséquence, décroissance temporelle eos_decay. Ni seuil ni terme de chemin séparé.

relief_score(eos) = max flou p=8 des 6 slots, poids de canaux normalisés [1,4,2,2,3,1] : `relief = 2·(Σ w_j·slot_j^8)^(1/8) − 1` ∈ [−1,1]. impact = max(|relief|, wm_impact_floor).

## Forçage hindsight

Additif par slot, eps ~ U(0.015, 0.0375) (eos_force_eps_lo/hi), plancher de neutralité à 0.5.

## Modèles

- world model : entrée 29→d_model 128, séquence de 205 tokens (13 salves, fenêtre à saut lookahead), masque causal pur + scale multiplicatif d'entrée sur contexte/dénouement (une seule fois, après in_proj avant le PE), double tête : 128→16 signaux (MSE + sigmoid) et tête EOS 128→M logits (codebook de M=32 prototypes appris par k-means sur les EOS observés ; cross-entropy sur le prototype le plus proche — sampling LM-style nucleus top-p sur le logp appris, plancher eos_min_p). Le préfixe n'est jamais prédit ni appris.
- policy : tick 10 Hz (pol_hz), entrée 51 scalaires (tokens 0..11, hors bilan métabolique) + latent 128 = 179, MLP 179→512→512→512→5 (hidden = 512).
