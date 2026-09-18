# Densification des tokens — plan

## Contexte

Actuellement chaque signal sensoriel est un token séparé (65 tokens par salve : 59 état + 6 action). Chaque token est un entier (id + valeur) projeté dans 96 dimensions par des embeddings appris. C'est comme alimenter GPT avec des bits individuels.

Le nouveau schéma : un token par canal (groupe de signaux), avec un encodage manuel concaténé. Pas d'embedding appris.

## Structure du token dense

Un token = vecteur de 25 floats, concaténation de :

| Section | Dims | Contenu |
|---------|------|---------|
| type | 1 | 0.0 = sensoriel, 1.0 = moteur |
| modalité | 4 | encodage manuel orthogonal (interne, somatosensoriel, visuel, motivationnel) |
| canal | 4 | encodage manuel dans la modalité (ex: avant=1.0/arrière=0.0) |
| signaux | 16 | un scalaire normalisé [0,1] par dimension, zero-pad si < 16 signaux |

Total : 25 dimensions. Pas d'embedding appris — l'encodeur produit ces vecteurs directement.

## Hiérarchie des tokens

### Tokens sensoriels (13 tokens d'état)

| # | Modalité | Encodage modalité (4d) | Canal | Encodage canal (4d) | Signaux | Nb |
|---|----------|----------------------|-------|---------------------|---------|-----|
| 0 | PROPRIO | (1,1,0,0) | Feedback | (0,0,0,1) | force_actuateur_avant, force_actuateur_arriere, couple_queue | 3 |
| 1 | PROPRIO | (1,1,0,0) | Posture | (0,0,1,0) | tronc_angle, queue_angle, accel_tete_avant, accel_tete_haut | 4 |
| 2 | PROPRIO | (1,1,0,0) | Membre avant | (0,1,0,0) | membre_angle_avant, membre_distance_avant | 2 |
| 3 | PROPRIO | (1,1,0,0) | Membre arrière | (1,0,0,0) | membre_angle_arriere, membre_distance_arriere | 2 |
| 4 | VISION | (0,0,1,0) | Luminosité | (1,1,1,1) | vis_c1..vis_c16 | 16 |
| 5 | ENV | (0,0,1,1) | Curseur | (0,0,0,1) | curseur_dir, curseur_prox, curseur_vx, curseur_vy | 4 |
| 6 | ENV | (0,0,1,1) | Flux optique | (0,0,1,0) | flux_surface, flux_x, flux_y | 3 |
| 7 | ENV | (0,0,1,1) | Son | (0,1,0,0) | son_0, son_1, son_2, son_3, son_4 | 5 |
| 8 | TOUCH | (0,1,0,0) | Contact sol | (0,0,0,1) | contact_sol_avant, contact_sol_arriere | 2 |
| 9 | TOUCH | (0,1,0,0) | Collision | (0,0,1,0) | collision_tronc_x, collision_tronc_y, collision_tronc_cx, collision_tronc_cy | 4 |
| 10 | INTERO | (1,0,0,0) | État interne | (1,1,1,1) | fatigue, souffrance | 2 |
| 11 | REWARD | (1,0,0,1) | Coûts | (0,0,0,0) | effort, douleur, courbature, instabilite, vertige | 5 |
| 12 | REWARD | (1,0,0,1) | Récompense | (1,0,0,0) | confort | 1 |

### Tokens moteurs (3 tokens d'action)

| # | Modalité | Encodage modalité (4d) | Canal | Encodage canal (4d) | Signaux | Nb |
|---|----------|----------------------|-------|---------------------|---------|-----|
| 13 | ACTION | (0,0,0,0) | Patte avant | (0,0,0,1) | membre_avant_theta, membre_avant_d | 2 |
| 14 | ACTION | (0,0,0,0) | Patte arrière | (0,0,1,0) | membre_arriere_theta, membre_arriere_d | 2 |
| 15 | ACTION | (0,0,0,0) | Queue | (0,1,0,0) | queue_theta | 1 |

## Changements à faire

### tokenize.py
- Remplacer toute la structure : séparateurs, `_GROUP_LAYOUT`, `SALVE_IDS`, `STATE_IDS`, etc.
- Nouvelle classe `DenseEncoder` qui produit des `list[list[float]]` (16 × 25)
- Supprimer `TokenEncoder` (l'ancien format (id, value))
- Garder `_SPECS`, `_BY_KEY`, `_quantize`, `_dequantize`, `salve_cost`
- `encode_action` retourne 3 tokens denses au lieu de 6 tuples
- `policy_scalars` extrait les 47 valeurs non-reward depuis les tokens denses
- `salve_cost` lit les canaux Coûts et Récompense

### models.py
- Supprimer `id_embed`, `val_embed`, `pos_embed`
- Le `forward` reçoit directement `[B, L, 25]` au lieu de ids+vals
- Ajouter un encodage positionnel sinusoidal non-appris (ou supprimer, à discuter)
- `d_model = 96` (passer en entrée de 25 à 96 par projection linéaire)
- `predict_next` adapté : prédit les 13 tokens d'état
- La head passe de classification 9 classes à régression (prédire les floats de chaque token)

### brain.py
- `ExperienceBuffer` stocke des `list[list[float]]` au lieu de `list[int]`
- `_batch_tensors` construit des tenseurs `[B, L, 25]`
- `_MASK` masque pour transformer causal
- `wake_tick`, `act`, `train_world`, `train_policy` adaptés

### smoother.py
- `salve()` appelle `DenseEncoder` au lieu de `TokenEncoder`

### main.py
- Remplacer `TokenEncoder` par `DenseEncoder`

### test_brain.py
- Adapter tous les tests

## Autre choix

* **d_model** : garder 96 (avec projection linéaire 25→96 en entrée)
* **Position encoding** : sinusoidal non-appris mais mettre la même position pour tous les tokens d'une salve: il s'agit d'encoder la position de la salve (c'est à dire le temps) et non les tokens entre eux (irrelevant).
*. **Masque par groupe** non, **Masque par salve dans le type**: quand on prédit Si, on masque les tokens sensoriels de la salve i.
* **Head** : régression (MSE) — `nn.Linear(96, 16)`, préserve l'ordre des valeurs, simple, pas de multimodalité nécessaire en stage 1.
* **LayerNorm du latent** : pertinent — normalise le latent 96-dim avant concaténation avec les 47 scalaires [0,1] de la policy. 192 paramètres appris.
* **Policy** : N_POLICY_STATE = 47 (non-reward signals) + 96 (latent d_model) = 143 entrées.
