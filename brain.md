# Le cerveau de λ̄ — l'IA à 2 modèles

Ce document décrit la mise en œuvre du système de décision apprenant de λ̄ :
un **world model** (transformer autorégressif) et une **politique** (réseau de
renforcement), entraînés en continu selon l'architecture décrite dans
`Lambda barre.md` § « Architecture du système de décision apprenant ».

L'implémentation se compose de trois modules :

```
lambda_barre/
├── models.py     ← les 2 réseaux PyTorch (WorldModel, Policy)
├── brain.py       ← orchestrateur (act / record / sleep)
└── test_brain.py  ← tests de bout en bout (sans fenêtre)
```

ainsi que des helpers ajoutés à `tokenize.py` et du branchement dans `main.py`.

Sources de la spec : `Lambda barre.md` (architecture, apprentissage, mise en
œuvre), `encodeur.md` (layout des tokens), `interoception.md` (états internes),
`proprioception.md` (circuit de reward).

---

## Vue d'ensemble

```
                état (salve de 67 tokens)
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
       WORLD MODEL         POLITIQUE
             │                 │
       état + action           │
             │                 │
             ▼                 ▼
      conséquence            action
      prédite (s')           (5 consignes)
             │                 │
             └───────┬─────────┘
                     ▼
                  action ──► environnement (physique pymunk)
                     │
                     ▼
                nouvel état
```

Pendant l'**éveil**, la politique pilote les actuateurs et chaque transition
`(s, a, s')` est journalisée. Pendant le **sommeil**, le world model est
entraîné hors-ligne sur ces transitions, puis la politique est entraînée par
RL sur des trajectoires *imaginées* par le world model.

---

## Le world model (`models.py` — `WorldModel`)

Un **transformer autorégressif** sur le flux de tokens multimodaux.

### Tokens

Chaque token est une paire `(id, valeur)` sur 2 octets (voir `encodeur.md`).
L'id identifie le type de signal (structural, fixe par le layout de la salve) ;
la valeur est quantifiée sur 256 niveaux. Le world model prédit la **valeur**
du token suivant — c'est cette prédiction discrète sur 256 classes qui permet
une distribution de trajectoires **multi-modale** plutôt qu'une gaussienne
centrée (point clé de la spec).

### Embedding

```
x = (id_embed(id) + val_embed(valeur) + pos_embed(position)) * sqrt(d_model)
```

L'id et la valeur ont chacun leur table d'embedding (256 × d_model), la
position a la sienne. La somme est normalisée par `sqrt(d_model)`.

### Architecture

- `d_model = 96`, `nhead = 4`, `layers = 2`, `dim_ff = 384` (taille réduite
  pour la vitesse CPU)
- `TransformerEncoder` avec `norm_first=True` (pre-LN) et activation GELU
- Masque causal triangulaire (chaque position n'attend que les précédentes)
- Tête linéaire → 256 logits (prédiction de la valeur du token suivant)

### Séquence d'entraînement

Une transition est encodée comme une séquence de 128 tokens :

```
[état_t (61)] [action_t (6)] [état_{t+1} (61)]
```

Le modèle est entraîné par **teacher forcing** : à chaque position, prédire la
valeur du token suivant. La loss (cross-entropy) ne porte que sur les
positions dont la cible est un **scalaire** — les tokens de séparation
(valeur toujours 0) sont masqués car ils ne portent pas de signal.

### Méthodes

| Méthode | Rôle |
|---------|------|
| `forward(ids, vals)` | logits `[B, L, 256]` prédits à chaque position |
| `roll(ctx, n_gen)` | génération **autorégressive** token par token (lente) |
| `predict_next(ctx)` | prédiction **parallèle** du prochain état en une passe (rapide, utilisée pour l'imagination) |

`predict_next` conditionne chaque token du prochain état sur
`(état, action)` + un placeholder zéro pour le reste du prochain état. C'est
~60× plus rapide que `roll` et c'est ce que la politique utilise pour
imaginer des trajectoires pendant le sommeil. Le world model lui-même reste
entraîné avec l'objectif causal autorégressif complet.

---

## La politique (`models.py` — `Policy`)

Un **MLP** `π(a|s)` qui produit les 5 consignes d'actuateurs.

### Entrée

Les **55 valeurs scalaires sensorielles** de l'état (la partie état de la
salve, séparateurs exclus), normalisées sur `[0, 1]`.

### Architecture

```
Linear(55 → 128) → Tanh
Linear(128 → 128) → Tanh
Linear(128 → 128) → Tanh
Linear(128 → 5)            # raw (logit-space)
```

### Squash

Les 5 sorties sont squashed dans les plages valides des consignes :

| Sortie | Transformation | Plage |
|--------|----------------|-------|
| limb_l_theta | `tanh(raw) * π/2` | [−π/2, +π/2] |
| limb_l_d | `LIMB_MIN + sigmoid(raw)*(LIMB_MAX−LIMB_MIN)` | [10, 32] |
| limb_r_theta | `tanh(raw) * π/2` | [−π/2, +π/2] |
| limb_r_d | `LIMB_MIN + sigmoid(raw)*(LIMB_MAX−LIMB_MIN)` | [10, 32] |
| tail_theta | `tanh(raw) * π/2` | [−π/2, +π/2] |

Ainsi toute action échantillonnée est une consigne valide pour la physique.

### Exploration

Échantillonnage dans l'espace **pre-squash** (logit) avec une gaussienne
isotrope de std fixe (`explore_std = 0.4`), puis squash. Le `log_prob` est
calculé contre la même distribution dans l'espace logit (simplification
standard : on optimise en espace raw).

---

## L'orchestrateur (`brain.py` — `Brain`)

### Cycle de vie

```
brain.act(salve_t)              → 5 consignes d'actuateurs
    ... physique, capteurs, encodeur ...
brain.record(salve_t, salve_{t+1})   → journalise (s, a, s')
brain.sleep()                        → entraîne les 2 modèles hors-ligne
```

### Éveil

- `act()` : échantillonne une action depuis la politique (exploration) et la
  renvoie. Les consignes sont poussées dans le squelette par `main.py`.
- `record()` : journalise une transition dans le `ExperienceBuffer` (ring
  buffer de capacité 4000). Les états sont stockés comme listes de 61 valeurs
  de tokens (séparateurs inclus, valeur 0), les actions comme 6 valeurs
  (séparateur + 5 scalaires).

### Sommeil — `sleep()`

Deux phases, dans l'ordre de la spec :

**1. Entraînement du world model** (`train_world`) :
- Échantillonne des batches de transitions depuis le buffer
- Construit les séquences `[s, a, s']` de 128 tokens
- Cross-entropy sur les positions scalaires seulement
- Clip de gradient (norme 1.0), optimizer Adam (lr 3e-4)

**2. Entraînement de la politique** (`train_policy`) — model-based RL :
- Le world model est **gelé** (`eval`)
- Pour chaque batch d'états de départ :
  1. La politique **échantillonne** une première action (garde le gradient)
  2. Le world model **imagine** la trajectoire sur `horizon` pas
  3. À chaque pas : coût lu sur les tokens REWARD de l'état imaginé
  4. Return = −Σ coût actualisé (maximiser ⇔ minimiser le coût prédit)
- **REINFORCE** : `loss = −E[log π(a|s) · (return − baseline)]`
- Baseline = moyenne glissante des returns imaginés (stabilise la variance)
- Clip de gradient, optimizer Adam (lr 1e-3)

C'est la boucle décrite dans la spec : la politique apprend à produire des
actions qui **minimisent le coût prédit par le world model**, sans avoir
besoin d'effectuer réellement toutes ces actions.

### Coût d'une salve (`salve_cost`)

Le coût instantané est lu sur les tokens REWARD de l'état :

```python
cost = reward_neg (coût non signé, 0..1) + reward_pos (signé, −1..+1)
```

`reward_pos` négatif = récompense (réduit le coût) ; `reward_neg` positif =
pénalité (augmente le coût). Les valeurs sont déquantifiées vers leurs
échelles physiques (1.0 pour les deux).

---

## Helpers de tokenization (`tokenize.py`)

Ajoutés pour l'interface entre le cerveau et l'encodeur :

| Helper | Rôle |
|--------|------|
| `SALVE_IDS` | layout plat des 67 ids d'une salve (séparateurs + scalaires) |
| `STATE_IDS` / `ACTION_IDS` | ids des 61 tokens d'état / 6 tokens d'action |
| `STATE_LEN = 61` / `ACTION_LEN = 6` / `SALVE_LEN = 67` | tailles |
| `state_values(salve)` | 55 valeurs scalaires de l'état (pour la politique) |
| `action_values(salve)` | 5 valeurs scalaires de l'action |
| `state_token_values(salve)` | 61 valeurs de tokens d'état (pour le world model) |
| `action_token_values(salve)` | 6 valeurs de tokens d'action |
| `encode_action(5 consignes)` | encode les consignes en tokens ACTION |
| `salve_cost(salve)` | coût instantané lu sur les tokens REWARD |
| `state_vals_from_scalars` / `scalars_from_state_vals` | conversion 55 ↔ 61 |

---

## Branchement dans la boucle live (`main.py`)

### Contrôles clavier

| Touche | Action |
|--------|--------|
| **B** | Active/désactive le cerveau (mode auto vs joysticks manuels) |
| **S** | Déclenche un cycle de sommeil (entraîne les 2 modèles) |
| **R** | Récompense utilisateur (impulsion valencée positive) |
| **T** | Punition utilisateur (impulsion valencée négative) |
| BACKSPACE | Reset (efface aussi le buffer de transitions) |
| G | Toggle des marqueurs de cible |
| ESC | Quitter |

### Reward / punition utilisateur

Conformément à la spec, le reward n'est **pas une supervision spéciale** :
c'est un **événement sensoriel à valeur intrinsèque** injecté dans les tokens
REWARD. La touche **R** rend `reward_pos` plus négatif (réduit le coût) ; **T**
augmente `reward_neg` (augmente le coût). Les impulsions décroissent
exponentiellement (×0.9 par frame).

### Cadencement

La salve est générée à **6 Hz** (une salve toutes les 10 frames à 60 Hz),
conformément à la spec. C'est la cadence du world model, plus lente que la
proprioception (60 Hz) ou le curseur (30 Hz).

### HUD

Le HUD affiche en plus, quand le cerveau est actif :

```
fps 60 facing R [BRAIN]
buf 35 wm 29.70 pol 2.70 ret 3.44
```

— taille du buffer, dernière loss du world model, dernière loss de politique,
dernier return imaginé.

---

## Configuration technique

### PyTorch

PyTorch (build CPU) est ajouté aux dépendances (`torch>=2.0`).

**Note de compatibilité** : sur certains hôtes (ex. conteneurs avec un
`/proc/cpuinfo` inutilisable), les noyaux fusionnés oneDNN (MKL-DNN) lèvent un
SIGILL. `models.py` force donc par défaut le backend d'attention « math »
avec oneDNN désactivé et 1 thread. Pour restaurer les noyaux optimisés sur une
machine connue pour les supporter :

```bash
export LAMBDA_TORCH_MKLDNN=1
```

### Performances

Un cycle de sommeil complet (4 epochs WM + 12 pas de politique, batch 16,
horizon 3) prend ~5–10 s sur CPU. La prédiction parallèle `predict_next`
(une seule passe plutôt que 61 passes autorégressives) rend l'imagination
pratique pour le RL hors-ligne.

---

## Tests (`test_brain.py`)

Tous les tests tournent sans fenêtre (`SDL_VIDEODRIVER=dummy`).

| Test | Vérifie |
|------|---------|
| `test_policy_outputs_are_valid_consignes` | la politique renvoie 5 consignes dans les plages valides |
| `test_world_model_forward_and_predict_shapes` | formes du forward et de `predict_next` |
| `test_salve_cost_sign` | reward → coût négatif, punition → coût positif |
| `test_act_returns_five_consignes` | `act()` renvoie 5 valeurs |
| `test_sleep_trains_both_models_and_world_loss_decreases` | la loss du world model décroît sur données structurées |
| `test_brain_drives_real_sim_and_sleeps` | **intégration réelle** : le cerveau pilote le corps pymunk, journalise les vraies transitions, et enchaîne un sommeil |

Lancer :

```bash
SDL_VIDEODRIVER=dummy PYTHONPATH=. python -m lambda_barre.test_brain
```

Le test d'intégration reproduit la boucle live sans affichage : construit le
squelette pymunk, les capteurs, l'encodeur, laisse le cerveau piloter les
consignes pendant 4 s, puis déclenche un sommeil et vérifie que les loss
sont finies et que la politique post-sommeil produit encore des consignes
valides.

---

## Périmètre (minimal viable)

Conformément au choix « minimal viable », les briques suivantes de la spec
sont **laissées en stub** pour une iteration ultérieure :

- **Coreset / Addendum** : la consolidation par ablation pendant le sommeil
  (§ « Consolidation de l'apprentissage durant le sommeil »). Le buffer actuel
  est un ring buffer simple.
- **Segmentation d'épisodes** : la détection automatique des frontières
  d'épisode par l'erreur de prédiction `L_t = −log P(o_{t+1}, r_t | o_t, a_t)`
  (§ « Délimitation des séquences du dataset »).
- **Compression temporelle des actions** : le résumé des blocs moteurs entre
  deux salves (§ « Granularité temporelle »).
- **Vocalises** : l'actuateur vocal et l'instinct d'imitation.

L'objectif démontrable avec les seuls coûts innés est l'émergence d'une
posture stable (lutter contre gravité et instabilité). L'émergence de
« suivre le curseur » nécessite les reward/punish (R/T) et plusieurs cycles
veille/sommeil — c'est le test décrit dans la spec : après apprentissage,
arrêter de récompenser et observer si λ̄ continue spontanément à suivre le
curseur.
