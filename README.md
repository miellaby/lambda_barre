# λ̄ (Lambda barre)

**λ̄** est un projet d'**animat desktop pet apprenant** : un animal virtuel autonome doté d'une morphologie biomécanique simplifiée en 2D, d'un modèle du monde autorégressif (Transformer), d'un apprentissage par renforcement sur un champ sensoriel multimodal réaliste et d'un système inné de renforcement/confort.

---

## Installation

Le projet utilise Python (>= 3.10) et `uv` (ou `pip`) :

```bash
# Création de l'environnement virtuel
uv venv
source .venv/bin/activate

# Installation des dépendances en mode éditable
uv pip install -e .
```

---

## Ligne de commande

### Lancement

```bash
# Lancement sans aucune accélération matérielle, compatible tout CPU
.venv/bin/python -m lambda_barre.main

# Lancement avec accélération matérielle maximale (CUDA)
.venv/bin/python -m lambda_barre.main --accel

# Sélection du composant d'exécution
.venv/bin/python -m lambda_barre.main --device cuda
.venv/bin/python -m lambda_barre.main --device auto
.venv/bin/python -m lambda_barre.main --device cpu
.venv/bin/python -m lambda_barre.main --device cpu-fast # MKL-DNN
```

### Options

| Option | Description | Valeurs possibles |
|---|---|---|
| `--accel` | Active l'accélération matérielle maximale (alias de `--device auto`). | Drapeau booléen |
| `--device` | Spécifie le matériel de calcul pour les réseaux de neurones. | `cpu` *(défaut)*, `cuda`, `auto`, `cpu-fast` |
| `--reset` | Supprime les checkpoints existants (`wm_ckpt.pt`, `pol_ckpt.pt`) et démarre avec des poids neufs. | Drapeau booléen |
| `--headless` | Exécute la simulation sans fenêtre Pygame (nécessite `--steps`). | Drapeau booléen |
| `--steps N` | Nombre de frames (60 Hz) à simuler en mode headless avant arrêt. | Entier (ex: `300`) |

Options passées par l'environnement :

* **`LAMBDA_ACCEL=1`** : `--accel`.
* **`LAMBDA_DEVICE=cuda`**: `--device cuda`
* **`LAMBDA_TORCH_MKLDNN=1`** : `--device cpu-fast`
* **`SDL_VIDEODRIVER=dummy`** : Idéal pour le mode headless.

## Contrôles (GUI)

| Touche / Action | Fonction |
|---|---|
| **`ÉCHAP`** | Quitter |
| **`ESPACE`** ou **`B`** | *brain off/on* |
| **`S`** | Déclencher une phase de **Sommeil** (entraînement offline) |
| **`R`** | **Reset** : replace l'animat à sa position de spawn |
| **`G`** | Masquer les consignes |
| **joysticks** | Pilote l'animat en mode manuel |
| **Clic droit + glisser sur plateforme** | Déplacer une plateforme |
| **Souris** | Déplacer le curseur perçu par l'animat |

---

## Architecture & Cadences temporelles

```
               [ Simulation Physique 60 Hz ]
                Pymunk 2D, Contacts, Forces
                             │
            ┌────────────────┴────────────────┐
            ▼ (3 Hz)                          ▼ (6 Hz)
    [ World Model ]                      [ Policy ]
 Transformer autorégressif               MLP Acteur
 Prédit des salves de tokens         Consignes motrices (5)
  (13 états + 3 actions)          (Pattes avant/arrière, queue)
```

* **Physique & Capteurs (60 Hz)** : Squelette articulé Pymunk et capteurs: proprioception, extéroception et intéroception.
* **World Model (3 Hz)** : Tokenisation dense de 16 tokens par salve et 25 dimensions par token. Modèle autorégressif causal qui imagine un horizon de 2 secondes.
* **Policy (6 Hz)** : Sélectionne l'action motrice via 47 scalaires sensoriels et l'état latent extrait du World Model.

Pendant la **phase de Sommeil (`S`)** : Entraînement hors-ligne du World Model (prédiction auto-supervisée du futur) et de la Policy (tournoi de candidats guidé par les futurs imaginés par le World Model).

---

## Tests

```bash
SDL_VIDEODRIVER=dummy PYTHONPATH=. .venv/bin/pytest
```
