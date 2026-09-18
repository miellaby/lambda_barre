# λ̄ (Lambda barre)

**λ̄** est un projet d'**animat desktop pet apprenant** : un animal virtuel autonome doté d'une morphologie biomécanique simplifiée en 2D, d'un modèle du monde autorégressif (Transformer), d'un apprentissage par renforcement sur un champ sensoriel multimodal réaliste et d'un système inné de renforcement/confort.

---

## 1. Installation

Le projet utilise Python (>= 3.10) et `uv` (ou `pip`) :

```bash
# Création de l'environnement virtuel
uv venv
source .venv/bin/activate

# Installation des dépendances en mode éditable
uv pip install -e .
```

---

## 2. Utilisation en ligne de commande (CLI)

### Lancement standard (Interface graphique)

```bash
# Lancement sécurisé (mode CPU Safe par défaut)
.venv/bin/python -m lambda_barre.main

# Lancement avec accélération matérielle maximale (CUDA si disponible)
.venv/bin/python -m lambda_barre.main --accel

# Sélection explicite du composant d'exécution
.venv/bin/python -m lambda_barre.main --device cuda
.venv/bin/python -m lambda_barre.main --device auto
.venv/bin/python -m lambda_barre.main --device cpu
.venv/bin/python -m lambda_barre.main --device cpu-fast

# Réinitialiser les poids appris (supprime les checkpoints et repart à zéro)
.venv/bin/python -m lambda_barre.main --reset
```

### Options de la ligne de commande

| Option | Description | Valeurs possibles |
|---|---|---|
| `--accel` | Active l'accélération matérielle maximale (alias de `--device auto`). | Drapeau booléen |
| `--device` | Spécifie le matériel de calcul pour les réseaux de neurones. | `cpu` *(défaut)*, `cuda`, `auto`, `cpu-fast` |
| `--reset` | Supprime les checkpoints existants (`wm_ckpt.pt`, `pol_ckpt.pt`) et démarre avec des poids neufs. | Drapeau booléen |
| `--headless` | Exécute la simulation sans fenêtre Pygame (nécessite `--steps`). | Drapeau booléen |
| `--steps N` | Nombre de frames (60 Hz) à simuler en mode headless avant arrêt. | Entier (ex: `300`) |

### Variables d'environnement

Les options matérielles peuvent également être configurées via l'environnement sans modifier la ligne de commande :

* **`LAMBDA_ACCEL=1`** : Équivalent à `--accel`.
* **`LAMBDA_DEVICE=cuda`** (ou `cpu`, `auto`, `cpu-fast`) : Définit le matériel par défaut.
* **`LAMBDA_TORCH_MKLDNN=1`** : Active l'accélération MKL-DNN sur CPU.
* **`SDL_VIDEODRIVER=dummy`** : Permet d'exécuter l'application sur un serveur distant sans serveur X/Wayland.

---

## 3. Profils matériels & Compatibilité

Pour concilier compatibilité maximale et performances extrêmes, 3 modes matériels sont gérés :

1. **CPU Safe (Mode par défaut)** :
   * Désactive MKL-DNN/oneDNN et force l'attention mathématique portable.
   * **Garantie zéro crash (SIGILL)** sur les machines virtuelles Linux ou conteneurs dont les drapeaux `/proc/cpuinfo` sont incomplets.
2. **CUDA GPU (`--device cuda` ou `--accel`)** :
   * Détecte et utilise le GPU NVIDIA (ex: RTX 2060).
   * Active FlashSDP, MemEfficient attention et cuDNN benchmark.
   * Réduit la durée d'un cycle de sommeil complet de **plusieurs minutes à ~8–15 secondes** ($\times 20$).
   * *Repli automatique* : si CUDA n'est pas disponible, bascule proprement sur le CPU Safe avec un avertissement.
3. **CPU Fast (`--device cpu-fast` ou `LAMBDA_TORCH_MKLDNN=1`)** :
   * Active les optimisations MKL-DNN / oneDNN et le multi-threading CPU sur les machines physiques compatibles.

Le matériel actif est affiché en direct dans le HUD de la fenêtre Pygame (ex: `dev CUDA (NVIDIA GeForce RTX 2060)` ou `dev CPU (Safe/Portable)`).

---

## 4. Contrôles & Raccourcis clavier (GUI)

Lorsque la fenêtre de simulation est ouverte :

| Touche / Action | Fonction |
|---|---|
| **`ESPACE`** ou **`B`** | Basculer entre le mode **Manuel** (*brain off*) et le mode **Autonome** (*brain on / auto*). |
| **`S`** | Déclencher une phase de **Sommeil** (*sleep cycle*) : entraîne le World Model (128 époques) puis la Policy (32 étapes) sur l'expérience accumulée. |
| **`R`** | **Reset** : replace le squelette à sa position de spawn, réinitialise les filtres et vide le tampon d'action. |
| **`G`** | Afficher / masquer les cibles visuelles des actuateurs (*targets*). |
| **`ÉCHAP`** | Quitter l'application. |
| **Clic gauche + glisser sur les joysticks** | Piloter manuellement l'orientation et l'extension des membres (G: gauche, D: droite) ou de la queue en mode manuel. |
| **Clic droit + glisser sur une plateforme** | Déplacer une plateforme dans le monde physique. |
| **Souris** | Déplacer le curseur dans l'environnement (perçu par l'animat via l'extéroception et le toucher). |

---

## 5. Architecture & Cadences temporelles

```
                 [ Simulation Physique 60 Hz ]
                Pymunk 2D, Contacts, Forces
                             │
                             ▼ (60 Hz)
                      [ Smoother IIR ]
          Lissage temporel des signaux sensoriels
                             │
            ┌────────────────┴────────────────┐
            ▼ (3 Hz)                          ▼ (6 Hz)
    [ World Model ]                     [ Policy ]
 Transformer autorégressif            MLP Acteur
  Prédit les salves 16-tokens      Consignes motrices (5)
  (13 états + 3 actions)          (Pattes avant/arrière, queue)
```

* **Physique & Capteurs (60 Hz)** : Squelette articulé Pymunk, frottements sol, proprioception, extéroception et intéroception.
* **World Model (3 Hz)** : Tokenisation dense de 16 tokens par salve (25 dimensions/token). Modélisation causale et imagination autorégressive complète (états + actions) sur un horizon futur de 2 secondes.
* **Policy (6 Hz)** : Sélectionne l'action motrice à partir des 47 scalaires sensoriels et de l'état latent extrait du World Model.
* **Phase de Sommeil (`S`)** : Entraînement hors-ligne simultané du World Model (prédiction auto-supervisée du futur) et de la Policy (tournoi de candidats guidé par l'anticipation du World Model).

---

## 6. Tests

Pour lancer la suite de tests automatisée :

```bash
SDL_VIDEODRIVER=dummy PYTHONPATH=. .venv/bin/pytest
```
