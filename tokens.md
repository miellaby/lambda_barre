# Spécification et structure des tokens de $\bar{\lambda}$

Les échanges entre le corps simulé de l'animat, le World Model et la Politique reposent sur un flux de **tokens denses** de dimension fixe, générés par `DenseEncoder` dans `tokenize.py`.

---

## Structure d'un token dense

Chaque token est un vecteur de **25 flottants**, produit directement par l'encodeur sans couche d'embedding apprise :

| Section | Dimensions | Rôle et contenu |
| :--- | :--- | :--- |
| **Type** | 1 | `0.0` = État / Sensoriel, `1.0` = Action / Moteur |
| **Modalité** | 4 | Vecteur d'activation de la modalité sensorielle ou motrice |
| **Canal** | 4 | Vecteur orthogonal identifiant le sous-système au sein de la modalité |
| **Signaux** | 16 | Scalaires normalisés dans $[0.0, 1.0]$. Remplissage par zéros (*zero-padding*) si $< 16$ |

### Codes de modalité (4D)

* **Action** (`_M_ACTION`) : `(0.0, 0.0, 0.0, 0.0)`
* **Intéroception** (`_M_INTERO`) : `(1.0, 0.0, 0.0, 0.0)`
* **Somatosensoriel** (`_M_SOMATO`) : `(0.0, 1.0, 0.0, 0.0)`
* **Visuel & Extéroception** (`_M_VISUAL`) : `(0.0, 0.0, 1.0, 0.0)`
* **Motivationnel & Récompense** (`_M_MOTIV`) : `(1.0, 0.0, 0.0, 1.0)`

---

## Salve temporelle (3 Hz)

Une **salve** regroupe l'ensemble des 16 tokens ($S_i + A_i$) produits à chaque tick du World Model (fréquence 3 Hz, soit $\Delta t = 0.33\text{ s}$) :

* **13 tokens d'état** ($S_0 \dots S_{12}$) : 53 signaux scalaires actifs.
* **3 tokens d'action** ($A_{13} \dots A_{15}$) : 5 consignes motrices physiques.

---

## Répertoire des 16 tokens

### Tokens sensoriels et d'état (Tokens 0 à 12)

| Indice | Modalité | Canal | Signaux contenus | Nb signaux |
| :--- | :--- | :--- | :--- | :--- |
| **0** | Somatosensoriel `(0,1,0,0)` | Feedback `(0,0,0,1)` | `force_actuateur_avant`, `force_actuateur_arriere`, `couple_queue` | 3 |
| **1** | Somatosensoriel `(0,1,0,0)` | Posture `(0,0,1,0)` | `tronc_angle`, `queue_angle`, `accel_tete_avant`, `accel_tete_haut` | 4 |
| **2** | Somatosensoriel `(0,1,0,0)` | Membre avant `(0,1,0,0)` | `membre_angle_avant`, `membre_distance_avant` | 2 |
| **3** | Somatosensoriel `(0,1,0,0)` | Membre arrière `(1,0,0,0)` | `membre_angle_arriere`, `membre_distance_arriere` | 2 |
| **4** | Visuel `(0,0,1,0)` | Cône rétinien `(1,1,1,1)` | `vis_c1` à `vis_c16` (16 cellules du champ de vision) | 16 |
| **5** | Visuel / Extéro `(0,0,1,0)` | Mobile physique `(0,0,0,1)` | `curseur_dir`, `curseur_prox`, `curseur_vx`, `curseur_vy` | 4 |
| **6** | Visuel `(0,0,1,0)` | Flux optique `(0,0,1,0)` | `flux_surface`, `flux_x`, `flux_y` | 3 |
| **7** | Visuel / Audio `(0,0,1,0)` | Son `(0,1,0,0)` | `son_0` à `son_4` (5 canaux fréquentiels) | 5 |
| **8** | Somatosensoriel `(0,1,0,0)` | Contact sol `(0,0,0,1)` | `contact_sol_avant`, `contact_sol_arriere` | 2 |
| **9** | Somatosensoriel `(0,1,0,0)` | Collision tronc `(0,0,1,0)` | `collision_tronc_x`, `collision_tronc_y`, `collision_tronc_cx`, `collision_tronc_cy` | 4 |
| **10** | Motivationnel `(1,0,0,1)` | Coûts innés `(0,0,0,0)` | `effort`, `douleur`, `courbature`, `instabilite`, `vertige` | 5 |
| **11** | Motivationnel `(1,0,0,1)` | Récompense `(1,0,0,0)` | `confort` | 1 |
| **12** | Intéroception `(1,0,0,0)` | Bilan métabolique `(1,1,1,1)` | `fatigue`, `souffrance` (token terminal EOS) | 2 |

### Tokens moteurs et d'action (Tokens 13 à 15)

| Indice | Modalité | Canal | Consignes motrices | Nb signaux |
| :--- | :--- | :--- | :--- | :--- |
| **13** | Action `(0,0,0,0)` | Patte avant `(0,0,0,1)` | `membre_avant_theta`, `membre_avant_d` | 2 |
| **14** | Action `(0,0,0,0)` | Patte arrière `(0,0,1,0)` | `membre_arriere_theta`, `membre_arriere_d` | 2 |
| **15** | Action `(0,0,0,0)` | Queue `(0,1,0,0)` | `queue_theta` | 1 |

---

## Règles d'encodage et cas particuliers

### Masquage et zéros de remplissage (*Padding slots*)
Sur les 16 dimensions de signaux d'un token, seules les $N$ premières contiennent des valeurs physiques. Les slots $N \dots 15$ sont maintenus à $0.0$ et sont exclus du calcul de perte MSE via le masque binaire `target_valid`.

### Token 12 (Intéroception & Terminal EOS)
* **Dans l'`ExperienceBuffer`** : conserve les valeurs absolues réelles enregistrées à chaque pas ($\text{fatigue}, \text{souffrance} \in [0.0, 1.0]$).
* **Dans les tenseurs d'entraînement du World Model (`_batch_tensors`)** :
  * Les salves intermédiaires ($s_0 \dots s_9$) voient leur token 12 forcé à zéro.
  * La salve terminale ($s_{10}$, fin de séquence) porte la variation métabolique globale normalisée :
    $$d_{\text{fatigue}} = \frac{\text{fatigue}_{10} - \text{fatigue}_0 + 1.0}{2.0}, \quad d_{\text{souffrance}} = \frac{\text{souffrance}_{10} - \text{souffrance}_0 + 1.0}{2.0}$$
* **Pour la Politique (`sample_for_policy`)** : le delta net $(s_{10} - s_0)$ sert de score de soulagement (*relief score*) pour prioriser l'échantillonnage de l'apprentissage sur les trajectoires de rétablissement.

---

## Intégration dans les modèles

### World Model (`WorldModel`)
* **Entrée** : projection linéaire $25 \to d_{\text{model}}$ ($d_{\text{model}} = 64$).
* **Encodage positionnel** : sinusoïdal non-appris indexé par salve (tous les 16 tokens d'une salve partagent le même encodage temporel).
* **Séquence** : 10 transitions, soit 11 salves consécutives ($10 \times 16 + 13 = 173$ tokens, les actions terminales après $s_{10}$ n'étant pas prédites).
* **Sortie** : tête de régression linéaire `nn.Linear(64, 16)` prédisant les 16 slots de signaux du token suivant sous masque d'attention causale par salve dans le type.

### Politique (`Policy`)
* **Cadence** : intervient à **6 Hz** (`POL_DT = 1/6 s`, toutes les 10 frames de physique) pour un contrôle moteur réactif, plus rapide que le World Model (3 Hz).
* **Entrée** : 45 scalaires sensoriels (tokens d'état non-récompense 0 à 9) concaténés au vecteur latent intermédiaire de 64 dimensions du World Model, soit **109 dimensions**.
* **Architecture** : MLP ($109 \to 256 \to 256 \to 256 \to 5$ avec activations internes `ReLU`).
* **Sortie** : 5 consignes motrices squashées par `tanh` (angles) et `sigmoid` (extensions).

