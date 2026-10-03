# Encodeur de tokens (`DenseEncoder`) et lissage (`Smoother`)

L'encodeur (`tokenize.DenseEncoder`) convertit l'ensemble des signaux physiques, sensoriels et moteurs de l'animat en une salve de tokens denses directement assimilables par le World Model et la Politique, sans couche d'embedding apprise ni discrétisation.

En amont de l'encodeur, le module de lissage temporel (`smoother.Smoother`) agrège et filtre les données de simulation pour assurer la synthèse perceptuelle entre la physique haute fréquence et la cadence cognitive.

---

## Représentation continue

* Chaque signal est normalisé sous forme de scalaire flottant continu dans l'intervalle $[0.0, 1.0]$.
* Les scalaires sont injectés directement dans les slots de signaux du token dense correspondant (vecteur de 25 flottants).


Pour le détail complet des 16 tokens et de leurs canaux, voir [tokens.md](tokens.md).

---

## Prétraitement et lissage temporel (`Smoother`)

Le `Smoother` opère la **compression temporelle** entre la dynamique physique (60 Hz) et le rythme d'observation du cerveau (3 Hz et 6 Hz) :

* **Filtre IIR / EMA** : filtre passe-bas exponentiel du 1er ordre actualisé à chaque pas physique (60 Hz) avec une constante de temps $\tau = 0.33 / \ln(10) \approx 0.1433\text{ s}$ :
  * $90\%$ de la valeur cible est atteinte en exactement $0.33\text{ s}$ (1 tick WM à 3 Hz).
  * Demi-vie d'environ $100\text{ ms}$ (~6 frames physiques), offrant un compromis idéal entre réduction du bruit impulsionnel et réactivité motrice.
* **Filtrage sélectif par signal (*per-scalar*)** :
  * **Signaux lissés** : grandeurs dérivées bruitées (différences finies, chocs et pics instantanés) :
    * Forces et couples (`force_actuateur_avant`, `force_actuateur_arriere`, `couple_queue`).
    * Accélérations de la tête (`accel_tete_avant`, `accel_tete_haut`).
    * Métabolique et puissance (`effort`, `douleur`).
    * Vitesses du mobile (`ball_vr`, `ball_va`).
    * Flux optique (`flux_surface`, `flux_x`, `flux_y`).
  * **Signaux bruts directs (*passthrough* sans lag)** :
    * Posture et géométrie directe (`tronc_angle`, `queue_angle`, `membre_*`).
    * Extéroception spatiale (`ball_dir`, `ball_prox`, cône rétinien `vis_c1..16`).
    * Contacts discrets (`contact_sol_*`, `collision_tronc_*`).
    * Intégrateurs lents internes déjà amortis (`fatigue`, `souffrance`, `courbature`, `vertige`).
* **Réinitialisation instantanée (`reinit`)** : lors d'un basculement de direction du regard (*facing flip*), l'état interne du filtre est immédiatement écrasé par la nouvelle mesure pour éviter toute traînée ou interpolation croisée entre les deux référentiels.

---

## Salve

Une **salve** est la séquence complète des **16 tokens** ($13\text{ état} + 3\text{ action}$, soit $16 \times 25$ flottants) produite à chaque pas de décision du cerveau.

---

## Cadences temporelles

* **Physique Pymunk** : avance à **60 Hz** (découpée en 3 sous-pas de $1/180\text{ s}$ pour la stabilité numérique).
* **Politique (`brain.act`)** : intervient à **6 Hz** (`POL_DT = 1/6 s`, toutes les 10 frames de physique) pour réagir promptement et mettre à jour les consignes motrices $(\theta^*, d^*)$ transmises aux actionneurs.
* **World Model (`brain.record` / `wake_tick`)** : intervient à **3 Hz** (`WM_DT = 1/3 s`, toutes les 20 frames de physique) pour encoder la salve de tokens, enregistrer la mémoire et actualiser la représentation latente.
