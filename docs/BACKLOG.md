# Backlog & Spécifications Futures — λ̄ (Lambda barre)

Ce document répertorie les chantiers et orientations de conception actés pour les développements futurs de λ̄.

---

## 1. Mode « Exploratoire » dans l'IHM (Remplacement du dataset généré offline)

### Rationale & Philosophie
Le dataset synthétique généré hors-ligne (`dataset.py` / `--bootstrap-dataset`) est **retiré**. Cette auto-génération déconnectait l'apprentissage de la boucle vivante, du rendu visuel et de l'incarnation physique réelle de la créature.

Le **mode exploratoire dans la GUI remplace intégralement ce dataset généré** :
- Il permet de construire le buffer d'expérience directement depuis l'environnement réel et interactif.
- L'utilisateur peut **voir visuellement le buffer se remplir**.
- Il observe en direct les comportements physiques de la créature et la pertinence des transitions enregistrées.
- Il garde la main à tout moment (reprise de contrôle manuel, ajustement de la vitesse de simulation, pause, inspection).

Ce mode exploratoire interactif est une **solution transitoire** avant la mise en œuvre d'une véritable motivation intrinsèque (*curiosité / mode jeu*) où l'animal explorera l'espace des états-actions de manière autonome.

### Spécifications fonctionnelles

1. **Activation IHM :**
   - Raccourci clavier (ex. touche `E`) pour basculer entre mode `[manual]`, `[BRAIN]` et `[EXPLORE]`.
   - Indicateur visuel explicite dans le HUD.
   - Synchronisation visuelle des joysticks (`controls.sync(skel)`) pour que l'utilisateur visualise en direct les consignes motrices générées par l'explorateur.

2. **Régimes d'exploration dynamiques (*Motor Babbling*) :**
   L'explorateur alterne cycliquement entre plusieurs régimes conçus pour balayer l'espace d'action et faire découvrir au World Model la causalité physique $a_t \to s_{t+1}$ :
   - **Posture neutre dynamique :** Maintien de la pose d'équilibre avec micro-variations naturelles.
   - **Transfert de charge & balancement (*Sway*) :** Déplacement du centre de masse d'une patte à l'autre via les angles $\theta$ et extensions $d$.
   - **Variations motrices actives :** Flexions, extensions asymétriques des membres, poussées coordonnées.
   - **Stabilisation par la queue :** Mouvements de balancier pour observer l'effet de contre-couple.
   - **Rétablissement sur perturbations douces :** Petites impulsions pour enrichir le buffer en transitions de récupération.

3. **Détection de blocage (*Planté*) & Auto-reset propre :**
   - **Critères :** Angle du tronc $> 60^\circ$ ($\approx 1.05\text{ rad}$) ou torse au sol immobile pendant plusieurs pas sans possibilité de redressement.
   - **Actions lors du reset :**
     1. `brain.boundary()` : scelle immédiatement le segment d'expérience accumulé jusque-là. Cela garantit qu'aucune transition ne franchit la discontinuité du reset et que l'expérience physique valide précédant la chute est préservée.
     2. `B.reset(skel)` : réinitialise le squelette à la pose spawn.
     3. Réinitialisation des capteurs et du smoother sans interruption de la session.
     4. Reprise fluide de l'exploration dans un nouveau segment.

---

## 2. Comportement Instinctif / Curiosité (« Mode Jeu »)

À terme, le mode exploratoire procédural cédera la place à une pulsion d'exploration intrinsèque :
- Récompense interne basée sur l'erreur de prédiction ou la nouveauté du World Model (type *Curiosity-driven exploration / Random Network Distillation*).
- Pulsion de « jeu » lorsque le niveau de souffrance/fatigue est bas et le confort assuré : l'animal teste de nouveaux mouvements et découvre par lui-même ses limites de stabilité.

---

## 3. Externaliser les nombres magiques vers `config.toml`

> **Statut : ouvert** — `wm_impact_floor` externalisé (plancher additif de la pondération d'impact, contraste ~3x). Le reste de la liste est à implémenter au fil de l'eau.

Tout réglage actuellement câblé en dur et candidat à une exposition dans `config.toml`. Pattern à suivre (établi avec `eos_force_*` puis `wm_impact_floor`) : constante par défaut au niveau du module, paramètre du constructeur `Brain` (ou de `run()`), mapping `TOML_TO_PARAM` dans `config.py`, entrée dans `config.example.toml` et `config.toml`.

### `brain.py` — entraînement & sommeil

- `_WM_REWARD_BOOST = 16.0` (`brain.py:146`) — boost du gradient sur les slots récompense/EOS.
- `_WM_LOOKAHEAD_PERIOD = 4` (`brain.py:147`) — alternance causal:lookahead des passes WM (3:1).
- `_EXPLORE_MARGIN_PCT = 0.001` (`brain.py:143`) — marge d'acceptation des candidats alternatifs (fallback).
- `_WM_SALIENCY_MAX_LOSS = 0.02` (`brain.py:144`) — seuil de confiance de la distance pondérée par saliency EOS.
- `_EOS_DELTA_SCALE = 0.15` (`brain.py:151`) — échelle de normalisation du delta EOS.
- `compute_context_time_weights(decay=0.5)` (`brain.py:79`) — décroissance de la pondération de la fenêtre de décision.
- `prune_coreset(pct=0.33, beta=2.0)` (`brain.py:525`) — oubli actif : fraction du Coreset remise en jeu, poids de l'échantillonnage stochastique.
- `consolidate_addendum_into_coreset(eps=0.04, decay=0.01)` (`brain.py:627`) — rayon ε-net de déduplication croisée et décroissance de vivacité.
- `sleep(surprise_factor=1.0)` (`brain.py:2245`) — seuil de filtrage par surprise : `max(loss_coreset × facteur, 1e-4)`.
- `self._kalman_max_v = [0.20, 0.15, 0.20, 0.15, 0.20]` (`brain.py:961`) — clamping de vitesse de l'extrapolation Kalman.
- `sigma=0.30` (`brain.py:2200`) — écart-type du candidat bruité (fallback candidats).
- `self._time_alpha = 0.1` (`brain.py:983`) — EMA des temps d'inférence (HUD).
- `self._static_threshold = 6`, `self._static_tol = 1e-4` (`brain.py:988-989`) — détection de statisme (pause d'enregistrement).

### `proprio.py` — échelles des coûts innés

Normalisation et constantes de temps des six signaux de coût biologique :

- `EFFORT_SCALE = 6000.0`, `DOULEUR_SEUIL_COLLISION = 800.0`, `DOULEUR_SEUIL_ACCEL = 1500.0`, `DOULEUR_SCALE = 6000.0`, `DOULEUR_DECAY_TAU = 0.15` s (`proprio.py:24-28`).
- `COURBATURE_WINDOW = 1.0` s, `COURBATURE_SEUIL_VAR = 5.0`, `COURBATURE_ACCRUE = 0.005`, `COURBATURE_DECAY = 0.80` (`proprio.py:29-32`).
- `INSTABILITE_K = 0.01`, `INSTABILITE_SCALE = 1.0` (`proprio.py:33-34`).
- `VERTIGE_SCALE = 5.0` (`proprio.py:35`).
- `CONFORT_K = 0.3`, `CONFORT_DEG = 10°` (`proprio.py:36-37`).

### `intero.py` — constantes de temps

- `FATIGUE_RISE`, `FATIGUE_FALL` (demi-vie 100 s), `SOUFFRANCE_RISE`, `SOUFFRANCE_FALL` (demi-vie 240 s) (`intero.py:19-22`).

### `smoother.py`

- `TAU ≈ 0.143 s` (`smoother.py:28`) — constante de temps du filtre IIR.

### Hors périmètre (à garder en dur)

- `models.py` : `LIMB_MIN`, `LIMB_MAX`, `THETA_RANGE` — plages physiques des actionneurs.
- `main.py` : `SPEED_PRESETS` — liste de vitesses, pas un scalaire de réglage.
- Poids de canaux innés `[1, 4, 2, 2, 3, 1]` — constante de design du circuit de récompense, pas un hyperparamètre d'entraînement.

### Note d'implémentation

Les constantes de `brain.py` suivent le pattern `Brain.__init__` → `config.toml`. Celles de `proprio.py` / `intero.py` / `smoother.py` sont des constantes de module consommées à la construction des capteurs : les externaliser demandera de passer les valeurs aux constructeurs (ou un chargement config dans ces modules) — à prévoir dans des sections `[costs]` / `[interoception]` / `[smoother]` de `config.toml`.

### Passe du 2026-11-09 (inventaire des constantes encore en dur)

- `[costs]` : toutes les constantes de proprio.py (EFFORT_SCALE, DOULEUR_SEUIL_COLLISION, DOULEUR_SEUIL_ACCEL, DOULEUR_SCALE, DOULEUR_DECAY_TAU, COURBATURE_*, INSTABILITE_*, VERTIGE_SCALE, CONFORT_K, CONFORT_DEG, CONFORT_DOULEUR_MAX).
- `[interoception]` : FATIGUE_RISE/FALL, SOUFFRANCE_RISE/FALL (intero.py).
- `[contact]` : CONTACT_REF_LEN (extero.py).
- `[vision]` : VISION_* (extero.py).
- `[world]` : GRAVITY, BALL_RADIUS/MASS/SPAWN.
- `[smoother]` : TAU.
- Hors périmètre : morphologie de body.py (TORSO_VERTS, masses, ressorts) et constantes de rendu.

---

## 4. Métrique de santé : distance au geste démontré

> **Statut : acté, non planifié** (déplacé depuis `PROPAL.md`, 2026-11-02).

**Problème.** Aucune mesure ne dit si le WM a intériorisé les démonstrations de relevage : l'inversion P(a4 | contexte, EOS favorable) est-elle apprise ?

**Spécification.** À chaque sommeil, sur les fenêtres du Coreset à relief > 0,05 : générer l'action hindsight sur le contexte s0..s4 avec l'EOS réalisé favorable, et logguer ‖a4_généré − a4_démo‖ moyen — dans les stats de cycle (`sleep_cycle_stats` + résumé console), éventuellement HUD. Cible ~0,05 (mesuré à ~0,20 sur les checkpoints de novembre 2026).

**Usage.** C'est l'indicateur qui pilote les autres leviers du forçage (ε, fréquence lookahead) : si la distance descend, le WM sait refaire le geste ; si elle stagne malgré les leviers, le problème est en amont (couverture des données, pas de conditionnement).

---

## 5. Classification des séquences et déterminisme action→dénouement

> **Statut : acté, non implémenté** (2026-11-10). Table exhaustive des cas : `classification_sequences.csv` (96 combinaisons, critères décrits en tête du fichier).

**Classification.** Six axes indépendants par séquence ; les ensembles se définissent par croisement (ex. « améliorable ET déterministe ») :

- `determinisme` — somme de deux scores (formule ci-dessous), « oui » si la somme dépasse un seuil à régler expérimentalement.
- `signe_denouement` — `+` / `0` / `−` (relief_score signé).
- `surprenant` — loss > seuil de surprise (`evaluate_sequences_surprise`).
- `nouveau` — dans l'addendum ou dans le coreset depuis peu (compteur individuel d'apprentissage proche de 0).
- `vicace` — vécue récemment ou remémorée et conservée.
- `impact_fort` — |relief| au-dessus du plancher.
- `ameliorable` — dénouement de la séquence sous le seuil moyen du corpus (indépendant du déterminisme).

**Formule du déterminisme** (WM supposé bien entraîné, test par contre-factuel — pas de fouille du corpus) :

- det(action→conséquence) = similarité(dénouement WM sur [contexte + action enregistrée], dénouement enregistré) / dispersion des dénouements prédits par le WM pour les actions déductibles du contexte (action enregistrée, action WM causale, action WM conditionnée EOS, action policy courante).
- det(conséquence→action) = similarité(action WM conditionnée par l'EOS enregistré, action enregistrée) / dispersion des actions générées pour d'autres EOS (ex. EOS neutre 0,5).
- Interprétation : conséquences similaires pour des actions différentes = causalité exogène ; dénouement bien prédit pour l'action enregistrée = déterminisme par l'action.
- **Quand calculer le déterminisme** : une seule fois par cycle de sommeil (pas de recalcul entre deux epochs), et seulement une fois le WM bien entraîné — ex. quand la loss WM atteint un plateau. Avant ce plateau, pas de calcul de déterminisme.

**Rejeté** : définir « même contexte » par clustering/saliency pondérée pour chercher des séquences voisines dans le corpus — la distance pondérée linéaire est un piège sur un problème non-linéaire (une variable peut compter selon la valeur des autres).

**Conclusions par axe** (à traduire en pondérations d'apprentissage et critères d'éviction) :

- non-déterministe → séquence proposée à l'éviction, d'autant plus qu'elle a déjà été souvent entraînée (compteur « nb de batchs WM » par séquence : **n'existe pas, à ajouter**, +1 à chaque batch coreset/addendum où la séquence est présente).
- signe du dénouement : pas un critère — une séquence déterministe s'apprend quel que soit son dénouement.
- surprenant : pour une séquence **ancienne** (compteur de batchs WM >= 1) la surprise basse (loss < surprise_floor) est un signe de déterminisme. Ce n'est pas un critère d'oubli sinon il ne restera que des séquences non-déterministes et donc non apprennable. Pour une séquence **neuve** (compteur de batchs WM = 0), une surprise basse (loss  < `surprise_floor`) signifie qu'on peut oublier la séquence.
- vivacité : à combiner avec le déterminisme pour l'éviction.
- impact : peut moduler le gain d'apprentissage, mais le dénouement neutre s'apprend aussi.
- améliorable : ne pas apprendre de nouvelle action si l'action politique courante est suffisamment bonne — dénouement politique prédit ≥ dénouement enregistré de la séquence, ou ≥ quantile élevé du dénouement moyen du corpus — sinon on désapprend.

---

## 6. Proprioception de l'angle du tronc → exteroception (oreille interne)

> **Statut : acté, non implémenté** (2026-11-12).

L'angle du tronc (équilibre/verticalité) est actuellement classé en proprioception, mais sémantiquement c'est de l'**exteroception** : c'est l'oreille interne (vestibulaire), un capteur de position dans le monde, pas un retour moteur. À reclasser dans le layout des tokens (`tokenize.py`) lors d'un prochain changement de nomenclature.

---

## 7. Séquences « time-event » (symboliques, sans horloge fixe) — 2026-11-15

### Rationale
Les séquences actuelles sont cadencées par tick constant (3 Hz) : une salve de 16 tokens par tick, fenêtres de taille fixe. Or un geste (ex. relevement) dure plusieurs secondes et contient plusieurs extrema de consignes (extension + retour du pied = 2 extrema, parfois 3). Le tick sert surtout de bourrage entre deux moments clés du geste.

### Proposition
- **Unité de séquence = l'extremum de consigne** (déjà enregistré comme extrême local par tick), pas le tick : un token par moment clé du geste, avec le **délai vers la prochaine action en slot de la consigne** — pas d'horloge externe, longueur de séquence variable = complexité du geste (nombre d'extrema).
- **Label « amorçage »** pour le dénouement : amorçage d'un run d'extrema à la décision (n_ctx) + signe conforme au geste futur enregistré → BON ; sinon NEUTRE. (Ne PAS seuiller la valeur absolue des consignes : ce sont des angles/distances.)
- **Filtrage dataset** : garder les pôles (amorçage immédiat à n_ctx / vraiment neutre = aucun run + aucune inflexion de coût), lâcher les séquences à action tardive.

### Impact attendu
Convergence de l'apprentissage WM sur les séquences pertinentes ; le dénouement encode "l'action s'engage maintenant et débouche sur un soulagement".

### Coût
Réécriture de la géométrie : salves de 16 tokens, fenêtres de taille fixe, masque causal, positions de salve, buffer, théâtre. À faire APRÈS avoir validé le label amorçage dans l'architecture actuelle.
