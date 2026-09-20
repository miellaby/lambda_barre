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
