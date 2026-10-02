# Backlog & Spécifications Futures — λ̄ (Lambda barre)

Ce document répertorie les chantiers et orientations de conception actés pour les développements futurs de λ̄.

---

## 1. Supervision efficiente de la politique : le WM en mode « EOS-lookahead »

> **Statut : implémenté** (2026-10-01, voir `session_2026_10_01.md`). Détail d'implémentation par rapport à la spécification : la ligne du token EOS du masque lookahead est restreinte au contexte de décision (et à elle-même) — sans cette restriction, la clé EOS laissait fuiter l'action enregistrée vers la requête de décision en couche ≥ 2, et la génération compacte n'était pas exactement équivalente à l'entraînement.

### Rationale & Philosophie

La prédiction d'action par le WM sous MSE produit **l'action moyenne** : pour un contexte identique, le MSE mélange toutes les intentions en une action tiède à demi-amplitude que personne n'a jamais prise. Le problème n'est pas la tête mais le **conditionnement** : $P(a \mid \text{contexte})$ est multimodale dès que la politique est stochastique.

La solution est de **conditionner par le dénouement** (*hindsight*) : $P(a_4 \mid \text{contexte}, EOS)$. Forcer un EOS favorable filtre la multimodalité — l'action générée est celle *associée au soulagement* dans des situations semblables, pas la moyenne de tout ce qui a été tenté. Le WM n'invente rien : il réinterroge l'expérience vécue (« dans des situations pareilles, qu'est-ce qui a effectivement soulagé ? »).

Ce chantier remplace la machinerie actuelle de `train_policy` — 4 candidats (baseline, démonstration, bruit faible, bruit fort) × 3 régimes (Kalman, WM autoregressif, Policy closed-loop) = **12 rollouts imaginés par échantillon** — par **3 forwards**. Sur CPU sans accélération, c'est le dividende principal. Le régime 2 (WM autoregressif, action moyenne) devient obsolète d'un coup.

### Spécifications techniques

1. **Masque d'attention « EOS-lookahead » :**
   - Variante du masque causal où la position qui prédit $a_4$ (dernier token d'état de $S_4$, position 76) peut attendre la position du token EOS terminal (172).
   - Entraînement **alterné** avec le masque causal classique sur le même tronc (précédent : prefix-LM, UL2, GLM). Ratio à régler (ex. 3:1 causal:lookahead).
   - **Pré-requis code** : `WorldModel.forward` passe `is_causal=True` en dur — à rendre conditionnel, sinon le chemin rapide causal ignore silencieusement le masque lookahead.
   - L'encodage positionnel sinusoidal par index de salve existe déjà : le modèle sait où il est indépendamment de la causalité.

2. **Entraînement du mode lookahead :**
   - Aucun forçage : chaque séquence du Coreset/Addendum porte son EOS **réalisé**. La paire (contexte $S_0 \dots S_4$, EOS réel $\to$ $a_4$ réel) s'apprend sur toutes les données, bonnes et mauvaises issues (les mauvaises enseignent la frontière).
   - Passes alternées dans `wm_coreset` / `wm_addendum`, loss identique (masquée, boostée EOS/récompense, pondérée par impact).

3. **Génération (supervision de la politique) — pipeline par échantillon :**
   - Rollout **baseline** avec l'action de la politique $\pi$ depuis le contexte → EOS prédit $\widehat{EOS}_{\pi}$ + coût accumulé $C_{\pi}$ (même $\gamma = 1.1$ que le Bellman actuel).
   - **Forçage gradué relatif** : $EOS_{\text{forcé}} = \widehat{EOS}_{\pi} - \varepsilon$, avec $\varepsilon$ tiré d'une petite distribution fixe dans le sens favorable. Pas de scan du Coreset (non scalable) : l'ancre contextuelle vient du modèle lui-même. Curriculum automatique — les cibles se resserrent à mesure que la politique s'améliore.
   - **Génération** de $a_4$ par le mode lookahead sous $EOS_{\text{forcé}}$.
   - **Vérification forward** : rollout causal de 6 pas avec $a_4$ → $C_{\text{gen}}$.
   - **Acceptation purement comparative** (le critère décisif, pas « proche de l'EOS forcé ») : si $C_{\text{gen}} < (1 - 0.001) \cdot C_{\pi}$ (marge de stabilité existante `_EXPLORE_MARGIN_PCT`) → $a_4$ devient **LA sortie supervisée** de la politique ($\mathcal{L} = \text{MSE}(\pi, a_4)$) ; sinon → pas de mise à jour (ou cible = baseline).
   - La vérification forward cumule trois rôles : test de **causalité** (l'action générée cause-t-elle le soulagement ?), test d'**atteignabilité** (subsume tout test de support), et **découverte gracieuse du dépassement** (mieux que demandé = gardé).

4. **Ce qui ne change pas :**
   - Réveil, latent, KV-cache, imagination : **mode causal uniquement**. Le masque est un argument d'appel, le wiring du réveil est intact.
   - Évaluation de surprise, filtrage, consolidation : inchangés.

### Scalabilité

- **O(1)** en taille de dataset et en longueur de séquence : aucun scan de voisins, aucune lecture des séquences à la génération.
- Complément optionnel : quantiles globaux d'EOS maintenus en streaming (mise à jour O(1) à la consolidation) pour calibrer l'échelle de $\varepsilon$.

### Risques & monitorage

- **Capacité** : d_model 64, 3 couches — les deux objectifs se disputent le tronc. Si `last_wm_coreset_loss` stagne ou remonte, baisser le ratio lookahead.
- **Interférence sur le latent** : le latent du réveil est produit causalement mais façonné aussi par les gradients lookahead — surveiller le comportement de la politique après sommeil.
- **Perte lookahead stratifiée par quantile d'EOS** : si les requêtes favorables échouent nettement plus que les médianes, on interroge hors distribution.
- **Fallback** : si la génération échoue (rejet systématique), retomber sur les candidats stochastiques actuels — garder le code en survivance derrière un drapeau.

---

## 2. Mode « Exploratoire » dans l'IHM (Remplacement du dataset généré offline)

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

## 3. Comportement Instinctif / Curiosité (« Mode Jeu »)

À terme, le mode exploratoire procédural cédera la place à une pulsion d'exploration intrinsèque :
- Récompense interne basée sur l'erreur de prédiction ou la nouveauté du World Model (type *Curiosity-driven exploration / Random Network Distillation*).
- Pulsion de « jeu » lorsque le niveau de souffrance/fatigue est bas et le confort assuré : l'animal teste de nouveaux mouvements et découvre par lui-même ses limites de stabilité.
