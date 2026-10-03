# Le cerveau de λ̄ — l'IA à 2 modèles

Ce document décrit le système de décision apprenant de λ̄ :
un **world model** (transformer autorégressif) et une **politique** (réseau de
renforcement), entraînés en continu.

L'implémentation se compose de `models.py` (les deux réseaux), `brain.py`
(l'orchestrateur `Brain`, la mémoire à deux étages `ExperienceBuffer` et le
pipeline de sommeil), `tokenize.py` (l'encodage dense des salves),
`smoother.py` (le lissage IIR des capteurs), `body.py`/`world.py` (la physique
et les consignes), `main.py` (la boucle interactive), `dream.py` (le Dream
Theater), `wm_theater.py` (la visualisation des rollouts) et `dataset.py` (le
bootstrap de données d'équilibre). Le pipeline complet est documenté dans
`DESIGN.md` ; les détails bas niveau des capteurs dans `exteroception.md`,
`encodeur.md` et `tokens.md`.

---

## Vue d'ensemble

```
                état (salve de 16 tokens)
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
      prédite (s')         (consignes)
             │                 │
             └───────┬─────────┘
                     ▼
                  action ──► environnement simulé
                     │
                     ▼
                nouvel état
```

Pendant l'**éveil**, la physique tourne à 60 Hz. À 3 Hz, chaque tick du world
model produit une **salve** : 16 tokens denses (13 d'état + 3 d'action) qui
résume la dernière seconde d'expérience sensorielle (lissée par IIR). La
salve est journalisée dans la mémoire (`Brain.record`, avec pause
d'enregistrement quand plus rien ne bouge) et un forward pass du world model
rafraîchit le **latent** observé par la politique (`Brain.wake_tick`) — que
le cerveau pilote ou non : c'est le passe perceptif, l'observation de la
politique ne doit jamais être périmée. À 6 Hz (brain on), la politique
échantillonne 5 consignes à partir des scalaires sensoriels frais et du
latent caché (`Brain.act`) ; les consignes ne rentrent pas directement dans la physique :
des **valeurs effectives** convergent vers elles (50 % de l'écart par période
policy, voir `body.py`), amortissant les changements brutaux.

Pendant le **sommeil** (`Brain.sleep`, déclenché par `S`), tout se passe
hors-ligne et en générateur (l'UI consomme un pas par frame) : optimisation
du dataset (extraction de l'addendum, oubli actif d'un tiers du coreset,
déduplication géométrique), entraînement du world model sur le coreset puis
l'addendum survivant au filtre de surprise, entraînement de la politique par
imitation de l'action rétrospective (*hindsight*) générée par le world model
en mode EOS-lookahead, puis consolidation de l'addendum dans le coreset et
sauvegarde des checkpoints. Le Dream Theater (`dream.py`) rejoue les
trajectoires imaginées du dernier cycle.

---

## Le world model (`models.py` — `WorldModel`)

Un transformer autorégressif sur le flux de tokens denses qui apprend
P(s(t+1) | s(t), a(t)) — en pratique la régression des 16 slots de signaux
du token suivant à chaque position. Entraîné uniquement pendant le sommeil,
sur les séquences du coreset et de l'addendum.

### Tokens

Chaque tick produit une **salve** de 16 tokens.

Chaque token est un vecteur dense (25 floats) :

* un préfixe structurel de 9 floats (1 float binaire état/action,
  4 floats de modalité et 4 floats de canal)
* suivi de 16 slots de signaux normalisés sur [0, 1].

Le layout (quel capteur dans quel token/slot) est fixé par
`tokenize.py` :

* 13 tokens d'état: proprioception, membres, rétine, balle,
flux optique, sons, toucher, coûts, récompense, interoception
* 3 tokens d'action (valeurs des consignes après lissage).

Famille interoceptive — deux canaux orthogonaux au sein de la même
modalité : le **bilan métabolique** (fatigue, souffrance — niveaux absolus,
canal `(0,0,0,1)`) et le **token EOS** (canal `(0,0,1,0)`, construit en batch
dans `brain.py`, jamais enregistré dans les salves). Les niveaux absolus
sont des intégrateurs lents (demi-vie 100 s / 240 s) : la **porteuse** du
dénouement, quasi-statique à l'échelle d'une fenêtre de décision — masquée
de toutes les entrées du WM. Le dénouement lui-même est porté par l'EOS : le
delta des canaux de coût innés entre la fenêtre de conséquence s5..s10
(pondérée à l'inverse du temps restant) et la fenêtre de décision s0..s4 —
normalisé 0.5 = neutre, < 0.5 = rétablissement, > 0.5 = aggravation ; la
hiérarchie innée usuelle s'applique dès que l'EOS est lu comme un scalaire.

Les slots de padding au-delà du nombre de signaux valides
d'un token donné restent à 0 et sont masqués.

### Embedding

Les embeddings ne sont pas appris.

Le token 25-dim est projeté par une couche linéaire vers `d_model` (64),
puis un encodage positionnel **sinusoïdal non appris** est ajouté,
indexé par **numéro de salve** : les 16 tokens d'une même salve
partagent donc la même position: on encode le tick temporel, pas
l'ordre dans la salve déjà identifié par le préfixe structurel.

Un même token peut donc être inséré à une position d'entrée arbitraire
et rester à son créneau temporel. C'est notamment utilisé par la
génération rétrospective pour accoler l'EOS terminal en fin de contexte.

### Architecture du modèle

`nn.TransformerEncoder` : d_model 64, 4 têtes, 3 couches, feed-forward 384,
GELU, pre-norm (`norm_first`), dropout 0.

Masque **causal par salve** : tous
les tokens d'une salve voient les salves passées, jamais les suivantes. La
tête est une linéaire d_model → 16 qui prédit les slots de signaux du token
suivant.

Le **latent** lu par la politique est capturé (**hook**) sur la couche du
milieu : l'embedding du dernier token de la dernière forward, une
représentation compressée de l'état courant conditionnée par tout
l'historique.

Pour l'imagination il existe un chemin de **cache KV incrémental**
(`cache_forward`, `predict_next_state_cached`) : projection K/V stockée par
couche, strictement équivalent au recalcul complet, réservé à l'inférence et
cloné à chaque embranchement de trajectoire. Un cache n'est valide que tant
que les poids du world model ne bougent pas.

### Séquence d'entraînement

Une séquence = une trajectoire de `SEQ_STEPS` (10) salves :
`[s0, a0, s1, a1, ..., a9, s10]`, soit 173 tokens. La cible est
la section signaux du token suivant (décalée d'un) : MSE sur les slots
valides seulement.

Une pondération par position/canal renforce le gradient sur les canaux
de récompense et sur l'**EOS terminal** — la position avant-dernière apprend
l'EOS : le delta de dénouement des canaux de coût innés (effort, douleur,
courbature, instabilité, vertige, confort) entre la fenêtre de conséquence
s5..s10 (temps pondéré à l'inverse du temps restant — le long terme domine)
et la fenêtre de décision s0..s4 ; normalisé 0.5 = neutre, < 0.5 =
rétablissement, > 0.5 = aggravation ; hiérarchie innée usuelle pour toute
lecture scalaire.

L'entraînement alterne les passes :

* 3 passes avec masque causal classique,
* **1 passe EOS-lookahead**

En mode _lookahead_, l'action après S4 peut aussi attendre
l'EOS *réalisé* de la séquence : le
world model apprend ainsi P(a4 | contexte, EOS), l'action rétrospective qui
mène au dénouement réel. C'est cette prédiction que la politique imite.

### Méthodes

* `forward(x, attn_mask, salve_positions)` — passe parallèle [B, L, 25] →
  [B, L, 16], prédictions à toutes les positions.
* `predict_next_state(ctx)` — génère autorégressivement les 13 tokens d'état
  suivant (template structurel + slots prédits clampés et masqués, chaque
  token prédit renvoyé dans le contexte).
* `predict_next_action(ctx)` / `predict_next_salve(ctx)` — idem pour les 3
  tokens d'action, ou les deux blocs enchaînés.
* `cache_forward` / `predict_next_state_cached` — variantes à cache KV.
* `last_latent()` — le latent intermédiaire du dernier token (observation
  de la politique).

---

## La politique (`models.py` — `Policy`)

π(a|s) : un petit réseau qui produit les 5 consignes d'actionneurs. Il
n'apprend pas une fonction de valeur : il est entraîné par imitation de
l'action rétrospective du world model (et, en repli, par tournoi de
candidats guidés par les futurs imaginés).

### Entrées

Deux sources d'entrées :

* les **45 scalaires sensoriels** non-récompense de l'état
  (tokens 0..9 : posture, membres, rétine, balle, flux, sons, toucher)
  déjà normalisés [0, 1]
* le **latent du world model** (d_model 64), normalisé par une `LayerNorm`
  pour le mettre à la même échelle que les scalaires.

Le vecteur d'état fait donc 109 dimensions. Les scalaires sont
lus à chaque tick policy (6 Hz) ; le latent est rafraîchi au rythme du
world model (3 Hz) et conservé entre les ticks.

### Architecture

Un MLP : 3 couches cachées de 256 (ReLU), puis une tête linéaire 256 → 5.
L'échantillonnage se fait dans l'espace pré-squash avec une gaussienne
isotrope de std fixe (`act_std` = 0.05 ; `explore_std` = 0.2 pendant
l'exploration en sommeil).

### Sorties

Les 5 consignes squashed dans leurs plages physiques, toujours valides :

* θ des membres et de la queue via `tanh` ([-π, +π]),
* d des membres via `sigmoid` ([LIMB_MIN=10, LIMB_MAX=32] px).

Convention égocentrée avant/ arrière : la correspondance anatomique
gauche/droite se fait dans `main.py` selon le facing.

Les consignes rentrent dans la physique à travers le
lissage par valeurs effectives.

---

## L'orchestrateur

### Cycle de vie

`Brain` (brain.py) porte les deux réseaux, leurs optimiseurs (Adam, lr_wm
3e-4, lr_pol 2e-4, weight decay 1e-4 sur la politique), le `latent_norm`, la
mémoire `ExperienceBuffer` et les masques. Trois checkpoints persistants
sont sauvés à la fin de chaque sommeil :

* `buf_ckpt.pt`: mémoire
* `wm_ckpt.pt`: world model
* `pol_ckpt.pt`: politique

Les discontinuités temporelles (retour au spawn, flip de facing)
passent par `clear_history`/`boundary` : le contexte réveil est vidé,
le segment en cours est scellé, et le smoother est réinitialisé sans
interpolation.

### Éveil

* `record(salve)` (3 Hz) journalise la salve dans les segments linéaires
  de la mémoire ; l'enregistrement se met en pause après 6 ticks
  successifs où animal, environnement et consignes sont immobiles (tolérance
  1e-4), et reprend dès qu'un signal bouge sans vider le contexte.
* `wake_tick(salve)` (3 Hz) forward du world model sur l'historique
  glissant (4 dernières transitions + état courant), rafraîchit le latent
  normalisé et les scalaires cachés.
* `act(salve)` (6 Hz) échantillonne les 5 consignes
  (scalaires frais + latent caché) et les renvoie à la boucle, qui les écrit
  comme consignes dans le squelette ; Ce sont des valeurs lissées
  de ces consignes qui sont entrées dans la physique et journalisées
  (convergence à 50 % de l'écart par période policy).

La policy (act) peut être désactivée pendant l'éveil (*brain off*).
Les consignes sont alors contrôlés par l'utisateur.

### Sommeil

La mémoire `ExperienceBuffer` est à deux étages :

* le **coreset** : mémoire long terme consolidée, dataset effectif du world model,
  soumis à la décroissance de vivacité et l'éviction par capacité (compté en
  séquences)
* l'**addendum** : mémoire court terme contenant les expériences de veille fraîches
  et des expériences passées revisitées pour l'oubli actif.

Le sommeil, `sleep()`, suit alors la séquence suivante.

* **Consolidation du journal** : toutes les fenêtres glissantes de
  `seq_len` (11) salves de la session de veille sont extraites vers
  l'addendum.
* **Oubli actif** : 1/3 du coreset est tiré au sort et remis en jeu dans
  l'addendum comme candidats à l'oubli.
* **Déduplication géométrique** : Les quasi-doublons sont trouvés
  et supprimés d'abord intra-addendum, puis contre le coreset.
* **World model, phase coreset** : entraînement jusqu'au
  quota d'epoch *et* la loss cible (0.01 par défaut). 1 passe
  EOS-lookahead est alternée avec 3 passe à masque causale ; le poids
  d'impact de chaque séquence est saliency-proportionnel (gated par la
  loss coreset : en dessous de 0.02, les distances EOS-saliency sont
  fiables).
* **Filtre de surprise** : la divergence de dynamique et de coût
  de chaque séquence de l'addendum est évaluée par le world model;
  le *connu* est jeté, le *surprenant* est gardé.
* **World model, phase addendum** : entraînement sur l'addendum après
  déduplication et filtrage, même alternance de masque.
* **Politique** : supervision *hindsight* EOS-lookahead : l'action propre
  de la politique est roulée pour obtenir le coût de base C_pi et l'EOS
  imaginé ; l'écart au neutre du dénouement prédit est forcé
  multiplicativement vers le favorable (`EOS_pi − ε·|EOS_pi − 0.5|`,
  ε ~ U(2 %, 10 %) par slot) ; le
  world model en mode lookahead génère l'action associée à ce dénouement ;
  un rollout causal de vérification ne l'accepte que si elle réduit
  réellement le coût imaginé. En repli (rejet systématique pendant
  plusieurs batches, ou horizon d'imagination trop court) : la machinerie
  historique de candidats stochastiques (π, démonstration enregistrée,
  bruit gaussien) reprend la main.
* **Consolidation finale** — l'addendum survivant est committé dans le
  coreset (vivacité rafraîchie, éviction par capacité 1000), le latent de
  réveil est rafraîchi, les checkpoints sont sauvés, et le dernier dream
  record est publié pour le Dream Theater.

---

## Configuration technique

* **Hyperparamètres** :
  gamma 1.1, boost de gradient ×16 sur reward et EOS.
* **Matériel** : Pas de dépendances matérielles fortes, l'accélération CPU/GPU
  s'active sur option.
