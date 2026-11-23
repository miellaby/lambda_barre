# Extéroception de λ̄

L'extéroception est la perception de l'environnement externe : « que se
passe-t-il autour de moi ? ».

Contrairement à la proprioception (qui mesure l'état du corps), l'extéroception
mesure le monde extérieur. L'animat ne perçoit pas des abstractions informatiques
discrètes (« fenêtre », « bouton », « souris », « clavier ») mais un petit
nombre de valeurs continues, respectant l'égocentricité.

## Égocentricité et facing

Même principe qu'en proprioception : le `facing` est une variable interne de
miroir, jamais perçue. L'animal ignore s'il regarde à gauche ou à droite.

Mais l'extéroception introduit une différence : l'environnement est **fixe dans
le repère monde**, même si le corps de l'animat se déplace et tourne. La
conversion se fait en deux temps :

1. **Translation** : soustraire la position monde de la tête (point
   `HEAD_ANCHOR` projeté dans le monde) pour obtenir des coordonnées
   tête-centrées.
2. **Miroir facing** : multiplier la composante horizontale (x) par `facing`
   pour que « devant la tête » soit toujours positif. La composante verticale
   (y) n'est pas miroirée (gravité).

```python
head_world = torso.local_to_world(HEAD_ANCHOR)
dx_ego = facing * (stimulus_x - head_world.x)
dy_ego = stimulus_y - head_world.y
```

Aucun influence de `torso.angle` : l'animat 2D ne tourne que
selon l'axe gauche/droite (facing). Le miroir facing suffit.

## Vision

La vision est rétinienne, très basse résolution, à courte portée. Elle est en
niveaux de gris (N&B).

### Brightness

Le cône de vision est un triangle dans le prolongement de la tête, pointant
vers l'avant (côté facing). Il est découpé en 16 cellules disposées en 4×4.

Pour chaque cellule, on calcule la brightness moyenne (1 scalaire [0, 1]).

Soit 16 valeurs.

```python
[VISION]
  cellule_1   # brightness [0, 1]
  cellule_2
  ...
  cellule_16
```

Cela suffit pour distinguer une surface uniforme, un bord, un objet, du mouvement.

### Flux optique

Le flux optique est une perception globale (non liée au cône rétinien) qui
détecte le mouvement des solides statiques de la scène (plateformes). À chaque
frame visuelle (6 Hz), on compare les rectangles de chaque plateforme à t-1
et t. La surface qui a changé est la différence symétrique des rectangles :

```
changed = area(old) + area(new) - 2 * area(intersection)
```

Le barycentre de la zone changée est calculé puis égocentré :

- `flux_surface` : somme des surfaces changées (scalaire, sans miroir)
- `flux_x` : `facing * (bx - head_x)` du barycentre, référentiel tête
- `flux_y` : `by - head_y` du barycentre, référentiel tête

Cela restitue un flux optique grossier : « quelque chose de gros bouge à
gauche/droite ».

## Mobile physique (ballon / balle)

L'environnement physique Pymunk intègre un corps mobile dynamique (balle rouge, $r = 16\text{ px}$, masse $0.5\text{ kg}$) soumis à la gravité (`GRAVITY = -900 px/s²`), au rebond (`restitution = 0.75`) et au frottement de roulement, manipulable à la souris (saisie, lancer).

Les signaux du capteur mesurent la position et la vitesse de ce mobile dans le référentiel de la tête :

- `ball_dir` : `atan2(dy, facing * dx)` — angle égocentré vers le ballon (devant = 0, [-π, +π]).
- `ball_prox` : `max_r / (max_r + r)` — proximité du ballon, décroît avec la distance mais s'aplatit (proche = 1, loin = ~0), simulant une résolution fovéale.
- `ball_vr` : `dr/dt` — vitesse radiale du ballon relative à la tête (négatif = rapprochement, positif = éloignement) ; le mouvement propre de l'animat compte (avancer vers un ballon immobile lit un `ball_vr` négatif).
- `ball_va` : `d(dir)/dt` — vitesse angulaire de la ligne de visée (rad/s), différence finie en plus court chemin à travers la couture ±π.

La position est encodée en polaire (direction + proximité) et la vitesse aussi (radiale + angulaire). Le ballon est également visible dans le cône rétinien de la **Vision** (détecté avec une intensité dynamique `DYNAMIC_GRAY = 0.90`) et peut heurter physiquement l'animat (perçu par le capteur de toucher).

```python
[ENV]
  ball      # (dir, prox, vr, va) égocentrés vers le mobile
```

## Toucher

Le tronc et les 2 pattes postérieures sont des capteurs tactiles. Comme pour la
proprioception, les pattes sont échangées selon le facing (avant = patte vers la
tête, arrière = l'autre).

Tout solide étranger à l'animat (sol, plateforme, objet dynamique) est perçu
indifféremment : un solide est un solide.

Sensations tactiles :

- `contact_sol_avant` : force normale du solide sur la patte avant (scalaire,
  impulsion / dt). Sans miroir (composante y).
- `contact_sol_arriere` : idem patte arrière.
- `collision_tronc` : collision du tronc avec un solide. Le tronc étant large,
  le point d'application du contact importe. Produit 4 scalaires :
  - `collision_tronc_x` : `facing * impulse_x / dt` — composante horizontale
    de la force, égocentrée (avant = +).
  - `collision_tronc_y` : `impulse_y / dt` — composante verticale (vers le
    haut = +). Sans miroir (gravité).
  - `collision_tronc_cx` : `facing * local_x` — position du point de contact
    dans le repère du tronc, égocentrée (avant = +).
  - `collision_tronc_cy` : `local_y` — position du point de contact dans le
    repère du tronc (vers le haut = +). Sans miroir.

Les pattes étant petites, le point d'application n'est pas fourni pour elles :
la force normale scalaire suffit.

```python
[TOUCH]
  tronc      # (collision_x, collision_y, contact_x, contact_y) égocentrés
  patte_av   # (contact_sol,)
  patte_ar   # (contact_sol,)
```


## Audition

Le son est perçu par 5 cellules fréquentielles, chacune produisant un scalaire
normalisé sur [0, 1]. Le son est toujours émis depuis la position du mobile (ballon) :
la proximité du mobile (`ball_prox`) module l'intensité perçue.

### Dynamique IIR

Chaque cellule fréquentielle possède un compteur `cps` mis à jour par un filtre
passe-bas à réponse impulsionnelle infinie (IIR) :

- Sur un événement (clic ou touche), `cps += 1` pour les cellules activées.
- À chaque échantillon (30 Hz), `cps *= decay` où `decay = 0.5^(dt / demi_vie)`.
  La demi-vie est de 1 s : le `cps` perd la moitié de sa valeur chaque seconde
  sans événement.

L'intensité perçue d'une cellule est :

```
intensité = min(1, (cps / cps_max) * ball_prox)
```

où `ball_prox = max_r / (max_r + distance_tete_ballon)` (proche = 1,
loin = ~0). Ainsi un événement proche du ballon est fort, un événement lointain est faible, et la
décroissance temporelle est gérée par le filtre IIR — pas d'enveloppe séparée.


### Spectre fréquentiel

Les 5 cellules forment un spectre à 5 bandes. Chaque événement active un
masque de 5 bits déterminant quelles cellules sont incrémentées :

- **Clic souris** : masque = `0b00001` (cellule 0 seule).
- **Touche clavier** : masque = code de Gray de l'index de la touche dans un
  parcours boustrophédon du clavier physique (voir ci-dessous).

L'encodage boustrophédon + code de Gray garantit que deux touches voisines sur
le clavier ne diffèrent que d'un seul bit, donc d'une seule cellule
fréquentielle.

### Parcours boustrophédon du clavier

Le clavier physique est parcouru en serpentine. Les keycodes pygame
(`ev.key`) dépendent du layout (AZERTY, QWERTY...), on utilise donc
`ev.scancode` (code physique SDL, indépendant du layout) pour identifier
la position de la touche :

```
Rangée 0 (gauche→droite) : q w e r t y u i o p
Rangée 1 (droite→gauche) : l k j h g f d s a
Rangée 2 (gauche→droite) : z x c v b n m
Rangée 3 (droite→gauche) : return ... space
```

L'index de chaque touche dans ce parcours (1 à 28) est encodé en code de Gray
sur 5 bits. L'index 0 = silence (aucune touche). Espace = index 28, entrée =
index 27.

```python
[ENV]
  ball      # (dir, prox, vr, va, son_0, son_1, son_2, son_3, son_4)
```

L'audition de sa propre voix (vocalises) est un signal proprioceptif, pas
extéroceptif — c'est la perception d'un actuateur interne, pas d'un événement
externe.

## Cadencement

Les modalités extéroceptives sont plus lentes que la proprioception :

- **Vision** : cadencée à ~6 Hz (10× plus lent que la proprioception à 60 Hz).
  Le cône visuel est un instantané de l'écran, pas besoin de 60 Hz.
- **Flux optique** : dérivé de deux frames visuelles consécutives, donc ~6 Hz.
- **Mobile (ballon) + Son** : 30 Hz (intermédiaire — le mouvement du ballon est rapide mais
  pas besoin de 60 Hz). Le son (clic + clavier) est inclus dans ce canal car
  la source sonore est la position du mobile.
- **Toucher** : 60 Hz (contact calculé via `CollisionHandler` sur le pas
  physique).

## Format de sortie

Chaque modalité produit un `dict` ou une liste de `dict`, assemblés par
l'encodeur du transformer en séquences de tokens denses (voir `tokens.md`).

## Implémentation — état actuel

Implémentées :
- **Mobile (ballon) + Son** : position polaire (direction + proximité), vitesse
  cartésienne, et 5 cellules fréquentielles IIR pour le son (clic souris +
  clavier). Le son est émis depuis la position du mobile — `extero.py` classe
  `Cursor`. Cadencée à 30 Hz.

- **Vision** : cône rétinien 4×4, 16 cellules × 1 brightness (N&B) = 16
  valeurs, échantillonnage par `space.point_query_nearest` — `extero.py`
  classe `Vision`. Cadencée à 6 Hz. Pas de masquage (chaque cellule est
  indépendante).
- **Flux optique** : différence symétrique des rectangles de plateformes entre
  deux frames visuelles (6 Hz). Surface changée + barycentre égocentré —
  `extero.py` classe `Vision`, signaux `flux_surface`, `flux_x`, `flux_y`.
- **Toucher** : forces de contact des pattes (2 scalaires) et collision du
  tronc (force + point de contact dans le repère du tronc, 4 scalaires) via
  `CollisionHandler` `post_solve` — `extero.py` classe `Touch`. Cadencée à
