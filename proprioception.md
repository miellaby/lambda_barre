# Proprioception de λ̄

La proprioception est la perception de l'état de son propre corps.

Le système proprioceptif de λ̄ réinjecte les mouvements et la posture
en entrée du modèle.

Ces signaux **mesurent** l'état réel produit par les actions
motrices des actuateurs, contrôlés par les **consignes** du modèle,
et les lois de la physique.

## Égocentricité et facing

Extéroception et Proprioception sont calculés selon le principe d'égocentricité.

Tout d'abord, le champ sensoriel se déplace avec la tête de l'animat.

De plus, il dépend du `facing`, l'orientation droite/gauche
de l'animat dans un espace 2D.

- facing = +1 (tête à +x)
- facing = -1 (tête à -x)

"Avant" = vers la tête, toujours positif, quel que soit le facing.

Les scalaires horizontaux (x, angle) sont multipliées par `facing`
pour passer du repère physique au repère égocentré.

Les scalaires verticaux (y) sont définis par la gravité et
invariant au facing gauche/droite.

Les scalaires sans direction (distance, force normale, force de ressort) ne
sont pas miroirés

Les 2 pattes postérieures ne sont pas déplacées lors du changement d'orientation,
Il faut les échanger selon le principe d'égocentricité.

- avant = patte R, arrière = patte L
- avant = patte L, arrière = patte R

Ce n'est pas qu'un changement de label: les valeurs relatives aux pattes
doivent être échangées en entrée du modèle.

## Liste des signaux

Angles:
* `membre_angle_avant` — `facing * theta_torso` mesuré sur la patte avant.
   Angle du pied ; + = vers l'avant.
* `membre_angle_arriere` — `facing * theta_torso` mesuré sur la patte arrière.
* `tronc_angle` — `facing * torso.angle`. Inclinaison du tronc ; + = penché
   vers l'avant (tête).
* `queue_angle` — `(tail.angle - torso.angle) - (-facing * π/2)`. Angle
    relatif à la neutrale facing ; + = queue vers l'avant.
* `couple_queue` — `facing * tail_spring.impulse / substep_dt`. + = couple vers
    l'avant.

Scalaires sans direction:
* `membre_distance_avant` — distance mesurée de la patte avant. Sans miroir
   (scalaire radial).
* `force_actuateur_avant` — `spring.impulse / substep_dt` de la patte avant.
   Effort musculaire. Sans miroir (scalaire de tension).
* `membre_distance_arriere` — distance mesurée de la patte arrière. Sans
   miroir.
* `force_actuateur_arriere` — `spring.impulse / substep_dt` de la patte arrière.
   Sans miroir.

Scalaires horizontaux:
* `accel_tete_avant` — `facing * ax_monde`. Accélération de la tête (point
    `HEAD_ANCHOR`), composante x en repère monde, différence finie. + = vers
    l'avant.

Scalaires verticaux:
* `accel_tete_haut` — `ay_monde`. Composante y en repère monde, différence
    finie. + = vers le haut. Sans miroir (gravité).

## Circuit de renforcement de la proprioception

### L'effort

L'effort est un coût instantané des actuateurs du corps (membres + queue)
déduit de l'activité des ressorts contrôlés:

```python
effort = abs(force_av) + abs(force_ar) + abs(couple_queue)
```

### Douleur

La douleur est un coût instantané quand:

- la collision du tronc dépasse un seuil (`collision_tronc_x` / `_y`), ou
- l'accélération de la tête dépasse un seuil (`accel_tete_avant`, `accel_tete_haut`).

```python
douleur = max(0, |collision_tronc| - seuil) + max(0, |accel_tete| - seuil)
```

### Courbature

La courbature augmente quand la proprioception ne varie
pas suffisament (immobilité prolongée). Elle décroit
rapidement par le mouvement.

On suit la variance des signaux proprio sur une fenêtre glissante.
L'inconfort croît si elle est faible.

Dès qu'un mouvement suffisamment ample revient, il décroît rapidement.

```python
variance_proprio = variance_fenetre(signaux_proprio, fenetre=1.0 s)
inconfort_immobilite += k_immob * max(0, seuil_var - variance_proprio) * dt
inconfort_immobilite *= decay_immob   # décroît dès que variance_proprio > seuil
```

### Instabilité

Un signal qui encourage l'écartement des pattes.
Coût pénalisant quand la projection du centre de masse sort du polygone d'appui.

```python
# pieds en repère monde
left = min(fx_front, fx_back)
right = max(fx_front, fx_back)

# marge : distance du COM projeté au bord le plus proche du support
marge = min(com_x - left, right - com_x)   # > 0 = stable, < 0 = en chute

instabilite = max(0, -marge)   # coût seulement si le COM sort du support
```

•  pieds rapprochés + tronc lourd au-dessus → marge faible → instable
•  pieds écartés → marge grande → stable
•  λ̄ allongé au sol → le tronc est bas mais le COM est toujours dans le support → pas de coût
•  λ̄ penché avec pieds serrés → le COM sort du support → coût

Donc ça pousse à écarter les pattes sans pénaliser la position couchée.

### Confort de la posture

Un renforcement postural. Léger plaisir près d'une des quelques poses instinctives: Debout, couché, ...

`confort_postural` est négatif (récompense) près de la verticale.

Exemple: verticalité du tronc.

```python
confort_postural = k_post * tronc_angle**2
```

... a préciser.

### Vertige

Coût proportionnel à l'accéleration de la tête.

## Calculs concrets

### Angle et distance réels d'un membre (mesure, pas consigne)

Pour chaque patte arrière, on reconstitue (theta, d) à partir de la position
réelle du pied dans le repère du tronc — l'inverse exact du calcul de cible
fait dans `apply_consignes` :

```python
ol = torso.world_to_local(foot.position) - HIP
d = hypot(ol.x, ol.y)
theta_torso = atan2(-ol.x, ol.y)   # convention torso-local, comme theta_star
```

La mesure égocentrée est `theta_ego = facing * theta_torso`. La différence
`(theta_star - theta_torso)` est l'erreur poursuivie par le PD ; la perception
renvoie `theta_ego`, pas l'erreur.

### Queue

L'angle brut relatif au tronc est `tail.angle - torso.angle`. La neutrale
facing-dépendante est `-facing * π/2` (le contrepoids de la queue pointe du
côté opposé à la tête). La mesure égocentrée soustrait cette neutrale :

```python
queue_angle = (tail.angle - torso.angle) - (-facing * pi / 2)
```

Ainsi "queue levée vers l'avant" a le même signe quel que soit le facing.

### Forces des actuateurs

pymunk 7 n'expose pas `spring_force` / `spring_torque` comme propriétés
appelables directement. On utilise `constraint.impulse` (impulsion accumulée
du dernier pas de simulation) divisée par le `substep_dt` pour obtenir une
force/couple moyenne sur le sous-pas.

- Membres : `spring.impulse / substep_dt` est un scalaire (force le long de
  l'axe du ressort, signé selon tension/compression). C'est l'effort
  musculaire instantané. Pas de miroir : la force est une magnitude signée le
  long de l'axe, pas une direction dans le plan.
- Queue : `tail_spring.impulse / substep_dt` (scalaire, couple du ressort
  rotatif). Miroiré par `facing` pour que "vers l'avant" soit positif.

### Force de contact avec le sol

La force de contact avec le sol n'est plus un signal proprioceptif : c'est
une mesure de l'environnement externe (« que touche-je ? »), pas de l'état
du corps. Elle est désormais rangée en extéroception, modalité toucher, sous
les noms `contact_sol_avant` / `contact_sol_arriere` (voir `exteroception.md`
§ Toucher).

### Accélération de la tête

La tête est au point `HEAD_ANCHOR`, déporté du COM du tronc. Son accélération
fusionne l'accélération linéaire du tronc et l'accélération angulaire (la
rotation projette en tangentiel sur le point déporté) — un seul vecteur capte
"je tombe en avant", "je suis projeté", "je bascule".

pymunk n'expose pas l'accélération. On la calcule par différence finie de la
position monde du point tête entre deux pas consécutifs :

```python
head_world = torso.local_to_world(HEAD_ANCHOR)
ax = (head_world.x - head_prev.x) / dt
ay = (head_world.y - head_prev.y) / dt
head_prev = head_world
```

L'accélaration est calculée en repère monde (mesure inertielle),
donc pas de rotation selon `torso.angle` mais la
composante x est miroirée par `facing` pour donner le sens égocentré
"avant/arrière" ; la composante y (haut/bas) reste brute.

## Format de sortie

`dict` plat nommé, lisible, non ordonné. L'encodeur du transformer (groupes
délimités par tokens PROPRIO, VISION, …) viendra plus tard. Pour l'instant le
`dict` suffit pour l'IHM de visualisation et les tests.

## Cadencement

On expose une classe `Proprio` qui produit un snapshot plat. Le cycle par
frame est : les sous-pas physiques, puis `proprio.update(skel, dt)` pour
obtenir le `dict`. La quantification en tokens et le cadencement viendront
avec l'encodeur du transformer (Stage ultérieur).

Les **mesures instantanées** (angles, forces ressort, couple) sont lues à
chaque frame de rendu (60 Hz) à partir du dernier état physique.

L'**accélération** nécessite un pas précédent : mémorisée frame à frame
(60 Hz).
