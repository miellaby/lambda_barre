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
* `force_contact_sol_avant` — impulsion normale / dt, patte avant. Sans miroir
   (scalaire normal).
* `membre_distance_arriere` — distance mesurée de la patte arrière. Sans
   miroir.
* `force_actuateur_arriere` — `spring.impulse / substep_dt` de la patte arrière.
   Sans miroir.
* `force_contact_sol_arriere` — impulsion normale / dt, patte arrière. Sans
   miroir.

Scalaires horizontaux:
* `accel_tete_avant` — `facing * ax_monde`. Accélération de la tête (point
    `HEAD_ANCHOR`), composante x en repère monde, différence finie. + = vers
    l'avant.

Scalaires verticaux:
* `accel_tete_haut` — `ay_monde`. Composante y en repère monde, différence
    finie. + = vers le haut. Sans miroir (gravité).

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

pymunk n'expose pas directement la force de contact hors d'un `Arbiter`. On
ajoute un `CollisionHandler` `post_solve` entre les pieds et le sol, qui
accumule l'impulsion normale reçue par chaque pied sur le pas de simulation.
La force = `impulsion / dt`. Le signal binaire `foot_contacts` (heuristique
positionnelle existante) est conservé pour la transition mais la force continue
est la valeur à apprendre.

```python
space.on_collision(FOOT_TYPE, GROUND_TYPE, post_solve=callback)
def post_solve(arb, space, data):
    ny = arb.total_impulse.y
    for shape in arb.shapes:
        if shape.body is foot: contact_impulse[foot] += ny
```

Pas de miroir : la force normale est un scalaire (composante y), pas une
direction dans le plan.

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
frame est : `proprio.reset_contacts()` avant les sous-pas physiques, puis
`proprio.update(skel, dt)` après pour obtenir le `dict`. La quantification en
tokens et le cadencement viendront avec l'encodeur du transformer (Stage
ultérieur).

Les **mesures instantanées** (angles, forces ressort, couple) sont lues à
chaque frame de rendu (60 Hz) à partir du dernier état physique.

L'**accélération** et la **force de contact** nécessitent un pas précédent :
calculées à chaque pas physique (180 Hz, 3 sous-pas) ou mémorisées frame
à frame (60 Hz).
