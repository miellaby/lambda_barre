# Extéroception de λ̄

L'extéroception est la perception de l'environnement externe : « que se
passe-t-il autour de moi ? ».

Contrairement à la proprioception (qui mesure l'état du corps), l'extéroception
mesure le monde extérieur. L'animat ne perçoit pas des abstractions informatiques
discrètes (« fenêtre », « bouton », « souris », « clavier ») mais un petit
nombre de valeurs continues, respectant l'égocentricité.

Source : `Lambda barre.md` § "L'environnement sensoriel de λ̄" (l.195-258) et
§ "Encodage de la perception" (l.313-352).

## Égocentricité et facing

Même principe qu'en proprioception : le `facing` est une variable interne de
miroir, jamais perçue. L'animal ignore s'il regarde à gauche ou à droite.

Mais l'extéroception introduit une différence : l'environnement est **fixe dans
le repère monde**, alors que le corps de l'animat se déplace et tourne. La
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

Aucune rotation par `torso.angle` : l'animat est un être 2D qui ne tourne que
selon l'axe gauche/droite (facing). Le miroir facing suffit à égocentrer.

## Vision

La vision est rétinienne, très basse résolution, à courte portée.

### Couleurs

Le cône de vision est un triangle dans le prolongement de la tête, pointant
vers l'avant (côté facing). Il est découpé en 16 cellules disposées en 4×4.

Pour chaque cellule, on calcule :
- une couleur moyenne HSV (3 composantes : teinte, saturation, valeur)
- une mesure d'hétérogénéité (1 scalaire : variance ou écart-type des couleurs
  dans la cellule)

Soit 16 × 4 = 64 valeurs.

```python
[VISION]
  cellule_1   # (H, S, V, hétérogénéité)
  cellule_2
  ...
  cellule_16
```

Ces informations rudimentaires permettent de distinguer une surface uniforme,
un bord de fenêtre, un objet coloré, du mouvement.

### Flux optique

Le moteur graphique calcule le différentiel spatial entre deux frames
(ce qui a changé dans le champ visuel). Ce différentiel est synthétisé en :

- `surface_diff` : surface totale du différentiel (scalaire, sans miroir)
- `flux_x` : `facing * dx` du barycentre du différentiel, référentiel tête
- `flux_y` : `dy` du barycentre, référentiel tête

```python
[VISION]
  ...
  flux_optique   # (surface, dx_ego, dy_ego)
```

Cela restitue un flux optique grossier : « quelque chose de gros bouge à
gauche/droite ».

## Curseur de la souris

Position et vitesse du curseur, dans le référentiel de la tête :

- `curseur_dx` : `facing * (mouse_x - head_x)`
- `curseur_dy` : `mouse_y - head_y`
- `curseur_vx` : `facing * mouse_vx`
- `curseur_vy` : `mouse_vy`

4 valeurs. Le miroir facing s'applique aux composantes horizontales (dx, vx)
pour que « le curseur est devant moi » soit toujours positif.

```python
[ENV]
  curseur   # (dx, dy, vx, vy) égocentrés
```

## Toucher

Le tronc et les 2 pattes postérieures sont des capteurs tactiles. Comme pour la
proprioception, les pattes sont échangées selon le facing (avant = patte vers la
tête, arrière = l'autre).

Sensations tactiles :

- `contact_sol_avant` : contact sol / pied avant, valeur binaire
- `contact_sol_arriere` : contact sol / pied arrière, binaire
- `force_normale_avant` : force normale du sol sur la patte avant (déjà calculée
  en proprioception sous `force_contact_sol_avant` ; partagée)
- `force_normale_arriere` : idem patte arrière
- `collision_tronc` : vecteur force (x, y) de collision du tronc avec un bord de
  fenêtre ou un objet, autre que le sol. Égocentré : composante x miroirée par
  facing.
- `collision_avant` : vecteur force de collision de la patte avant (hors sol)
- `collision_arriere` : vecteur force de collision de la patte arrière (hors sol)

Les forces de contact sol sont partagées avec la proprioception (même mesure,
deux interprétations : « suis-je soutenu ? » = proprio, « que touche-je ? » =
extéro). Les collisions non-sol sont spécifiques à l'extéroception.

```python
[TOUCH]
  tronc      # (collision_x, collision_y) égocentrés
  patte_av   # (contact_sol, force_normale, collision_x, collision_y)
  patte_ar   # idem
```

### Inconfort d'immobilité

L'immobilité prolongée produit une sensation d'inconfort avec un coût
intrinsèque croissant. Ce renforcement système encode l'instinct biologique qui
pousse au mouvement.

Ce signal est frontière entre extéroception et intéroception : il s'agit d'une
mesure interne (depuis quand n'ai-je pas bougé ?) mais sa sémantique est
motivée par l'environnement. Proposition : le ranger en intéroception, car c'est
un état interne dérivé de la proprioception (absence de variation des signaux
proprioceptifs).

## Audition

Champ auditif simulé à partir d'événements du bureau : clic gauche/droit, touche
clavier, notification/événement système, fenêtre qui apparaît/disparaît.

Chaque source sonore est perçue par un triplet :

- **angle** : `facing * atan2(source_y - head_y, source_x - head_x)` — direction
  de la source dans le référentiel de la tête. Miroiré par facing.
- **intensité** : `I = I0 / (1 + d²)` où `d` = distance source-tête. Pour les
  clics de souris, la force du signal encode le CPS (clics par seconde). Sans
  miroir (scalaire radial).
- **fréquences** : vecteur 2D caractérisant la source. Pour une touche clavier :
  fréquence X = colonne de la touche, fréquence Y = ligne. Pour un clic :
  type de clic. Pour un événement système : type d'événement.

```python
[AUDIO]
  source_1   # (angle, intensité, freq_x, freq_y)
  source_2
  ...
```

Le nombre de sources est variable (liste d'événements récents, avec décroissance
temporelle de l'intensité). À chaque frame, seules les sources « actives »
(intensité > seuil) sont émises.

L'audition de sa propre voix (vocalises) est un signal proprioceptif, pas
extéroceptif — c'est la perception d'un actuateur interne, pas d'un événement
externe.

## Cadencement

Les modalités extéroceptives sont plus lentes que la proprioception :

- **Vision** : cadencée à ~6 Hz (10× plus lent que la proprioception à 60 Hz).
  Le cône visuel est un instantané de l'écran, pas besoin de 60 Hz.
- **Flux optique** : dérivé de deux frames visuelles consécutives, donc ~6 Hz.
- **Curseur** : 30 Hz (intermédiaire — le mouvement du curseur est rapide mais
  pas besoin de 60 Hz).
- **Toucher** : 60 Hz (contact sol déjà calculé en proprioception ; les
  collisions non-sol suivent le pas physique).
- **Audition** : pilotée par événements (asynchrone), avec décroissance
  temporelle. Un buffer de sources actives est maintenu et échantillonné à
  chaque frame de vision (~6 Hz).

## Format de sortie

Chaque modalité produit un `dict` ou une liste de `dict`, assemblés par
l'encodeur du transformer en séquences d'embeddings délimitées par tokens
symboliques : `[VISION]`, `[TOUCH]`, `[AUDIO]`, `[ENV]`. Pour l'instant, des
`dict` plats suffisent pour l'IHM de visualisation et les tests.

## Implémentation — état actuel

Aucune modalité extéroceptive n'est implémentée. La force de contact sol
(`force_contact_sol_avant/arriere`) existe dans `sensors.py` mais est rangée
en proprioception. Le toucher non-sol, la vision, le curseur et l'audition
nécessitent un accès à l'environnement de bureau (capture d'écran, position
réelle de la souris, événements système) qui n'existe pas encore dans
l'application pygame isolée.
