# Encodeur de tokens de λ̄

L'encodeur transforme l'ensemble des signaux de λ̄ en une séquence de tokens
destinés au world model. C'est l'interface entre les capteurs (proprio,
extéro, intero, reward, actions) et le transformer.

Source : `Lambda barre.md` § "Un transformer multimodal" (l.104-127) et
§ "Encodage de la perception" (l.313-352).

## Structure d'un token

Chaque token est encodé sur 2 octets :

- **Octet 0 — ID** : identifie le type de token (1-255).
- **Octet 1 — valeur** : valeur quantifiée sur 256 niveaux (0-255).

Deux sortes de tokens :

- **Tokens de séparation** : marquent le début d'un groupe de modalité. Leur
  valeur est 0 (ignorée). 7 séparateurs : PROPRIO, VISION, ENV, TOUCH,
  INTERO, REWARD, ACTION.
- **Tokens scalaires** : encodent un signal concret. La valeur est quantifiée
  selon le type du signal (signé ou non signé).

## Quantification

La valeur brute est normalisée par une échelle propre à chaque signal, puis
quantifiée sur 0-255 :

- **Signé** (valeurs dans [-scale, +scale]) : `(raw / scale + 1) / 2 * 255`
  - 128 = zéro, 0 = -scale, 255 = +scale
- **Non signé** (valeurs dans [0, scale]) : `raw / scale * 255`
  - 0 = zéro, 255 = scale

La quantification sur 256 niveaux est essentielle pour que le world model
prédise une distribution de trajectoires **multi-modale** (distinctes,
hétérogènes) plutôt qu'une distribution gaussienne centrée autour d'un futur
moyen.

## Salve

Une **salve** est la séquence complète de tokens produite à chaque tick du
world model (6 Hz). Elle contient les 7 groupes dans l'ordre fixe :

```
[PROPRIO]  11 tokens scalaires
[VISION]   19 tokens scalaires (16 cellules × 1 brightness + 3 flux optique)
[ENV]      9 tokens scalaires (curseur + son)
[TOUCH]    6 tokens scalaires
[INTERO]   2 tokens scalaires
[REWARD]   8 tokens scalaires
[ACTION]   5 tokens scalaires
```

Total : 67 tokens par salve (7 séparateurs + 60 scalaires).

## Vocabulaire

Les IDs 1-7 sont les séparateurs. Les IDs 10+ sont les scalaires, assignés
séquentiellement par groupe.

### PROPRIO (IDs 10-20)

| ID  | Code | Signal                  | Scale        | Signé |
|-----|------|-------------------------|--------------|-------|
| 10  | TA   | tronc_angle             | 45°          | oui   |
| 11  | AA   | membre_angle_avant      | π            | oui   |
| 12  | DA   | membre_distance_avant   | 32           | non   |
| 13  | FA   | force_actuateur_avant   | 2000         | oui   |
| 14  | AR   | membre_angle_arriere    | π            | oui   |
| 15  | DR   | membre_distance_arriere  | 32           | non   |
| 16  | FR   | force_actuateur_arriere | 2000         | oui   |
| 17  | QA   | queue_angle             | π            | oui   |
| 18  | CQ   | couple_queue            | 200000       | oui   |
| 19  | XA   | accel_tete_avant        | 2000         | oui   |
| 20  | YA   | accel_tete_haut         | 2000         | oui   |

### VISION (IDs 21-39)

16 cellules × 1 brightness (grayscale [0, 1]) + 3 flux optique.

| IDs     | Code format | Signal                          | Scale | Signé |
|---------|-------------|---------------------------------|-------|-------|
| 21-36   | V{i}        | vis_c{i} (i = 1..16)            | 1.0   | non   |
| 37      | FS          | flux_surface                    | 5000  | non   |
| 38      | FX          | flux_x                          | 150   | oui   |
| 39      | FY          | flux_y                          | 150   | oui   |

### ENV (IDs 88-96)

| ID  | Code | Signal       | Scale  | Signé |
|-----|------|--------------|--------|-------|
| 88  | CD   | curseur_dir  | π      | oui   |
| 89  | CP   | curseur_prox | 1.0    | non   |
| 90  | CV   | curseur_vx   | 1500   | oui   |
| 91  | CW   | curseur_vy   | 1500   | oui   |
| 92  | S0   | son_0        | 1.0    | non   |
| 93  | S1   | son_1        | 1.0    | non   |
| 94  | S2   | son_2        | 1.0    | non   |
| 95  | S3   | son_3        | 1.0    | non   |
| 96  | S4   | son_4        | 1.0    | non   |

### TOUCH (IDs 97-102)

| ID  | Code | Signal               | Scale  | Signé |
|-----|------|----------------------|--------|-------|
| 97  | SA   | contact_sol_avant    | 1500   | non   |
| 98  | SR   | contact_sol_arriere  | 1500   | non   |
| 99  | TX   | collision_tronc_x    | 1500   | oui   |
| 100 | TY   | collision_tronc_y    | 1500   | oui   |
| 101 | TC   | collision_tronc_cx   | 200    | oui   |
| 102 | TD   | collision_tronc_cy   | 200    | oui   |

### INTERO (IDs 103-104)

| ID  | Code | Signal     | Scale | Signé |
|-----|------|------------|-------|-------|
| 103 | FT   | fatigue    | 1.0   | non   |
| 104 | SF   | souffrance | 1.0   | non   |

### REWARD (IDs 105-112)

| ID  | Code | Signal       | Scale | Signé |
|-----|------|--------------|-------|-------|
| 105 | EF   | effort       | 1.0   | non   |
| 106 | DO   | douleur      | 1.0   | non   |
| 107 | CO   | courbature   | 1.0   | non   |
| 108 | IN   | instabilite  | 1.0   | non   |
| 109 | VE   | vertige      | 1.0   | non   |
| 110 | CF   | confort      | 1.0   | oui   |
| 111 | RP   | reward_pos   | 1.0   | oui   |
| 112 | RN   | reward_neg   | 1.0   | non   |

### ACTION (IDs 113-117)

| ID  | Code | Signal        | Scale | Signé |
|-----|------|---------------|-------|-------|
| 113 | LT   | limb_l_theta  | π/2   | oui   |
| 114 | LD   | limb_l_d      | 32    | non   |
| 115 | RT   | limb_r_theta  | π/2   | oui   |
| 116 | RD   | limb_r_d      | 32    | non   |
| 117 | TQ   | tail_theta    | π/2   | oui   |

## Cadencement

La salve est générée à **6 Hz** (une salve toutes les 10 frames de rendu à
60 Hz). C'est la cadence du world model, plus lente que la proprio (60 Hz) ou
le curseur (30 Hz). Les signaux sont échantillonnés au moment de la salve,
pas interpolés.

## Format de sortie

`encode()` retourne une `list[tuple[int, int]]` — chaque tuple est
`(id, valeur)`, tous deux dans 0-255. `decode()` retourne une liste de chaînes
pour l'affichage (`[PROPRIO]`, `TA=135`, etc.).

## IHM

Les 10 dernières salves sont affichées en 2 colonnes à droite de l'écran,
en texte 10pt, avec auto-scroll vertical (les plus récentes en bas).

## Implémentation — état actuel

- `tokenize.py` : classe `TokenEncoder`, vocabulaire, quantification,
  encodage et décodage.
- `render.py` : `draw_tokens` — affichage des 10 dernières salves.
- `main.py` : génération à 6 Hz, buffer circulaire des 10 dernières salves.

### À préciser

- **Compression temporelle** : les actions entre deux salves (200 ms) sont
  échantillonnées ponctuellement, pas résumées. La compression des blocs
  moteurs et la gestion des événements ponctuels (clic, reward) viendront
  avec l'encodeur du transformer (voir `Lambda barre.md` § "Granularité
  temporelle").
- **Échelles** : les scales de quantification sont reprises de l'IHM de
  visualisation (`render.py`). Elles devront être ajustées quand le world
  model sera entraîné.
