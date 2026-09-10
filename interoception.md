# Intéroception de λ̄

L'intéroception est la perception de l'état interne : « dans quel état interne
suis-je ? » (faim, fatigue, douleur, excitation, etc.).

Contrairement à la proprioception (état du corps) et à l'extéroception (monde
extérieur), l'intéroception agrège des signaux de
renforcement: la fatigue est un cumul d'effort, la souffrance un
cumul de douleurs, etc.

Note: L'intéroception n'est **pas égocentrée**.

## Etats cumulés des coûts physiques

* `fatigue`: L'effort est un coût instantané des actuateurs.
  L'effort s'accumule dans un état de fatigue qui décroît lentement.
* `souffrance`: La douleur est un coût instantané associé à la proprioception
  (collisions et accélération excessives, voir proprioception.md).
  La douleur s'accumule dans un état de souffrance qui décroit lentement.

## Etats motivationnels

* `ennui`: 1ère implémentation : Coût pénalisant l'absence de reward
  significatif (`cumul(|reward|) < seuil`) sur une période récente.
  Seconde implémentation: prédictibilité du flux sensoriel basé sur
  l'erreur de prédiction du world model.
* `eveil`: Cycle métabolique de type cicardien. Sans renforcement.
  Ce cycle est perturbé par la fatigue et la souffrance.
* `faim`: non implémenté
* `playfulness`/excitation: L'esprit joueur est à son max
  quand la souffrance et la fatigue sont minimales
  et l'éveil maximal. L'esprit joueur conditionne le renforcement
  d'un comportement joueur de l'animat (non implémenté).

## Formules

### Fatigue

La fatigue accumule l'effort. Filtre IIR passe-bas :

```python
fatigue = fatigue * decay_eff + k_eff * effort * dt
```

`decay_eff` est un facteur de récupération (demi-vie ~100 s).

### Souffrance (cumul de douleur)

La souffrance accumule la douleur.

```python
souffrance += k_douleur * douleur
souffrance *= decay_souffrance   # demi-vie ~1000 s, plus lente que la fatigue
```

### Ennui (deux implémentations)

**1ère version — absence de reward.** On suit le temps écoulé depuis le dernier
renforcement significatif (`|reward| > seuil`). Si rien n'arrive, l'ennui
croît ; dès qu'un reward arrive (positif ou négatif), il décroît. Calculable
sans world model.

```python
if |reward| > seuil_reward:
    ennui *= decay_ennui        # décroît brutalement sur stimulation
else:
    ennui += k_ennui * dt       # croît lentement en l'absence de reward
```

**2ème version — prédictibilité du reward.** Généralisation: même un
flux riche de récompenses devient ennuyeux s'il est prévisible. Nécessite
l'erreur de prédiction du world model :

```python
imprevision = L_t = -log P(o_{t+1}, r_t | o_t, a_t)
ennui = k1 * temps_sans_reward + k2 * (1 - imprevision)
```

### Eveil

Cycle métabolique. Pas de renforcement. Juste un état. Quand le cycle arrive
à 0 l'animal s'endort (à préciser).

```python
base = 0.5 + 0.5 * sin(2*pi*t / periode_circadienne)
eveil = max(0, base - k_fat * fatigue - k_souff * souffrance)
```

### Faim

*Non implémenté pour l'instant.*

## Playfulness / excitation et instincts du jeu

*Non implémenté pour l'instant*

Un état de **playfulness** module l'intensité des instincts de jeu,
qui renforcent des comportements joueur: poursuite, contact, etc.

```python
play_rewards = playfulness * (reward_poursuite + reward_contact + ...)
```

`playfulness` ∈ [0, 1] est un seuillage haut de l'éveil, neutralisé
par la fatigue et la souffrance. C'est un signal dérivé sans
renforcement associé.

```python
playfulness = max(0, eveil - seuil_eveil) * (1 - fatigue) * (1 - souffrance)
```

## Format de sortie

Comme les autres modalités : `dict` plat nommé, lisible, non ordonné.

## Cadencement

Cadencé avec le world model.
