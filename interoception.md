# Intéroception de λ̄

L'intéroception est la perception de l'état interne : « dans quel état interne
suis-je ? » (faim, fatigue, souffrance, excitation, etc.).

## Etats cumulés des coûts physiques

Les signaux de renforcement (coûts innés) comme la douleur ou le confort,
font partie de la proprioception, voir proprioception.md.

L'intéroception est une perception de la synthèse de ces signaux:

* `fatigue`: L'effort est un coût instantané des actuateurs physiques.
  L'effort s'accumule en **fatigue**.
* `souffrance`: La douleur est un autre coût instantané (chocs, accélération
  excessives, ...).  La douleur s'accumule en **souffrance**.

La **récupération** consiste à faire décroite lentement ces accumulateurs.

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

===

## Renforcements motivationnels (TODO)

### Ennui

L'ennui est un coût pénalisant une trop grande prédictibilité du flux sensoriel.

Il s'appuie sur l'erreur de prédiction du world model :

```python
imprevision = L_t = -log P(o_{t+1}, r_t | o_t, a_t)
ennui = k1 * temps_sans_reward + k2 * (1 - imprevision)
```

L'ennui pourrait être amplifié par un renforcement trop neutre,
par exemple, d'après le temps écoulé depuis le dernier
renforcement significatif (`|reward| > seuil`).

```python
if |reward| > seuil_reward:
    ennui *= decay_ennui        # décroît brutalement sur stimulation
else:
    ennui += k_ennui * dt       # croît lentement en l'absence de reward
```

### Eveil (TODO)

Cycle métabolique influencé par la fatigue et la souffrance.
Quand le cycle arrive à 0 l'animal s'endort (phase de veille).

```python
base = 0.5 + 0.5 * sin(2*pi*t / periode_circadienne)
eveil = max(0, base - k_fat * fatigue - k_souff * souffrance)
```

### Faim (TODO)

TODO

## Playfulness / esprit joueur (TODO)

`playfulness`/`esprit joueur` est un seuillage haut de l'éveil, neutralisé
par la fatigue et la souffrance. 

```python
playfulness = max(0, eveil - seuil_eveil) * (1 - fatigue) * (1 - souffrance)
```

`playfulness` ∈ [0, 1]

L'esprit joueur conditionne le renforcement
de "comportements de jeu": comme

* la poursuite d'un mobile attrayant (balle),
* le contact,
* etc.

```python
play_rewards = playfulness * (reward_poursuite + reward_contact + ...)
```

`playfulness` pourrait être influencé par l'ennui.