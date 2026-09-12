# Encodeur de tokens de λ̄

L'encodeur transforme l'ensemble des signaux de λ̄ en une séquence de tokens
destinés au world model.

## Structure d'un token

Voir @densification_tokens.md

## Quantification

Quantification sur 9 niveaux (0 à 8, avec toujours un niveau au milieu).

## Salve

Une **salve** est la séquence complète de tokens produite à chaque tick du
world model (6 Hz). Voir @densification_tokens.md

## Cadence

La salve est générée à **3 Hz**. Plus lent que la politique (6 Hz).
Les données de modalités sensorielles et moteurs sur chaque période sont synthétisées.
