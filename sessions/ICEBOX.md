# Backlog & Spécifications Futures — λ̄ (Lambda barre)

## Externaliser les nombres magiques vers `config.toml`

Tout réglage actuellement câblé en dur est candidat à une exposition dans `config.toml`.

## l'angle du tronc à déplacer de Proprioception à Exteroception (oreille interne)

A vérifier.

## Séquences « time-event » symboliques, sans horloge fixe

Un geste (ex. relevement) dure plusieurs secondes et contient plusieurs extrema de consignes (extension + retour du pied = 2 extrema, parfois 3). Les ticks intermédiaires forment du bourrage inutile entre deux moments clés.

Proposition:

- **Unité de séquence = l'extremum de consigne** (déjà enregistré comme extrême local par tick), pas le tick
- un token par moment clé du geste, avec le **délai vers la prochaine action en slot de la consigne**
- pas d'horloge externe, longueur de séquence variable = complexité du geste (nombre d'extrema).

Problème: il faut tout refaire! Séquence de taille variable!!!
