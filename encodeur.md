# Encodeur de tokens de λ̄

L'encodeur transforme l'ensemble des signaux de λ̄ en une séquence de tokens
destinés au world model. C'est l'interface entre les capteurs (proprio,
extéro, intero, reward, actions) et le transformer.

Source : `Lambda barre.md` § "Un transformer multimodal" (l.104-127) et
§ "Encodage de la perception" (l.313-352).

## Structure d'un token

Voir @densification_tokens.md

## Quantification

Quantification sur 9 niveaux (0 à 8, avec toujours un niveau au milieu).

## Salve

Une **salve** est la séquence complète de tokens produite à chaque tick du
world model (6 Hz). Voir @densification_tokens.md

## Cadencement

La salve est générée à **2 Hz**. La cadence du world model,
est plus lente que la proprio (6 Hz). Les 3 "frames" de modalités
sensorielles et moteurs sur la période sont synthétisées.

