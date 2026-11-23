# Spécification — World Model à dénouement (entraînement rétrotemporel)

## Séquence enregistrée

Une séquence est une fenêtre temporelle ordonnée ainsi :

`contexte | conséquence rétro-causale | action de décision`

- **contexte** : `[s0,a0, ..., a_{n_ctx-1}, s_{n_ctx}]`, ordre normal
  (`s_{n_ctx}` seul, sans action).
- **conséquence rétro-causale** : unités de 16 tokens pour
  `k = seq_steps .. n_ctx+1`, du plus tardif au plus précoce — `s_k` avec ses
  13 tokens d'état **inversés** (token 12 en premier), puis `a_{k-1}`
  (3 tokens, ordre interne normal) : `s26, a25, s25, a24, ..., s5, a4`.
- **token de dénouement** : le token 12 de la première unité (`s_{seq_steps}`),
  premier token après le contexte (`eos_pos = ctx_len`), valeur observée réelle
  (ground truth) calculée à l'extraction. Il résume l'issue de la trajectoire
  post-décision et est prédit juste après le contexte à l'inférence.
- **action de décision** : `a_{n_ctx}`, dernier token de la séquence.

## Entraînement du transformer

Un seul masque, un seul entraînement. La loss porte sur tous les tokens à
partir du dénouement ; le contexte sert de prompt et n'est pas appris (poids
0), comme dans un transformer de traduction (loss sur la cible, pas sur la
source).

Dans le même forward :

- le **dénouement est prédit depuis le contexte** (plusieurs dénouements
  possibles : mauvais, bon, …) ;
- la **conséquence inversée puis l'action de décision sont prédites
  conditionnellement au dénouement**.

Aucun dénouement contrefactuel n'est injecté à l'entraînement : pas de faux
dataset, pas d'actions factices.

Les sorties sont des régressions continues (signaux en [0,1]) ; aucune
quantification des tokens intermédiaires. Seul le dénouement fait l'objet
d'un choix discret (échantillonnage).

## Inférence

1. Depuis le contexte, le WM prédit une distribution de dénouements.
   Échantillonnage top-K / top-p avec température (pas d'argmax : l'argmax
   donnerait la moyenne conditionnelle, toujours identique).
2. Les dénouements échantillonnés sont classés selon `f(probable, favorable)`
   et le meilleur est élu — pas nécessairement le plus probable.
3. Génération auto-régressive standard depuis `[contexte, dénouement élu]` :
   la conséquence inversée est déroulée jusqu'à l'action de décision, dernier
   token de la séquence.

L'élection d'un dénouement meilleur que l'observé (extrapolation
cross-contexte) n'est utilisée que si aucun dénouement observé depuis ce
contexte n'est satisfaisant ; sinon on reste sur l'observé.

## Politique

Le latent d'entrée de la policy est produit par le WM sur la partie contexte.

L'apprentissage de la policy imite l'action de décision générée. La
descente de gradient est pondérée **linéairement** par la probabilité du
dénouement élu, `w = p`, par séquence du batch (`loss *= p.detach()`), où `p`
est lue dans les logits du token dénouement (softmax des slots du dénouement).
