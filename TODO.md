# TODO — Refonte du World Model (REFONTE.md)

Spec source : REFONTE.md. Le WM passe à un entraînement rétrotemporel à
 dénouement. Le reste de la doc (README.md, DESIGN.md, sessions, ADL) décrit
l'ancien fonctionnement à deux passes causale/lookahead : doc legacy, à
relire avec recul.

## Layout des séquences enregistrées

- [ ] Fenêtre ordonnée : `contexte | dénouement | conséquence inversée | action de décision`
- [ ] Dénouement = valeur observée réelle (ground truth), placée juste après le contexte
- [ ] Conséquence inversée : états post-décision du plus tardif au plus précoce
- [ ] Action de décision = dernier token de la séquence
- [ ] Changement de layout = checkpoints invalides (`buf_ckpt.pt`, `wm_ckpt.pt`) : prévoir le `--reset`

## Entraînement du WM

- [ ] Un seul masque, un seul entraînement : loss sur tous les tokens à partir du dénouement
- [ ] Contexte = prompt à poids 0 (loss sur la cible, pas sur la source, comme un transformer de traduction)
- [ ] Dans le même forward : dénouement prédit depuis le contexte ; conséquence inversée puis action de décision prédites conditionnellement au dénouement
- [ ] Aucun dénouement contrefactuel injecté à l'entraînement (pas de faux dataset, pas d'actions factices)

## Sorties

- [ ] Régressions continues [0,1] sur tous les tokens intermédiaires, aucune quantification
- [ ] Seul le dénouement est discret : slots + softmax (sert à l'échantillonnage et à la proba p)

## Inférence

- [ ] Depuis le contexte, distribution de dénouements : échantillonnage top-K / top-p avec température (pas d'argmax)
- [ ] Classement des dénouements échantillonnés selon f(probable, favorable) ; le meilleur est élu, pas nécessairement le plus probable
- [ ] Génération auto-régressive depuis `[contexte, dénouement élu]` : dérouler la conséquence inversée jusqu'à l'action de décision (dernier token)
- [ ] Dénouement élu meilleur que l'observé (extrapolation cross-contexte) uniquement si aucun dénouement observé depuis ce contexte n'est satisfaisant

## Policy

- [ ] Latent d'entrée de la policy = sortie du WM sur la partie contexte
- [ ] Apprentissage par imitation de l'action de décision générée
- [ ] Loss pondérée linéairement par la proba du dénouement élu : `loss *= p.detach()` par séquence du batch, p lu dans les logits (softmax des slots du dénouement)

## Intégration & doc

- [ ] Reporter les changements dans wm_manager, wm_theater, dream
- [ ] Mettre à jour DESIGN.md, README.md, tokens.md / encodeur.md si le layout évolue
- [ ] Proposer un ADR dans ADL/ pour la nomenclature de la refonte
- [ ] Ouvrir session_YYYY_MM_DD.md pour tracer la refonte
- [ ] Tests test_brain.py : masque unique (poids 0 sur le contexte), forward conditionnel au dénouement, échantillonnage top-K/température, pondération policy par `p.detach()`
