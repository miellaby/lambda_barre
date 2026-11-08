## Forçage EOS dans la distribution du vécu

**Fait.** `eos_force_eps_lo/hi` : 0,30–0,70 → 0,10–0,25.

Le forcage donnait des EOS en dehors du vécu.

## Donner du gradient à la prédiction avec anticipation (lookahead)

Le WM n'apprend pas assez en lookahead.

**Fait** Passer `_WM_LOOKAHEAD_PERIOD` de 4 à 2 (une passe lookahead sur deux).

A voir: Amplifier le loss des passes lookahead sur les séquences à dénouement positif (ex. `lookahead_relief_boost = 4`).

## Contraster la pondération d'impact WM

Le WM ne se concentre pas assez sur les séquences qui ont de l'impact.

**Fait** Actuellement le WM se concentre sur les séquences qui ont des conséquences via un poids = |relief| + plancher. Le plancher avait été monté 0,25 pour adoucir mais c'est trop, ca écrase la pondération par soulagement. `wm_impact_floor` repasse à 0,10.

## Diluer la masse immobile

Rien fait: Le corpus semble dominé par de la redondance immobile.
Mieux lisser avant de détecter les pauses? les détecter différement (dérivée des signaux plutôt que moyenne)? Sous-échantillonner les séquences à dénouement nulle? A revoir mais pas maintenant.

## Protéger les démonstrations de l'oubli actif

Rien fait:

Supprimer des séquences « familières » supprime du dataset des séquences qui ne peuvent être réapprises. Le corpus s'auto-détruit?

Empécher une partie des séquences à fort dénouement positif d'être oubliables mais alors comment empecher l'accumulation?

Ou placer les démos dédiées dans un dataset à part? Non, ce serait
vers de l'apprentissage supervisé classique...

Peut-être pondérer le filtre de surprise et la sélection des séquences oubliables avec fav(relief)? Pour que les séquences à dénouement positives soient plus dures à oublier...

## Donner du gradient à la clé EOS

**Fait**: Le masque lookahead laisse maintenant plus de tokens assister l'EOS terminal.

## Gain de correction sur la suggestion d'insight

**A voir**

Appliquer un gain sur la correction (différence entre l'action suggérée par l'insight et l'action policy) avant vérification et apprentissage: `action retenue = action_policy + gain × (action_générée − action_policy)`. Un gain à 4 combat le moyennage du vécu par le WM. Vérification après amplification que cela donne le dénouement attendu.

## La passe lookahead apprend a4 d'après le dénouement: la loss est l'écart entre a4 estimé et a4 observé

**Fait** La loss en lookahead couvre uniquement les tokens consigne de a4, et pas les états, les actions, et les autres slots de la salve terminale.

## l'a4 hindsight est toujours la cible de la policy

**Fait** Retrait de la vérification par rollout causal de
l'a4 générée par EOS forcé.

Le rollout causal de vérification d'une action (prédite
par la passe lookahead à EOS forcé) ne fonctionne pas.
Et c'est normal, si le corpus contient en majorité des
échecs, il prédira un échec même si l'action proposée est
une action qui entraîne au succès.

Note: une vérification d'atteignabilité, consiste à inférer *sous condition de dénouement*. Peut-être on pourrait vérifier l'action proposée?

## Croquettes : faim innée et déplacement émergent (chasse passive, non supervisée)

**Statut : à décider**

**Conception.** N croquettes carrées (12 px) sont des corps Pymunk dynamiques de masse très faible (0,05) : les pieds de l'animat les poussent, mais elles ne déplacent pas l'animat — cela émerge de la répartition des impulsions selon les masses inverses, aucun filtre de collision spécial. La consommation n'est PAS une collision : si la tête est à moins de `SEUIL_CONSO` (ex. 30 px) de la croquette la plus proche, celle-ci est mangée en `DUREE_CONSO = 6 s` (le carré passe du vert au gris selon la progression). Pendant la consommation, un unique nouveau scalaire de coût bénéfique `délectation` vaut son maximum (il alimente `reward_pos`) ; la `faim` descend linéairement pendant les 6 s pour atteindre exactement −50 % à la fin de la croquette. `faim` est un nouvel intégrateur lent (sature en ~600 s, clamp [0,1]) et un 6e coût inné du token 10 (poids 4, comme la douleur) : il entre dans `salve_cost` / relief / EOS sans toucher le pipeline brain (`n_cost = len(_COST_KEYS)` est dynamique).

**Simplifications retenues.**
- Un seul nouveau scalaire de reward (`délectation`) : le « confort au max » pendant la bouchée passe par lui ; le canal `confort` existant n'est pas modifié.
- Capteur croquette = 2 scalaires (`croq_dir`, `croq_prox` vers la croquette la plus proche) placés dans les slots libres du token 5 (env/ball, pattern `Ball`) — pas de nouveau token, pas de vr/va (cibles lentes).
- Consommation = test de distance tête→croquette la plus proche, pas de collision pymunk tête-croquette.
- « Pousser mais pas l'inverse » = masse 0,05, purement physique.
- `faim` uniquement dans le token 10 (coût) + HUD — pas dans le token 12 intero, pour ne pas toucher au zero-padding intero ni à l'adaptation « 45 phys + 2 intero » de la policy.
- Une seule croquette consommable à la fois (la plus proche).

**Fichiers.** `tokenize.py` (4 nouvelles specs ; token 5 → 6 signaux, token 10 → 6 (`faim`), token 11 → 2 (`délectation`) ; `salve_cost` généralisée à tous les signaux du canal 11), `proprio.py` (`Reward.update` reçoit `faim` et `mange_en_cours` en paramètres optionnels ; `reward_neg = (… + 4·faim)/16`, `reward_pos = confort + délectation`), `intero.py` (intégrateur `faim`), `world.py` (`CROQUETTE_TYPE = 5`, spawn/respawn de N croquettes, `update_croquettes(space, skel, dt)` : conso + sortie de limites), `main.py` + `dataset.py` (câblage ; paramètres optionnels pour ne rien casser), `render.py` (carrés vert→gris, HUD faim/délectation), `brain.py` (remplacer les `SIG_OFFSET + 5` hardcodés par `n_cost`, lignes ~1260–1269 et ~1854), `config.toml` (`[croquettes]`), `test_brain.py`.

**Mesures de validation.** Les séquences « manger » ont un relief fortement positif ; la faim HUD monte entre deux repas et chute de moitié à chaque croquette ; après sommeil, la policy initie des déplacements vers la croquette (croq_prox qui augmente en egocentré).

**Humain :** OK

