
## Idées abandonnées

- NON: Entraînement de la policy basé sur l'imagination.
   * Apprentissage supervisé basé sur la meilleure action (d'après le futur prédit par le WM) entre 3 candidats: [action historique telle qu'enregistrée dans la séquence] [action générée par la politique courante] [exploration gaussienne σ = 0,30 autour de la politique courante].
   * Pour chaque action candidate: Imagination WM multi-futurs. On évalue l'action par son meilleur futur ("optimisme de Bellman").
   * 2 futurs: [prolongation Kalman amortie de l'action candidate] et [application de la policy sur les ticks suivants].
   * Prolongation Kalman avec clamping de vitesse physique (±0,20 rad/pas, ±0,15 extensions).

- NON: quantification des signaux de token.

- NON: entraînement REINFORCE de l'action inférée à partir du gain prédit par le WM

- NON: Générateur de dataset d'amorçage headless (`--bootstrap-dataset`) avec auto-reset sur 
chute. Câblé dans le code (`main.py`, `dataset.py`) mais jamais testé et non voulu.

- NON: Compilation optionnelle `torch.compile` du pas d'entraînement WM (graphe fusionné, passe arrière incluse).

- NON **Classification** des séquences et déterminisme action→dénouement

    - `determinisme` — somme de deux scores (formule ci-dessous), « oui » si la somme dépasse un seuil à régler expérimentalement.
    - `signe_denouement` — `+` / `0` / `−` (relief_score signé).
    - `surprenant` — loss > seuil de surprise (`evaluate_sequences_surprise`).
    - `nouveau` — dans l'addendum ou dans le coreset depuis peu (compteur individuel d'apprentissage proche de 0).
    - `vicace` — vécue récemment ou remémorée et conservée.
    - `impact_fort` — |relief| au-dessus du plancher.
    - `ameliorable` — dénouement de la séquence sous le seuil moyen du corpus (indépendant du déterminisme).

    NON **Formule du déterminisme** (WM supposé bien entraîné, test par contre-factuel — pas de fouille du corpus) :
        - det(action→conséquence) = similarité(dénouement WM sur [contexte + action enregistrée], dénouement enregistré) / dispersion des dénouements prédits par le WM pour les actions déductibles du contexte (action enregistrée, action WM causale, action WM conditionnée EOS, action policy courante).
        - det(conséquence→action) = similarité(action WM conditionnée par l'EOS enregistré, action enregistrée) / dispersion des actions générées pour d'autres EOS (ex. EOS neutre 0,5).
        - Interprétation : conséquences similaires pour des actions différentes = causalité exogène ; dénouement bien prédit pour l'action enregistrée = déterminisme par l'action.
    - **Quand calculer le déterminisme** : une seule fois par cycle de sommeil (pas de recalcul entre deux epochs), et seulement une fois le WM bien entraîné — ex. quand la loss WM atteint un plateau. Avant ce plateau, pas de calcul de déterminisme.
    - NON **Rôle de la classification des séquences**
        - non-déterministe → séquence proposée à l'éviction, d'autant plus qu'elle a déjà été souvent entraînée (compteur « nb de batchs WM » par séquence : **n'existe pas, à ajouter**, +1 à chaque batch coreset/addendum où la séquence est présente).
        - signe du dénouement : pas un critère — une séquence déterministe s'apprend quel que soit son dénouement.
        - surprenant : pour une séquence **ancienne** (compteur de batchs WM >= 1) la surprise basse (loss < surprise_floor) est un signe de déterminisme. Ce n'est pas un critère d'oubli sinon il ne restera que des séquences non-déterministes et donc non apprennable. Pour une séquence **neuve** (compteur de batchs WM = 0), une surprise basse (loss  < `surprise_floor`) signifie qu'on peut oublier la séquence.
        - vivacité : à combiner avec le déterminisme pour l'éviction.
        - impact : peut moduler le gain d'apprentissage, mais le dénouement neutre s'apprend aussi.
        - améliorable : ne pas apprendre de nouvelle action si l'action politique courante est suffisamment bonne — dénouement politique prédit ≥ dénouement enregistré de la séquence, ou ≥ quantile élevé du dénouement moyen du corpus — sinon on désapprend.
