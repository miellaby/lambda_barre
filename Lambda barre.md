## Introduction

### Les desktop pets

Les simulations d'animaux de compagnie, vivant dans l’environnement graphique des ordinateurs,  formaient  une catégorie de logiciels assez populaire au tournant des années 2000\.

On parle de **computer pet** ou **desktop pet** ou encore de **Virtual pet**. Un des premiers exemples connus était **Neko**: un chaton qui court sur l'écran et suit la souris. Avec **Dogz** qui sort en 1995 puis **Catz**, des animaux vivent sur l'écran et développent des comportements. Dans **Creatures** [https://en.wikipedia.org/wiki/Creatures\_(video\_game\_series)](https://en.wikipedia.org/wiki/Creatures_%28video_game_series%29), les "Norns" démontraient une ébauche de machine learning et étaient présentées comme des créatures autonomes et une première forme d'*Artificial Life*. Dans "fin fin: On TEO, the Magic Planet", l'utilisateur pouvait même interagir avec l'animal virtuel par la voix. Par ailleurs, des **Screensavers** gratuits caractérisés par des personnages et d’autres animaux virtuels s'échangent sur les réseaux au début d’internet. Enfin de très nombreux sites web d’ animaux virtuels ont été mis en ligne dans les années 2000\. Voir [https://virtualpet.com/](https://virtualpet.com/)

Le monde de la recherche s’est intéressé à ces simulations d'animaux artificiels. L'équipe de "Virtual Petz" (auteurs de Catz, Dogz, ...) publia *Socially Intelligent Virtual Petz*: [https://cdn.aaai.org/Symposia/Fall/1997/FS-97-02/FS97-02-010.pdf](https://cdn.aaai.org/Symposia/Fall/1997/FS-97-02/FS97-02-010.pdf) ; les auteurs de Creatures publièrent également des articles. Les desktop pets ont été pris en exemple dans des revues sur l’Artificial Life, à côté d’autres concepts plus sérieux comme les **animats**. Assez récemment, des chercheurs ont même développé leur propre Desktop pet:

- [https://edwardyi.me/pdf/Michitop\_Paper.pdf](https://edwardyi.me/pdf/Michitop_Paper.pdf)
- [https://www.researchgate.net/publication/224333547\_Design\_Development\_of\_a\_Virtual\_Pet](https://www.researchgate.net/publication/224333547_Design_Development_of_a_Virtual_Pet)

### Desktop Pets ──► Animats

Les *desktop pets* antérieures ont des comportements largement scriptés. Le *Machine Learning* est très partiel voire totalement contrefait: des compétences innées sont déguisées en compétences acquises.

Ce document propose de concevoir un nouvel animal dont le comportement soit réellement façonné par l’expérience plutôt que d’être pioché dans une collection de routines pré-codées.

Pour cela, le comportement de l’animal est produit par un réseau neuronal alimenté par des perceptions (vue, ouïe, …) et des états internes (fatigue, peur, …). Ce réseau neuronal apprend continuellement à partir de l’expérience, notamment par les **interactions avec son propriétaire**.

L'utilisateur joue alors le rôle du **dresseur**. Il ne programme pas l'animal mais lui indique ce qui est souhaitable par renforcement (récompense / punition). L'animal initialement peu compétent mais soumis à des interactions quotidiennes, finit par développer un comportement avancé et unique.

L'apprentissage *continu* (CL) est une caractéristique essentielle de l’architecture retenue. L'expérience vécue devient un corpus d’apprentissage, non connu à l’avance (on parle de données d’apprentissage streamées).

Cette idée a même fait l’objet d’articles de recherche, mais l'apprentissage ne s’applique généralement qu'à une dimension réductrice du comportement, comme les "émotions" (dixit). De plus, **il n'y a pas de projet actif connu qui corresponde à cette idée d’animal apprenant**. Par exemple, les projets indexés sous le terme « virtual pet » ([virtual-pet · GitHub Topics · GitHub](https://github.com/topics/virtual-pet)) ou « desktop pet » dans Github sont essentiellement des représentations graphiques d'agents LLM avec des animations scriptées.

A noter que quelques projets proposent une forme d'apprentissage rudimentaire. Par exemple, dans *Dosidicus*, un réseau d’une dizaine de neurones matérialisant des concepts abstraits (des intentions et des situations scriptées) apprend en suivant les règles de Hebb.

## Le Projet "Lambda barre"

Le projet **λ̄** (lambda avec une barre suscrite) est un projet *d'animat en tant que desktop pet*, autrement dit un animal virtuel qui habite l'espace du bureau. Des intéractions sont possibles avec la souris (pointage, caresse), les fenêtres et le clavier. Son comportement est acquis par apprentissage automatique (machine Learning).

λ évoque le **lambda-calcul** (le cadre conceptuel de l'évaluation de programme) mais aussi l'expression **individu lambda**: un individu sans personnalité particulière mais qui acquiert de l'expérience.

**λ̄** peut alors évoquer *l'individu lambda* matérialisé comme entité artificielle.

En Unicode, le symbole **λ̄** est obtenu avec λ \+ le caractère Unicode **COMBINING MACRON** (U+0304); En LaTeX, par : \\bar{\\lambda} ou \\overline{\\lambda}.

## Architecture du système de décision apprenant

Voici une proposition d'architecture du système de décision (production d'actions) et d'apprentissage artificiel continu.

  Environnement
   correction \+ / \- utilisateur
       │
       ▼
Observations ──► Encodeur ──► Modèle du monde
       │                     (prédit conséquences)
       │                            │
       ▼                            ▼
État latent ───────────────► C(s,a) coût prédit
       │                            ▲
       ▼                            │
 Politique π(a|s) ──────────────────┘
       │
       ▼
     Action
       │
       ▼
 Environnement / desktop pet
       │
       └──────────────► nouvelles observations

### Le World Model

Le **world model** est un **modèle personnel du monde** qui apprend que « le monde fonctionne ainsi ». Il apprend également que « l'utilisateur réagit comme ça » puisque les réactions de l'utilisateur font partie de l'environnement.

Cela peut être typiquement un **Transformer autorégressif** entraîné sur la séquence cyclique état\>action... Autrement dit: S1, A1, S2, A2, S3, A3 … où Si est un état du monde sensoriel et Ai une action.

Le world model apprend: P(s(t+1) | s(t), a(t))

### La fonction de coût

Chaque état en entrée du modèle est associé à un score de renforcement: positif pour une récompense, négatif pour une punition.

L'animal a un système inné de récompenses lié à ses besoins: l'ennui, la curiosité, la socialisation, la recherche de sécurité, l'attachement, etc. Ce renforcement par le système permet à l’animal d’apprendre des comportements *instinctifs*.

Par exemple, un signal de “confort postural” incite l’animal à prendre une position réaliste et un signal de “courbatures”, à se déplacer régulièrement.

Ainsi le renforcement induit par l'utilisateur ne fait que s'ajouter aux **propres motivations** de l'animal.

Le world model est donc un **reward model** puisqu'il apprend aussi qu'elle action débouche à terme sur une récompense ou sur une punition.

Autrement dit, La fonction de coût d'une action C(s(t), a(t), s(t+1)) sera obtenue en sommant les récompenses/punitions à court terme comme à long terme de la trajectoire prédite par le modèle.

Par exemple, si le modèle apprend que voler quelque chose produit une récompense intrinsèque immédiate suivie d’une punition encore plus importante par l'utilisateur, alors l'action du vol sera prédite comme pénalisante.

### La politique

La politique RL π(action | state) est un second modèle chargé de choisir l'action immédiate qui va minimiser le coût prédit par le world model.

Le modèle de la politique apprend à éviter les actions ayant un impact négatif **avant d'en faire l'expérience**.

Il apprend à partir de **l'anticipation des conséquences futures** (la trajectoire) produite par le word model.

### Apprentissage continu

Les 2 modèles apprennent en continu.

* Le world model est entraîné à intervalle régulier lors de phases d'apprentissage classique, pendant le "sommeil"/le "rêve" de l'animal.
* La politique est elle corrigée immédiatement par RL, d'après la fonction de "coût prédit" obtenue du world model.

### Un transformer multimodal

Le world model apprend l'enchaînement temporel de « tokens » représentant des états et des actions.

Les **tokens d'état** encodent l’état de son environnement sensoriel:

**Proprioception** → « Dans quel état est mon corps ? »

* par exemple la position de l'animal
* associé à un score de renforcement: une douleur ou un plaisir immédiat

**Intéroception** → « Dans quel état interne suis-je ? » (faim, fatigue, douleur, excitation, etc.)

* également associé à un inconfort ou à un plaisir, mais souvent sous la forme d’un signal de signaux de renforcement synthétisés: la fatigue est un cumul d’effort, la souffrance un cumul de douleurs, etc.
* peut matérialiser des motivations intrinsèques comme la curiosité

**Extéroception** → « Que se passe-t-il autour de moi ? »

* observation de l'environnement visuel, sonore, tactile, y compris les interactions utilisateur

Les **tokens d'actions** encodent l’activation d’articulations (mouvement), de vocalises et d’autres actuateurs.

Les tokens sont formés en discrétisant (quantizing) les valeurs continues de l’espace des états et des actions. C’est essentiel pour prédire une distribution de trajectoires probables qui soit “multi-modale” (distinctes, hétérogènes, …) plutôt qu’une distribution gaussienne centrée autour d’un futur “moyen”.

Le world model est donc un Transformer qui prédit le token suivant. Ce n'est pas un LLM mais ça reste un **Transformer autorégressif** qui reçoit par exemple état → action → conséquence (nouvel état observé) → action → conséquence... et qui apprend à prédire une trajectoire du futur.

On parle aussi de **Trajectory Transformer** ou de **Decision Transformer** étant donné sa raison d'être.

Projets github conceptuellement proche:

* **Trajectory Transformer**. Le projet transforme une trajectoire en séquence de tokens représentant **états, actions et récompenses**, puis entraîne un Transformer autorégressif à modéliser ces séquences. Le modèle peut ensuite servir à planifier des actions en prédisant les conséquences futures. ([GitHub \- jannerm/trajectory-transformer: Code for the paper "Offline Reinforcement Learning as One Big Sequence Modeling Problem" · GitHub](https://github.com/jannerm/trajectory-transformer))
* **TWISTER**. Il apprend un Transformer comme *world model* qui prédit les états futurs et les récompenses conditionnellement aux actions, puis entraîne un acteur et un critique dans l'espace latent en utilisant des trajectoires imaginées par le world model. ([GitHub \- burchim/TWISTER: \[ICLR 2025\] Learning Transformer-based World Models with Contrastive Predictive Coding (TWISTER) · GitHub](https://github.com/burchim/TWISTER))
* **Transformer-based World Models (TWM)**. un Transformer apprend directement la dynamique d'un environnement à partir d'interactions, et sert ensuite de modèle du monde pour le RL. ([GitHub \- jrobine/twm: Transformer-based World Models · GitHub](https://github.com/jrobine/twm))
* **Robotic World Model** entraîne simultanément un modèle de dynamique et une politique, puis entraîne la politique sur des trajectoires « imaginées » par le modèle appris. ([GitHub \- leggedrobotics/robotic\_world\_model: Repository for our papers: Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics and Uncertainty-Aware Robotic World Model Makes Offline Model-Based Reinforcement Learning Work on Real Robots · GitHub](https://github.com/leggedrobotics/robotic_world_model))

A noter que contrairement à des **world-model RL** modernes, le « reward model » d'un animal virtuel apprenant, ne doit pas être explicite. C'est le résultat de motivations intrinsèques et d'observation **d'encouragements/réprimandes de l'utilisateur** ainsi que de leurs effets. L’**Animat apprend progressivement son propre monde**, puis utilise ce modèle pour évaluer les conséquences de ses actions sans les effectuer.

## Apprentissage de mouvements

Dans un environnement graphique 2D en **vue de côté**, l'animat est représenté par un corps constitué de segments articulés: : Un tronc triangulaire, 2 membres postérieurs, une queue, une tête, et 2 pattes avant. Le tout ressemble à un kangourou dessiné dans un style géométrique abstrait. Le rendu pourrait être réalisé par un moteur 2D de type SVG.

Ce corps évolue dans une simulation physique 2D réelle. Les actions (forces, couples, ...) mettent le corps en mouvement en tenant compte de la masse du tronc, la gravité, la friction avec le sol, les collisions et l'inertie.

L'apprentissage de l'animat ne se contente pas d'invoquer des mouvements scriptés (« saut », « marche », « recul »): les **mouvements eux-mêmes** sont des comportements émergeants.

Par exemple, l'utilisateur encourage l'animal à atteindre une plateforme, et le réseau découvre la séquence de commandes motrices pour y parvenir en sautant sans que le **« saut » n'existe nulle part dans le programme initial**. **Le RDN ne sait même pas ce qu'est une jambe.**

Le modèle produit les commandes motrices des **actuateurs** du squelette:

* Actuateur sur chaque membre postérieur avec pour paramètre l'angle et la distance au point d'attache.
* Actuateur de la queue contrôlée par un seul angle.

Il n'y a pas de coude/genou : un membre est un unique segment positionné par rapport au tronc. La queue avec son unique degré de liberté sert de contrepoids.

L’angle du corps par rapport à l’axe vertical détermine l’orientation du personnage: Les segments du corps esthétiques (tête avec oreilles animées, pattes avant, ...) sont dessinés en conséquence ne sont pas contrôlés. Cette morphologie / biomécanique simplifiée permet l'émergence de mouvements riches tout en limitant la complexité de l'apprentissage.

Les mouvements et la posture calculés par le moteur physique sont réinjectés en entrée du modèle sous forme de signaux *proprioceptifs*.

* orientation du tronc.
* angle et distance de chaque membre
* vitesse angulaire/relative
* accélérations
* forces ou couples exercés par les actuateurs
* force de contact avec le sol

Mais le RDN doit optimiser par renforcement ses mouvements pour les rendre **efficaces et confortables**.

Ces signaux sont associés à des coûts intrinsèques calculés par le système :

* énergie dépensée par les actuateurs, autrement dit l'effort musculaire.
* l'inconfort dû à l'instabilité du corps,
* les collisions/frictions.

Un autre coût intrinsèque s'ajoute au coût physique des mouvements: le **renforcement postural**. Ce coût inné encourage certaines **postures** « naturelles » : se redresser sur ses pattes arrières, se coucher, etc.

Ce renforcement postural forme un biais instinctif en faveur de certaines postures, mais sans encoder explicitement des connaissances comme « la queue sert à équilibrer. » car on souhaite que de telles connaissances soient découvertes.

**Le réseau doit en particulier apprendre à lutter contre la gravité et l'instabilité**. Au départ, λ̄ tombe simplement. Puis il découvre une configuration d'actions qui lui permet de rester debout, puis de déplacer son centre de gravité, et enfin de se propulser.

Il sait seulement : « J'ai ces actuateurs, ces perceptions d'état, et certaines actions ont historiquement produit certaines conséquences. » Puis le coût système lui apprend que certaines configurations sont énergivores, ou inconfortables. La « démarche » de λ̄ est donc une compétence acquise.

### Modalités de perception

Rappel:

**Proprioception** → « Dans quel état est mon corps ? »

**Intéroception** → « Dans quel état interne suis-je ? » (faim, fatigue, douleur, excitation, etc.)

**Extéroception** → « Que se passe-t-il autour de moi ? »

**L'environnement sensoriel de λ̄** est une version **minimaliste** de celui d'un animal dans le monde réel.

L'animat ne perçoit pas des abstractions informatiques discrètes, comme « fenêtre », « bouton », « souris » ou « clavier » mais un petit nombre de **valeurs** perçues en respectant l'**égocentricité** : quand λ̄ se déplace, ce monde sensoriel se déplace avec lui. Quand λ̄ se tourne (angle du tronc par rapport à la vertical qui change de signe), le champ visuel se tourne aussi.

#### Perception visuel des couleurs

La vision est une **vision rétinienne très basse résolution** à courte portée.

Le cône de vision est un triangle dans le prolongement de la tête vers la gauche ou la droite en fonction de l’orientation du corps. Ce triangle est découpé en 16 cellules disposées en 4×4; pour chaque cellule, une couleur moyenne HSV \+ mesure d'hétérogénéité est calculée.

Bien que rudimentaires, ces informations synthétiques permettent de distinguer :

* une surface uniforme ;
* un bord de fenêtre ;
* un objet coloré ;
* du mouvement.

#### Perception des mouvements visuels

Le moteur graphique calcule le différentiel spatial des fenêtres.

* surface du différentiel
* dx, dy du barycentre du différentiel, dans le référentiel de la tête

Cela restitue un **flux optique grossier** qui permet à λ̄ de détecter par exemple que « quelque chose de gros bouge à gauche ».

#### Perception du curseur de la souris

* dx, dy position dans le référentiel de la tête ;
* vx, vy vecteur vitesse dans ce même référentiel.

#### Perception du toucher

Le tronc et les 2 membres postérieurs sont des capteurs tactiles.

Sensation tactiles

* contact sol / pied: valeur binaire
* force normale: valeur
* collision avec le bord des fenêtres / objets: vecteur force dédié collision (autre que sol)

Inconfort d'immobilité

L'immobilité prolongée produit une sensation d'inconfort avec un coût intrinsèque croissant. Ce renforcement système vise à encoder l'instinct biologique qui pousse au mouvement.

#### Audition

Proposition de champ auditif simulé à partir d'événements:

* clic gauche/droit ;
* touche clavier ;
* notification/d'événement système ;
* fenêtre qui apparaît/disparaît.

Ces sont sont perçus par un triplet:

* **angle** : direction de la source dans le référentiel de la tête ;
* **intensité** : proportionnelle à la force du signal et à la distance de la tête. Pour des clics de souris, la force du signal encode le CPS
* **fréquences** : vecteur 2D des caractéristiques de la source: clic, touche, évènement...

L'intensité du son perçu dépend de la distance I \= I\_0 / (1+d^2) La fréquence encode notamment la position horizontale d'un appui de touche de clavier. Fréquence **2D**, ie. deux composantes fréquentielles :

→ fréquence X correspondant à sa colonne
→ fréquence Y correspondant à sa ligne

#### Vocalises

Nous avons vu les actuateurs du corps.

Le générateur de cri de l'animal est conçu selon le même principe: le réseau produit les **paramètres d'un actuateur vocal** plutôt que l'identifiant d'un son préfabriqué.

Les paramètres de l'actuateur vocal sont ceux d'un **petit synthétiseur basique**:

* fréquence fondamentale f₀ ;
* seconde fréquence
* amplitude ;
* durée ;
* timbre ;
* éventuellement enveloppe ADSR ;

**La voix devient un comportement appris** par renforcement.

Le signal *proprioceptif* d'audition permet à l'animat de percevoir sa propre voix.

Un plaisir inné instinctif est calculé par le système et fournit en entrée du modèle afin d'encourager par renforcement

* la production de vocalises
* l'imitation des sons perçus.

L'instinct **d'imitation** est un **renforcement** par le système plutôt que par l'utilisateur. L'animal n'a pas besoin de comprendre ce qu'est une « chanson ». Il possède simplement une disposition innée à **réduire l'écart entre ses propres vocalisations et certains sons perçus**.

Pour cela, le système calcule l'entropie informationnelle de la séquence sonore perçue pour en faire un signal d'apprentissage efficace du principe d'imitation.

séquence sonore produite par l'utilisateur
       ↓
perception auditive
       ↓
politique vocale
       ↓
séquence sonore produite par l'animal
       ↓
récompense de la vocalisation
perception auditive de sa propre voix
coût proportionnel à l'entropie informationnelle des sons récents obtenus
       ↓
mise à jour

On a donc deux niveaux de récompenses :

1. **Reward système** : la production d'un chant et la similitude avec un chant entendu est récompensé (instinct).
2. **Reward humain** : l'utilisateur encourage ou réprimande la reproduction.

Le premier est un **biais inné vers l'imitation** ; le second apprend **le comportement autour de cette compétence**.

Le biais inné vers l'imitation peut diminuer avec le temps. Comme avec les oiseaux, l'animat apprend des motifs musicaux quand il est jeune. Plus tard, l'environnement et les interactions sociales prennent davantage de poids.

Ici, **l'instinct n'est pas un comportement codé, mais une fonction de coût innée qui orientent l'apprentissage.**

### Encodage de la perception

On ne concatène pas tous les scalaires en un vecteur unique. Le système encode **chaque modalité en une séquence d'embeddings** fournis en entrée du transformer.

Par exemple, à un instant donné :

\[PROPRIO\]
  tronc\_angle
  tronc\_vitesse
  membre\_AV\_angle
  ...

\[VISION\]
  cellule\_1
  cellule\_2
  ...
  cellule\_16

\[AUDIO\]
  source\_1
  source\_2
  ...

\[TOUCH\]
  patte\_AV
  patte\_AR
  ...

\[INTERO\]
  faim
  fatigue
  ...

\[ENV\]
  curseur
  mouvement\_fenêtres

Le Transformer reçoit l'embedding de chaque groupe tel qu'il est produit par **un petit encodeur système**. Les groupes sont délimités par des tokens symboliques: PROPRIO, VISION, AUDIO, ...

Les modalités peuvent être perçues à des fréquences différentes. La proprioception pourrait être cadencée 10 fois plus vite que la vision.

### Consolidation de l'apprentissage durant le sommeil

L'apprentissage continu du world model est réalisé durant les phases d'inactivité de l'animat (sommeil).

Le world model n'apprend pas immédiatement de chaque nouvelle expérience. Le système accumule cette expérience dans un nouveau corpus puis fait **N passes d’apprentissage offline** (N=10, 100, ?) durant un cycle de sommeil.

On va distinguer les expériences anciennes et  les  expériences récente:

* Les anciennes expériences sont conservées dans un *coreset*.
* Les expériences nouvelles sont enregistrées dans un *addendum*.

De fait, pendant l'éveil, on journalise l'expérience vécue (séquences états-actions) :

      perception → action → conséquence
                    ↓
      enregistrement des expériences vécues de la journée

Puis, pendant le sommeil, on entraîne les modèles avec une combinaison du corpus existant (*coreset*) et de l’addendum :

         dataset complété avec les nouvelles expériences
                           ↓
                replay × plusieurs epochs
                           ↓
            mise à jour des modèles

Enfin le coreset est complété avec les nouvelles expériences de façon **cumulative**.

Mais on ne peut pas ajouter de nouvelles expériences dans le corpus d'apprentissage de jour en jour sans stratégie. D’une part, il faut maîtriser la taille finale du dataset. On ne peut pas conserver indéfiniment l’expérience vécue de chaque journée. On ne peut pas non plus supprimer du dataset des séquences sous prétexte qu’elles sont déjà connues, car cela déboucherait progressivement sur du **catastrophic forgetting**.

D’autre part, il faut équilibrer la représentation des différentes séquences à apprendre pour un comportement donné: une surreprésentation du même type de séquence favoriserait l’apprentissage d’un comportement arbitraire au dépend des autres.

Il faut d’abord filtrer les **expériences vécues de la journée** et ne conserver que les séquences ayant déjouées le world model existant, c'est-à-dire celles qui s'écartent de la trajectoire prédite. Ce calcul d’écart doit être amplifié par le signal de renforcement. Il s'agit de privilégier l'entraînement sur les séquences qui produisent une erreur élevée de l’estimation de la fonction de coût.

Autrement dit, une expérience n'est conservée que si le modèle courant se trompe sur la trajectoire du futur (événement inhabituel) et que les enjeux de cette erreur sont importants (écart sur l'estimation du coût induit). Par exemple, une séquence où λ̄ marche tranquillement dans son environnement ne produit **aucun nouvel exemple d'entraînement**.

Ensuite, il faut régulièrement rééquilibrer le dataset. Pour cela, avant l'apprentissage nocturne, 5% des séquences sont retirées du corpus *coreset* et déplacées dans *l’addendum* et soumis au même filtrage que des nouvelles expériences.

Puis l'apprentissage se produit en deux temps:

* D'abord on entraîne le modèle sur le coreset.
* Puis l'addendum est filtré avec le nouveau modèle obtenu.
  * On calcule la probabilité pondérée de chaque séquence de "addendum"
  * On compare son gain cumulé avec celui de la trajectoire prédite
  * Si la séquence est significativement probable et que l’écart entre le gain prédit et le gain observé et faible, la séquence est retirée.
* puis on entraîne le modèle avec "addendum"
* et enfin on ajoute "addendum" à "coreset" pour la prochaine phase d'apprentissage.

Le filtrage d'addendum permet de retirer du dataset final:

* l'expérience de la journée qui est déjà connue et donc inutile à conserver
* la partie des 5% du contenu du coreset qui s'avère être du contenu redondant (puisque correctement prédit avec l’entraînement sur le coreset allégé uniquement)

#### Bilan

Journée:

1. coreset\= dataset ayant servi au dernier apprentissage, inchangé.
2. Addendum \= dataset des expériences de la journée journalisée, grossit à partir de 0\.

Sommeil:

* On retire aléatoirement du corpus coreset 5 % des séquences que l'on déplace dans Addendum ;
* entraînement sur le coreset allégé ;
* Calcul de la probabilité  de chaque séquence de Addendum ; de plus, on compare le gain espéré de la trajectoire prédite avec celle observée: si la séquence est connue (probable) et que le gain est correctement prédit, elle disparaît ;
* À la fin, Addendum contient donc les informations du jour qui **résistent à la connaissance déductible du coreset** ainsi que **les connaissances de 5% du coreset proposés à l’oubli et qui s’avèrent pertinentes**.
* On entraîne sur cet addendum
* Pour finir: Foundation ← Foundation ∪ Addendum

L'idée forte pour maîtriser la taille et l’équilibrage du corpus est celle des **5 % candidatés pour l'oubli**. Pour savoir si des données anciennes peuvent être **oubliées**, on fait à chaque apprentissage un test expérimental sur 5% du corpus existant. Pour cela,

* On retire volontairement du corpus une partie de ce que le modèle apprend.
* On regarde ce qu'il est capable de reconstruire.
* Si un exemple retiré reste correctement prédit après réentraînement sur le dataset simplifié, c'est qu'il était redondant.
* s'il ne l'est pas, il contient une information qui n'est plus suffisamment représentée, et on le remet dans le corpus.

Note: 5% est une valeur arbitraire. En pratique, la proportion du dataset proposée à l'ablation est calculée pour que **le dataset reste de taille pratiquement constante**.

D'après ChatGPT, c'est une forme de **test de nécessité par ablation**, une **sélection adaptative des données nécessaires à maintenir les compétences du modèle**, du **rehearsal par ablation et consolidation**.

il y a des méthodes proches:

"L'*experience replay* conserve un sous-ensemble des anciennes expériences pour limiter l'oubli catastrophique, avec des stratégies comme reservoir sampling, herding ou sélection basée sur les gradients. ([GCR: Gradient Coreset Based Replay Buffer Selection for Continual Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Tiwari_GCR_Gradient_Coreset_Based_Replay_Buffer_Selection_for_Continual_Learning_CVPR_2022_paper.pdf?utm_source=chatgpt.com)) Certaines méthodes sélectionnent même les exemples qui subissent le plus d'interférence lorsqu'on apprend les nouvelles données, ce qui est conceptuellement proche de votre idée. ([GCR: Gradient Coreset Based Replay Buffer Selection for Continual Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Tiwari_GCR_Gradient_Coreset_Based_Replay_Buffer_Selection_for_Continual_Learning_CVPR_2022_paper.pdf?utm_source=chatgpt.com))"

Les briques sont connues — rehearsal, exemplar selection, continual learning — mais cette combinaison précise mérite effectivement d'être testée. ([Saliency Guided Experience Packing for Replay in Continual Learning](https://openaccess.thecvf.com/content/WACV2023/papers/Saha_Saliency_Guided_Experience_Packing_for_Replay_in_Continual_Learning_WACV_2023_paper.pdf?utm_source=chatgpt.com))

**Références fournies par ChatGPT:**

1. [GCR: Gradient Coreset Based Replay Buffer Selection for Continual Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Tiwari_GCR_Gradient_Coreset_Based_Replay_Buffer_Selection_for_Continual_Learning_CVPR_2022_paper.pdf?utm_source=chatgpt.com)
2. [Saliency Guided Experience Packing for Replay in Continual Learning](https://openaccess.thecvf.com/content/WACV2023/papers/Saha_Saliency_Guided_Experience_Packing_for_Replay_in_Continual_Learning_WACV_2023_paper.pdf?utm_source=chatgpt.com)

### Délimitation des séquences du dataset

C'est le même genre de problème que le chunking du RAG : **la bonne unité de mémoire n'est pas évidente a priori**.

Par exemple, si λ̄ marche pendant 30 secondes sans événement notable :

marche → marche → marche → marche → ...

Il ne faut pas stocker ces 30 secondes.

Mais si ensuite :

clic → λ̄ tourne la tête → avance → tombe

Cela constitue un épisode intéressant.

Le problème devient alors : **comment déterminer automatiquement les frontières d'un épisode ?**

Une possibilité est de laisser le world model décider. Tant que :

\[ L\_t \= \-\\log P(o\_{t+1},r\_t|o\_t,a\_t) \]

reste faible, on est dans une situation qu'il comprend. Lorsqu'elle augmente fortement, on ouvre un épisode autour de cet événement :

     erreur faible
───────────────┐
               │
               ▼
          événement
          surprenant
               │
               ▼
      ┌─────────────────┐
      │     épisode     │
      │ avant \+ pendant │
      │ \+ après         │
      └─────────────────┘

Il faut conserver **un peu de contexte avant l'erreur**, sinon le modèle risque de voir uniquement la conséquence sans savoir ce qui l'a provoquée.

Le chunk doit préserver suffisamment de **contexte causal** pour être utile.

WIP. TODO.

### Entraînement de la politique

Le **world model** répond à :

« Si je fais ça, qu'est-ce qui va probablement se passer ? »

La fonction de coût/reward est la somme des coûts (cumul du signal de renforcement) sur la trajectoire prédite par le *world model*. Elle sert à évaluer :

**« Est-ce que ce résultat est désirable ? »**

La **politique** répond à :

« Maintenant, qu'est-ce que je dois faire ? »

Concrètement :

                 état actuel
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
       WORLD MODEL          POLITIQUE
             │                 │
       état \+ action           │
             │                 │
             ▼                 ▼
       conséquence            action
       prédite                 │
             │                 │
             └───────┬─────────┘
                     ▼
                  action
                     │
                     ▼
                environnement
                     │
                     ▼
                nouvel état

Le world model est donc une sorte de **simulateur appris du monde**.

Par exemple, il pourrait avoir appris :

état \= debout, curseur à gauche
action \= tourner la tête à gauche
        ↓
prédiction :
  tête tournée à gauche
  curseur maintenant dans le champ visuel
  coût moteur faible

La politique, elle, produit directement les commandes :

état actuel
    ↓
politique
    ↓
couple tête \= \+0.31
couple queue \= \-0.04
couple patte \= ...

Pourquoi avoir les deux ? Parce que la politique peut apprendre par essais :

« J'ai fait X → j'ai obtenu une bonne récompense. »

Mais elle ne sait pas forcément **pourquoi**.

Le world model permet de faire des essais **dans sa tête**.

Supposons que la politique envisage trois actions :

A : avancer
B : reculer
C : tourner

Le world model peut simuler :

A → collision, coût élevé
B → aucun changement
C → curseur visible, récompense probable

La politique choisit alors C, sans avoir besoin d'essayer réellement A et B.

C'est ce qu'on appelle du **model-based reinforcement learning**.

Donc les trois fonctions sont distinctes :

WORLD MODEL
état \+ action → conséquence prédite

REWARD MODEL / coûts
conséquence → valeur

POLICY
état → action

Le world model prédit des conséquences sur des actions imaginaires, et la politique de choisir les actions qui donnent de bonnes conséquences **sans que λ̄ ait réellement effectué toutes ces actions**.

Il y a en fait trois fonctions :

                état \+ action
                      │
                      ▼
                WORLD MODEL
                      │
                      ▼
       futur prédit : s₁,r₁,s₂,r₂,...,sₙ,rₙ
                      │
                      ▼
              somme des coûts
                      │
                      ▼
                  C(action)
                      │
                      ▼
                 POLITIQUE
                      │
                      ▼
                   action

La politique ne reçoit pas directement un « reward » pour apprendre. Elle apprend à produire des actions qui **minimisent le coût prédit par le world model**.

La partie importante est que **le world model ne prédit pas seulement la prochaine récompense**. Il prédit une trajectoire et le coût final est obtenu en regardant toute cette trajectoire.

### Mise en oeuvre

Box2d-like pour la physique: gravité, friction, contrainte. Les parties rigides du corps de l'animal sont jointes en dur, mais les liaisons sont éditées en continue par les actuateurs, par exemple distance et angle d'un membre.

Le joint d'un membre a deux commandes continues :

\[ a\_i=(\\theta\_i,d\_i) \]

où (\\theta\_i) est l'angle et (d\_i) la distance au tronc. Le RDN produit directement ces valeurs, puis le moteur physique calcule les forces et mouvements résultants.

Python
 ├── Box2D / pymunk
 │      └── squelette \+ environnement
 │
 ├── PyTorch
 │      ├── policy
 │      └── world model
 │
 ├── dataset / replay
 │
 └── petite visualisation 2D

Une fenêtre montre le squelette et quelques éléments du bureau. Pas besoin de reproduire immédiatement le vrai desktop.

Pour la fluidité de l'animation et de la simulation, les transitions de valeurs distance et angle ne sont pas immédiates.

L'actuateur est une **commande de consigne**, et non une contrainte imposée.

Par exemple, le réseau produit :

\[ a\_i=(\\theta\_i^\*,d\_i^\*) \]

et l'actuateur transforme cela en force/couple :

\[ \\tau\_\\theta \= k\_\\theta(\\theta^\*-\\theta)-c\_\\theta\\dot\\theta \]

\[ F\_d \= k\_d(d^\*-d)-c\_d\\dot d \]

Donc le réseau dit essentiellement :

« Je voudrais que ce membre soit dans cette configuration. »

et non :

« Mets-le immédiatement dans cette configuration. »

Le moteur physique détermine ensuite ce qui arrive réellement.

Au minimum, il faut un récepteur simple (eg souris), des boutons récompense/punition, et on essaye de faire apprendre "suivre la souris".

L'environnement pourrait être extrêmement pauvre :

* animal 2D avec squelette ;
* gravité \+ collisions ;
* souris comme unique exteroception ;
* un bouton « récompense » ;
* un bouton « punition » ;
* coûts physiques innés ;
* politique RDN ;
* world model ;
* Apprentissage uniquement pendant le sommeil.

La souris fournit seulement :

position relative à la tête
vitesse relative

L'objectif émergent serait simplement d'apprendre **à suivre le curseur.**

Le bouton utilisateur serait le seul signal externe. La récompense est simplement un **événement/token du flux sensoriel**, au même titre que la souris. Plus précisément, le bouton n'est pas un « signal d'apprentissage » spécial, mais un **événement sensoriel à valeur intrinsèque**. Ainsi, **le reward n'est pas une supervision spéciale du réseau**. C'est un événement du monde de λ̄, dont la signification temporelle doit être apprise par le modèle.

Au final

* **événement** : ce qui arrive ;
* **valence** : agréable/désagréable, portée directement par l'événement ;
* **coût cumulé** : conséquence sur une trajectoire ;
* **politique** : comportement qui finit par favoriser les trajectoires à faible coût.

Donc, au départ, λ̄ ne sait pas que « suivre » est un concept.

Si l'utilisateur récompense systématiquement les situations où λ̄ se rapproche de la souris et punit celles où il s'en éloigne, **la tendance à suivre devrait idéalement émerger**.

Il y a même un test intéressant : après l'apprentissage, l'utilisateur arrête de récompenser. Si λ̄ continue spontanément à suivre la souris, vous avez effectivement obtenu un comportement acquis plutôt qu'une simple réaction au bouton.

Le mécanisme Coreset/Addendum devient testable après un cycle activité/sommeil complet.

Les coûts innés sont similaires conceptuellement. Une posture pénible pourrait produire en permanence un événement interne de type :

\[DISCOMFORT \+0.17\]

alors qu'une posture confortable produirait :

\[COMFORT \-0.05\]

Le réseau n'a donc finalement qu'un seul langage d'évaluation : **des événements valenciés dans le temps**.

### Granularité temporelle du système et compression des actions

L'idée: **décorréler la fréquence de la politique de la fréquence du world model**.

Le world model peut travailler à un cadence beaucoup plus lente que la politique.

Imaginons une cadence de 2s pour le world model. Pendant 2 secondes, le système collecte :

proprioception(t)
actions(t)
proprioception(t+100ms)
actions(t+100ms)

...
proprioception(t+2s)

Le système produit un seul groupe de **tokens de transition** pour le world model :

état initial
\+
résumé des actions
\+
état final
\+
événements survenus

Le world model apprend alors :

\[ (s\_t,; A\_{t:t+\\Delta}) \\rightarrow s\_{t+\\Delta} \]

où (A) n'est pas une action instantanée mais **un résumé de l'activité motrice pendant l'intervalle**.

Ainsi, marcher pendant 2 s devient un seul élément du contexte du world model.

Toutefois, les événements ponctuels qui ne sont pas compressibles doivent pouvoir interrompre la transition.

Si λ̄ marche tranquillement :

10Hz × 200 ms

→ un seul résumé.

Mais si pendant cet intervalle :

marche
marche
marche
\[CLIC\]
tête tourne
\[REWARD\]

alors les événements CLIC et REWARD doivent être conservés individuellement.

On aurait donc une séquence du genre :

état
   ↓
\[bloc moteur 1200 ms\]
   ↓
\[CLICK\]
   ↓
\[bloc moteur 800 ms\]
   ↓
\[REWARD \+1\]
   ↓
état

Les mouvements ordinaires sont compressés, tandis que les événements significatifs deviennent des tokens individuels.

Ainsi, le world model apprend à son échelle temporelle.
