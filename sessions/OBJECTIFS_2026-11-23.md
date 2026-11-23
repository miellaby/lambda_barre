    MODURATITY.md

# Instinct de jeu et autres instincts

Le jeu (ou plutôt les jeux) est un véritable instinct. Qu'est-ce qu'un instinct? une surcharge des actions de la politique par des actions instinctives, c'est à dire des actions issues de comportements stéréotypés, autrement dit de comportements "scriptés".

Complexifier le circuit du renforcement ne suffira pas à mettre en place l'instinct de jeu.  Certes, mon objectif est que, dans la mesure du possible, les comportements émergent du renforcement. Mais il faut pour celà qu'il y est des essais, des erreurs et des réussites enregistrés dans le dataset. Or c'est précisément des instincts comme l'instinct de jouer qui vont servir à l'enrichissement du corpus d'apprentissage.

Les instincts et l'activation des instincts sont scriptés. De nouveaux états physiologiques sont introduits à cet usage (aggressivité, soumission, etc) sans forcément une proprioception associée (état strictement interne). A tout moment, un instinct est actif (un seul à la fois), et cet instinct calcule une action.

On encode dans chaque salve l'instinct en cours (par exemple "poursuite") dans un nouveau token.

On encode également dans un nouveau token dédié l'action instinctive calculée (script) par l'instinct en cours.

Ce sont des tokens d'états. Le WM va pouvoir apprendre l'influence de l'instinct sur la trajectoire.

Si l'instinct l'emporte, l'action policy est donc remplacée par l'action instinctive de la même façon que les contrôles utilisateur peuvent actuellement surcharger la policy.

Mais ce court-circuit doit être modulé par la policy elle-même, qui choisit de "lâcher prise" et de "laisser libre court à ses instincts" ou non. La policy arbitre.

En plus des consignes motrices, un nouveau canal d'actuateur totalement contrôlé par la policy permet donc de moduler entre les consignes de l'action instinctives et celles de l'action policy.

Les consignes effectivement appliqués sur les actuateurs sont finalement, soit l'action instinctive, soit l'action apprise, soit une interpolation des deux.

L'entrainement apprend donc à générer des consignes motrices volontaires mais aussi à décider entre les appliquer ou laisser l'instinct l'emporter.

Un canal de renforcement récompense le suivi des actions instinctives: le canal de confort. Actuellement le confort est bloqué sur une posture constante et neutre. Dorénavant, la posture désirée est dynamique et produite par l'instinct courant: veille = debout, somnolence = couché, jeu = ca dépend, etc.

Il est important qu'au début l'instinct l'emporte sur les consignes, la policy ne doit pas bloquer l'instinct pour qu'il puisse remplir le dataset. Je propose de progressivement baisser le seuil de déclenchement (ou la pondération de l'interpolation) de l'instinct, cycle de sommeil par cycle de sommeil (prévoir un contrôle utilisateur pour ce seuil).

# Sommeil naturel

 Il faut intégrer le sommeil de façon réaliste dans la simulation. Actuellement déclenché artificiellement par l'utilisateur. L'objectif: l'envie de dormir (à distinguer de la fatigue) est un état proprioceptif qui domine au delà d'une certaine quantité de vécu (taille du journal).
 
 Cette envie de dormir déclenche un instinct qui encourage la position couchée. Il façonne le calcul du confort (et de la douleur pour neutraliser le coût de contact corps/sol). Cet instinct l'emporte sur d'autres instincts, comme le jeu.
 
 Le cycle de sommeil se déclenche de lui-même quand l'animat est dominé par l'instinct de repos, et que les coûts sont sous un seuil, ou plutôt plusieurs seuils définis par canaux, c'est à dire un confort satisfait par la posture couchée, ainsi qu'une absence d'efforts et de mouvements.

 A la sortie du sommeil, l'instinct dominant, par exemple le jeu, reprend le dessus.

# Refonte architecturale pour rendre le code modulaire

Système de modules pour étendre le contenu. Tous les modules dans un dossier modules sont chargés au démarrage.

Framework architectural à définir avant: framework maison (par exemple des hooks) et/ou une approche entité-role / aspects populaire dans le développement de jeux vidéo, je me rappelle plus son nom exacte.

D'abord le rendu de l'animat peut être substitué par un autre (à consignes égales): le rendu 2D et la physique associé au squelette de l'animat est donc un module. Le rendu existant est celui par défaut, c'est un premier module de type "species". Mais il peut être surchargé par un module de type "species", normalement unique. 

Chaque module peut lire sa config dédiée dans config.toml.

Un module peut enrichir l'environnement physique, ajouter des actuateurs et des récepteurs, définir de nouvelles dimensions associées aux objets, étendre le renforcement (nouveaux coûts et pondération pour le score) et la proprioception. La salve combine une base commune de tokens et des ajouts venant des extensions.

Un module de type "extension" peut définir:
- de nouveaux objets physiques dans l'environnement simulé
- un composant d'IHM de contrôle pour l'utilisateur (cadre à définir)
- de nouveaux percepteurs (tokens dédiés)
- de nouveaux actuateurs. Comme une action doit tenir dans un seul token, il s'agit de consignes dans le token d'actuation.
- de nouveaux coûts dans le token de renforcement
- une contribution dynamique à des coûts existants (confort, effort, douleur)
- de nouveaux intégrateurs dans le token de proprioception
- de nouvelles méta-données à rajouter aux objets physiques, comme l'appétance et l'attrayance.
- de nouvelles dimensions (voir plus bas).
- un onglet de paramètres
- de la config à placer dans config.toml

Des features existantes sont refactorées en modules. On commence par l'extension "balle" qui définit
- la dimension "attrayance"
- la dimension "agripable"/grabable pour tous les objets
qui peuvent être ramassés
- l'objet physique balle, attrayant et agripable
- l'IHM de contrôle de la balle. Elle n'existe pas
mais il faut que la balle soit absente au démarrage et
que pour la faire apparaître il faille drag&drop depuis
une entrée "d'inventaire" balle (on peut le faire plusieurs fois pour replacer la balle).
- le token de perception de la balle (ou tout autre objet attrayant)
- l'actuateur "grab" pour prendre 1.0 ou lâcher 0.0 un objet mobile "grabable" à proximité des mains.
- Un contrôle d'IHM (bouton à état) pour l'actuateur grab. Pénalité si on grabbe dans le vide (inconfort, effort)

Une nouvelle extension "croquette":
- ajoute la dimension "apétance" aux objets
- ajoute l'objet croquette dans la physique, objet apétant, une seule à la fois, instanciable par drag&drop depuis un inventaire, déplacable au clic comme la balle, et qui disparait après un certain temps de consommation.
- ajoute l'actuateur pour consommer un obet appétant / une "croquette" avec un bouton poussoir dans l'IHM pour le mode supervisé (ou bien l'acte de consommation est
automatique dès que la tête est dans le rayon d'influence
de la croquette) : attention si on garde cet actuateur il faut qu'il ait un coût d'inconfort et d'effort s'il est actif dans le vide (pas de croquette) et si l'animal bouge en même temps (impossible).
- ajoute le token de perception de croquette, un "odorat",
  coordonnées (radiales?) dans le référentiel égocentrée
- ajoute le cout négatif (récompense) "fumet de croquette": coût proportionnel à la proximité de la croquette: 1.0 si au niveau de la tête (consommable), tend vers 0 avec l'éloignement.
- ajoute le coût négatif (récompense) "délectation" (pondération forte) à 1.0 pendant la consommation de croquette.
- ajoute l'intégrateur proprioceptif "faim" qui baisse par délectation et qui sinon augmente lentement (pas d'usage pour l'instant)

A l'occasion de cette refonte, on introduit un nouveau fichier parameters.toml et une IHM de paramétrage.
Les parameters.toml sont écrasées par l'application sur
sauvegarde des paramètres. L'IHM de paramétrage dispose
d'onglet: un onglet des paramètres de base (aucun paramètres pour l'instant), et un onglet par module.
Le module "species" est unique et obligatoire, il a un onglet dédié de customisation (le premier onglet). Ensuite
un onglet par module: Un module peut rajouter ses propres paramètres.

Dépendances entre extensions et nouvelle "dimensions".

Un *dimension* _appétance_ est partagé par tous les objets consommables. Il est défini dans une extension "consommable", dont dépend l'extension "croquette". La notion d'appétance, les actuateurs, les récepteurs et le renforcement relatifs aux consommables sont définis une seule fois dans l'extension consommable. Le module "croquette" et de futurs modules eux rajoutent des objets physiques avec la dimension "appétance".

Un *dimension* _attrayance_ est défini par une extension réutilisée par "balle" et d'autres extension futures pour des objets amusants. Noter que l'attrayance est une valeur dynamique (une balle est attrayante quand elle roule, encore plus quand elle monte, encore plus quand elle change de direction).

Les intéractions entre une *dimension* (méta-donnée d'objet)  et le champ sensoriel existant fait partie de la définition de la dimension. Cela précise si la dimension s'ajoute à:
- chaque pixel du champ visuel,
- l'ouïe (à clarifier)
- l'odorat (à rendre canonique et à clarifier)
- etc.
Mais avant de tout casser vérifier si on peut imaginer d'autres dimensions.

La dimension grabable intéragit avec l'actuateur "grab", je sais pas si on peut généraliser à d'autres intéractions entre des dimensions futures et des actuateurs multi-dimenssionels. Par exemple on pourrait rajouter des dimenssions à la consommation.

J'ai du mal à imaginer si on pourrait faire des dimensions de dimensions: la soif vs la faim par exemple. Ca devient un peu trop abstrait.

Certains actuateurs ne sont pas compatibles entre eux ou sont incompatibles avec certains états de l'animat. Comme dit plus tôt, l'animat ne peut pas manger et bouger en même temps (quoi que ...). A priori, c'est l'inconfort et l'effort qui encode cette impossibilité en plus de la simulation qui neutralise l'action même si la consigne l'active (par exemple pas de consommation de croquette si la tête accélère).

Il pourrait y avoir de futurs actuateurs "à distance".  J'envisage la possibilité d'interagir (récepteur et actuateur) sur des élements du système d'exploitation et de l'environnement bureautique (ou son équivalent smartphone), voir l'environnement réseau, voir domotique. Prévoir de combiner les actuateurs à distance avec des actuateurs de type pointage/focus qui conditionnent des récepteurs (perception de l'entités pointée) et des actuateurs (impact sur l'entité pointée). Ce serait formidable si un jour un actuateur permettait à l'animat de projeter/étier le cône du champ de vision.

Un cout est souvent dérivé d'un ou plusieurs
récepteurs sensoriels mais je préfère pas le formaliser
explicitement dans le SDK.

Modules incompatibles: le module "species" ne peut fonctionner avec d'autres modules "species". Prévoir une IHM de sélection. Prévoir une IHM pour activer/désactiver des modules. Prévoir que l'activation d'un module peut en désactiver d'autres. Prévoir que l'activation d'un module produit la plupart du temps un reset du cerveau (réapprentissge obligatoire des modèles). Prévoir de retarder le reset du cerveau à la sortie de l'IHM de paramétrage (tracer les changements, si pas de changement destructif, pas de reset) avec confirmation de sortie si c'est le cas.

Une extension peut introduire de nouveaux états internes (qu'elle gère), de nouveaux instincts, et une matrice (ou un complément de matrice) de poids d'activation etat->instinct. Si plusieurs instincts sont actifs en même temps, une priorisation arbitre: A l'emporte sur B, B l'emporte sur C, etc. Risque de boucle (si C l'emporte sur A), choisir alors le dernier instinct activé!

