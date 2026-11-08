# ADR-0001 — Nomenclature

## Contexte

Les agents mélangent trop de variantes pour les mêmes concepts.

## Décision

On fixe une liste de termes canoniques.

**Utilisation**

- Doc : termes FR
- Code : termes EN sauf pour les tokens
- GUI : termes EN

### Termes pour décrire l'animat, son champ sensoriel et circuit de récompense

- **animat** : λ̄
- **squelette / skeleton** : corps physique de l'animat. corps / body
   * membre arrière/avant / front/back limbs
   * tête / head
   * tronc : élément principal du corps
   * Queue = Tail
   * L'animat contrôle la position des membres et l'angle de la queue.
   * Des ressorts sont simulés entre la consigne globale désirée et la position courante.
- **bascule de la direction du regard / facing-flip** : l'animat a un corps 2D qui change de coté; facing direction; facing-flip
- **environnement simulé**: l'animat 2D et ce qui l'entoure
- **état / state** : l'état du monde perçu par l'animat, les tokens en entrée du cerveau, produits par la perception de la physique et le circuit de renforcement
- **proprioception** : état du corps
- **extéroception** : monde extérieur
- **référentiel égocentré**
- **intéroception** : états internes lents (coûts intégrés)
- **consigne** : cible motrice, valeur d'actuateur. Consigne sur la position des membres (θ*, d*) et l'angle de la queue.
- **coûts / costs** : l'animat associe à son état des coût innés signés (bénéfiques ou pénalisants). Sans précision, il s'agit de coût instantané.
- **pénalités / penalty** : Coûts positifs: effort, douleur, courbature, instabilité, vertige. 
- **récompenses / reward** : Coûts négatifs comme "le confort".
- **reward_pos / reward_neg** : agrégats signés du renforcement
- **dénouement** : vecteur delta des coûts entre le contexte de décision et la conséquence de l'action
- **EOS token** : token terminal de la séquence. C'est un token d'intéroception qui porte le dénouement.
- **soulagement / relief** : Relief score ; sequence relief. le soulagement d'une séquence. Delta pondéré des coûts entre le contexte de décision et la fenêtre de conséquence.
- **score de dénouement / favorability** : favorability ; fav(denouement) ; score du token EOS, pour affichage.
- **Poids des canaux de coût**
- **position couchée** : position douloureuse après une chute
- **debout** : posture neutre
- **relèvement** : passé de la position couchée à debout
- **balle / ball** : mobile bondissant et attrayant
- **croquette / kibble** : objet simulé qui réduit la faim
- **audition** : balle, touches du clavier
- **bilan métabolique** : fatigue et souffrance.Intégrateurs lents internes des coûts
- **faim / hunger** : coût qui progresse lentement et diminue par la consommation de croquette
- **délectation / delight** : reward de consommation
- **Fatigue** La fatigue intègre l'effort
- **Souffrance** La souffrance intègre la douleur
- **récupération / recovery** :  Facteur de récupération
- **signaux normalisés de capteur** : stabilité (centre de masse vs polygone de sustentation), collision, choc, contact au sol, cône visuel constituée de cellule rétinienne, confort postural, effort musculaire, flux optique,
- **curiosité, ennui, playfulness** : circuit de renforcement avancé
- **vertige** : accélération de la tête
- **canal / channel** : emplacement d'un signal normalisé dans un token ; normalized signal slot

### Termes pour l'IA

- **token** : token d'action, token d'état, token terminal (EOS)
- **salve / burst** : un ensemble ordonné de tokens (états puis actions) produit à chaque tick WM.
- **séquence / sequence** : fenêtre glissante de salves
- **journalisation / recording** : journalisation linéaire des salves pendant l'éveil
- **segment** : portion de journal entre deux discontinuités
- **coreset** : mémoire de séquences à long terme
- **addendum** : mémoire récente de séquences vécues ou remémorées
- **buffer d'expérience / experience buffer** : coreset + addendum + journal
- **historique / history**: les n dernières salves produites pendant la veille.
- **world model (WM) / world model** : modèle pour prédire le futur à partir d'un contexte ou une action à partir d'un dénouement.
- **politique / policy** : modèle pour contrôler l'animat en inférant une action faite de consignes. En entrée l'état courant de l'animat et le latent.
- **Brain** : les 2 modèles
- **Vecteur latent intermédiaire** : la policy est alimentée par l'état courant et le vecteur latent extrait de la couche médianne du WM après inférence causale
- **Prise de décision** : L'action de la policy, dans la séquence enregistrée.
- **contexte de décision / decision context** : Le début de la séquence avant la prise de décision. contexte causal. contexte de décision. contexte de réveil. La pondération temporelle des coûts du contexte (context time weights) renforcent les plus récents.
- **la (fenêtre de) conséquence**: la fin de la séquence après prise de décision.
- **Entrainement sur l'action conditionnée par un meilleur dénouement** : Le WM supervise l'apprentissage de la politique: Il infère d'abord la conséquence de son action (passe causale), puis l'action vers un  dénouement anticipé meilleur (passe lookahead). EOS-forcé. Hindsight. insight.
- **passe causale / causal rollout**: Générer le futur après une action. Transformeur Auto-régressif. Latent passé à la politique.
- **passe d'anticipation / lookahead rollout** : Générer l'action qui va produire un certain dénouement.
- **action historique réelle** : l'action réelle enregistrée dans la séquence.
- **action de la politique** : l'action générée par la politique
- **action conditionnée** : l'action générée par le WM sous condition d'un dénouement. Action hindsight. Lookahead action.
- **tick Recording** : cadence la journalisation des salves
- **tick Brain** : cadence l'inférence du WM et de la politique pour générer une action pendant l'éveil
- **compression temporelle**: il faut synthétiser les canaux entre 2 ticks de recording pour produire des tokens
- **lissage par filtre IIR passe-bas**: lissage souvent utilisé pour les entrées et les sorties des modèles
- **valeur effective / effective value** : valeur après lissage.
- **forcage aux extrèmes**: extremum, l'action conditionnée est déplacée vers les extrèmes qui sont attracteurs

### Termes sur l'optimisation du dataset et l'entrainement:

- **cycle circadien éveil/sommeil**
- **veille / wake** : phase d'éveil, simulation active de l'animat autonome. Journalisation du vécu pour la phase de sommeil.
- **sommeil / sleep** : phase d'apprentissage hors ligne
- **oubli actif / active forgetting** : remise en jeu du coreset. échantillonnage priorisé des séquences.
- **vivacité / liveliness** : le caractère récent d'un souvenir
- phase addendum et phase coreset de l'entrainement WM
- **filtrage par surprise / surprise-based filtering** : écarte les séquences familières. Familiar sequences. Score de surprise.
- **Déduplication**
  * Intra-addendum duplicates
  * Addendum/coreset duplicates
  * Clustering Hiérarchique Agglomératif
  * Complete-linkage clustering
- Apprentissage WM, passes causale et passes par anticipation.
- Prioritized sampling
- Quota d'époques
- Apprentissage politique
- consolidation addendum vers coreset: on vide les séquences de l'addendum dans le coreset.
- éviction / Coreset Pruning : Quand on dépasse la capacicité du coreset, on enlève les séquences les moins vivaces.
- **Impact**: valeur absolue du soulagement ; abs(sequence_relief(sequence))
- **Entraînement pondéré par l'impact / Impact-proportional training** : les séquences avec plus d'impact ont plus de poids à l'entraînement WM
- **salience / saliency** : Déduplication selon une distance inter-séquence basée sur la salience ; on extrait du WM les poides de salience : saliency weights.

Autres détails d'implémentation :

- config.toml
- accélération CPU/GPU
- activations ReLU
- boost de gradient
- bruit d'exploration gaussien
- cache KV / KV-cache
- Cache LRU
- checkpoint
- couche médiane / median layer
- demi-vie
- détection de blocage
- épochs/époques
- eps = epsilon
- Error threshold
- exploration gaussienne
- lookahead gen mask
- lookahead mask
- loss
- manettes droites, gauches ; clamping
- masque d'attention causal / causal attention mask
- masque d'attention lookahead
- mode manuel / manual mode
- mode auto / inférence
- run mode headless
- pas d'entraînement
- ReLU
- reset positionnel
- snapshot
- échelle temporelle, temps accéléré
- Transformeur auto-régressif / autoregressive transformer
- valence
- World Model Theater
- Dream Theater
