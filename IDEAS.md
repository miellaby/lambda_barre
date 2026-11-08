# IDEAS.md — Idées implémentées dans λ̄

Une phrase simple par idée, organisées par domaine. Sources : historique des commits et sessions `session_*.md`.

## Architecture

- Créature 2D dans environnement simulé
- IA à 2 modèles : Un Transformer "Word Model" et un MLP "Politique".
- Le World Model prédit soit le futur, soit la conséquence d'une action, soit une action par anticipation du dénouement.
- La Politique décide la prochaine action
- La politique est reliée au latent intermédiaire du WM.
- Apprentissage continu obtenu par alternance entre éveil (inférence et enrichissement du dataset) et sommeil (apprentissage).
- Mode auto : inférence des consignes par l'IA.
- Mode manuel : pilotage manuel des consignes enregistrées comme données d'entraînement.
- Boucle principale linéaire : physics step → smoother → facing-flip → sleep check → brain_tick → record_tick.

## Corps & simulation physique

- Animat 2D. Vue de coté orientée vers le sens de la marche et du regard, à droite ou à gauche (`facing`).
- Simulation Pymunk avec ressorts, friction au sol et gravité. Des plateformes statiques et une balle rebondissante. Tick de simulation physique stable: 60 Hz.
- Un Actuateur pour les jambes (consignes d'angle et de longueur) et un pour la queue (angle).
- Lissage des consignes : la physique suit des valeurs effectives de consignes après lissage (50 % de l'écart résiduel par période policy).
- Référentiel égocentré des angles de membres : positif vers l'extérieur, orienté vers la marche.
- Retournement (`facing` **flip**) : Nouvelle orientation du référentiel. Réinitialisation du lisseur. Purge de l'historique WM. Reconstruction immédiate de la salve courante.
- Accélération du temps réel de 0,25x à 8x.

## Capteurs & signaux

- Proprioception : angles, extensions, vitesses, forces d'actuateurs et contacts au sol.
- Extéroception : vision en cône rétinien, flux optique sur les solides mobiles, toucher, audition et suivi de la balle.
- Intéroception : la fatigue accumule l'effort. La souffrance accumule la douleur. Intégrateurs lents avec Récupération encore plus lente.
- Circuit de récompense inné pondéré : effort ×1, douleur ×4, courbature ×2, instabilité ×2, vertige ×3, confort ×1.
- Douleur lissée à la source par une enveloppe asymétrique (attaque instantanée, décroissance exponentielle).
- Filtrage IIR sélectif : seuls les signaux dérivés bruités (accélérations, forces, flux optique) sont lissés, les signaux physiques directs passent sans retard.
- Capteur balle avec vitesses radiale et angulaire.
- Enroulement continu de l'angle du tronc sur [-15°, 345°].
- Actions encodées sans échelle physique, toujours dans l'intervalle [0,1] ; sigmoid en sortie de policy.
- Capteur d'instabilité. L'instabilité brute est la distance du centre de masse hors du polygone de sustentation (entre les deux pieds) : Plus le centre de masse sort du support, plus raw grandit.
- Normalisation de signaux (instabilité, vertige...) dans l'intervalle [0, 1) par saturation (raw / (raw + 1)) 


## Tokenisation & mémoire vive

- Salve dense de 16 tokens (13 états + 3 actions) produite à chaque tick du World Model. L'action répond à l'état.
- Tokens d'action = consigne pour les actuateurs.
- Agencement causal de la salve : physique → coûts → confort → intéroception, coupant tout raccourci intra-salve.
- Séquence = fenêtre glissante de salves consécutives
- 10 transitions par défaut, paramétrable. Les n_ctx = 4 premières slaves de la séquence forment le contexte de décision. Les suivantes forment la fenêtre de conséquence.
- Le token EOS est le dernier token de la dernière salve d'une séquence. Il porte le dénouement = le delta des coûts innés, entre la conséquence et le contexte de décision. On en déduit le taux de soulagement `sequence_relief`. 
- Coûts du contexte pondérés en décroissant vers le passé.
- Coûts de conséquence moyennés.
- En passe causale: 
  * Zero-padding de l'intéroception (fatigue/souffrance) sauf EOS.
  * Causalité stricte : masque par groupe empêchant les tokens d'une même salve de se voir entre eux.
- Journalisation linéaire des salves. Journal segmenté sur les discontinuités physiques (retournement, reset).

## World Model

- Transformer causal prédisant l'état et l'action suivants.
- Prédiction conjointe de la salve complète état + action (`predict_next_salve`).
- Latent intermédiaire capturé par hook sur la couche médiane, normalisé par LayerNorm apprenable, et fourni à la policy.
- Dans la phase d'éveil. Inférence WM basé sur un historique des 4 dernières salves vécues (taille du contexte pour l'apprentissage de la policy). L'historique est purgé quand nécessaire (`clear_history`).
- KV-cache incrémental pour l'inférence, invalidé à chaque mise à jour des poids.
- Compilation optionnelle `torch.compile` du pas d'entraînement WM (graphe fusionné, passe arrière incluse).
- Pondération de l'entraînement WM par l'impact de la séquence, c'est à dire abs(sequence_relief) + plancher: contraste recherché impact/neutre x3.
- Boost ×16 du gradient sur les canaux de récompense et les slots EOS terminaux.
- Passes d'entraînement alternées en mode EOS-lookahead : le World Model apprend à prédire l'action conditionnée par le dénouement.
- Masque d'attention lookahead : la position de décision voit le token EOS terminal, et la ligne EOS ne voit que le contexte de décision.
- Trois profils matériels configurables : CUDA accéléré, CPU portable, CPU rapide (MKL-DNN).
- Masquage strict à zéro des slots de padding invalides dans les prédictions.

## Policy

- Policy MLP recevant les scalaires bruts et le latent du World Model.
- Scalaires et latent recalculé à chaque tick policy plutôt qu'au dernier tick WM
- Espace de sortie normalisé (angles dans [-1, 1], extensions dans [0, 1]) séparé de la conversion physique.
- MLP "moderne" : 256 neurones internes, activations ReLU, sortie tanh/sigmoid.
- Entraînement par imitation : la policy apprend à reproduire une action obtenue par inférence du World Model
- Supervision hindsight EOS-lookahead : Le WM prédit le futur de l'action politique, puis génère une nouvelle action qui donne un meilleur dénouement. 
- Forçage additif de l'EOS vers un dénouement plus favorable : plancher à la neutralité (0,5) puis décalage d'amplitude ε ~ U(0,15 ; 0,40) par slot.
- Amplification des actions hindsight vers les extrêmes.
- Échantillonnage de la policy priorisé par soulagement (relief score basé sur le delta de souffrance et de fatigue).
- Bruit `act_std` sur la sortie policy
- Synthèse des consignes par extremum entre deux ticks WM pour capturer l'intention réelle.
- Politique gelée pendant le sommeil ; au réveil, le premier `brain_tick` produit un latent frais.

## Mémoire à deux niveaux & cycle de sommeil

- Coreset (mémoire long terme consolidée) + Addendum (mémoire court terme) + Journal (veille linéaire).
- Vidage du journal d'expérience entre la phase de sommeil et celle de veille.
- Pipeline de sommeil en étapes ordonnées : extraction, remise en jeu de l'oubli actif, déduplication, entraînement WM sur le Coreset, filtrage par surprise, entraînement WM sur l'Addendum, entraînement de la policy, consolidation.
- Oubli actif : 33 % du Coreset remis en jeu à chaque cycle, sélectionné stochastiquement (Efraimidis-Spirakis pondéré par la vivacité).
- Décroissance de la vivacité (-0,01 par cycle) et éviction prioritaire des souvenirs froids en cas de saturation.
- Déduplication géométrique intra-Addendum par clustering hiérarchique (conservation du médoïde).
- Déduplication croisée Addendum ↔ Coreset (ε-net) avec ravivage de la vivacité des souvenirs retrouvés.
- Distance pondérée par sensibilité EOS : saliency rétropropagée depuis le token terminal, utilisée seulement quand le WM a convergé.
- Filtrage cognitif par surprise : les séquences familières (erreur sous le seuil) sont éliminées avant entraînement.
- Quotas d'époques identiques pour le Coreset et l'Addendum.
- Critère d'arrêt dynamique : quota minimum d'époques ET loss cible, avec prolongation possible.
- Persistance disque des trois checkpoints (buffer, WM, policy) avec chargement résilient et `--reset`.
- Échantillonnage équiprobable des fenêtres glissantes par sommes préfixes et recherche dichotomique.
- Éviction O(1) des salves les plus anciennes à saturation de capacité.
- Extraction des fenêtres avec stride (actuellement 1)
- capacité portée à 100 000 séquences.

## Interface

- En mode supervisé: Pilotage manuel par manettes
- Bascule du mode supervisé sans à-coup par synchronisation des manettes avec la policy.
- Reset de la simulation sur commande
- Sortie de la manette: continuité polaire avec saturation à ±(π − 10°) au lieu d'un wrap ±π.
- Mode headless (`SDL_VIDEODRIVER=dummy`) pour tests et génération de dataset.
- Dream Theater : visualiseur des rollouts d'imagination pendant l'entraînement de la policy, avec storyboard hindsight et navigation PREV/NEXT/LIVE.
- Dream Theater standalone : entry point `dream-theater` indépendant de la boucle principale.
- WM Theater : visualiseur standalone comparant les séquences réelles du Coreset aux rollouts autorégressifs du World Model.
- WM Theater : tri des séquences par dénouement (relief, impact absolu, ordre brut), débounce du slider temporel et mise en valeur de l'EOS.
- WM Theater : mode lookahead affichant la trajectoire contrefactuelle générée par le conditionnement EOS.
- Vignette égocentrique du World Model en temps réel pendant la veille.
- HUD avec métriques utiles : temps d'inférence, pertes, tailles mémoire, taux d'acceptation hindsight.
- GUI standardisée : pas d'antialiasing, police monospace, fenêtre redimensionnable avec ratio, plein écran Alt+Entrée.
- Enregistrement automatique en pause pendant le sommeil.
- Mesure honnête des temps d'inférence : synchronisation device, EMA, cap à 2 threads CPU.

## Configuration & outillage

- Configuration TOML centralisée (`config.toml`) pour toutes les options structurelles (architecture, séquence, buffer, cadences).
- Paramétrage de la longueur de séquence (`seq_steps`, `n_ctx`) et des cadences (`wm_hz`, `pol_hz`) via `compute_seq_layout` et fonctions dédiées.
- Architecture du Brain entièrement paramétrable (d_model, nhead, layers, dim_ff, dropout, pol_hidden, capacités mémoire).
- Suite de tests pytest end-to-end couvrant le pipeline complet.
- Dossier `.tmp/` gitignoré pour les scripts temporaires.
- README en anglais, original français conservé dans `LISEZMOI.md`.
- DESIGN.md avec les détails d'implémentation

## Idées abandonnées / en cours d'abandon

- Entraînement alternatif de la policy sur option `--no-eos-lookahead`: entraînement basé sur l'imagination.
   * Apprentissage supervisé basé sur la meilleure action (d'après le futur prédit par le WM) entre 3 candidats: [action historique telle qu'enregistrée dans la séquence] [action générée par la politique courante] [exploration gaussienne σ = 0,30 autour de la politique courante].
   * Pour chaque action candidate: Imagination WM multi-futurs. On évalue l'action par son meilleur futur ("optimisme de Bellman").
   * 2 futurs: [prolongation Kalman amortie de l'action candidate] et [application de la policy sur les ticks suivants].
   * Prolongation Kalman avec clamping de vitesse physique (±0,20 rad/pas, ±0,15 extensions).
- NON: quantification des signaux de token.
- NON: entraînement REINFORCE de l'action inférée à partir du gain prédit par le WM
- NON: Générateur de dataset d'amorçage headless (`--bootstrap-dataset`) avec auto-reset sur chute. Câblé dans le code (`main.py`, `dataset.py`) mais jamais testé et non voulu.
