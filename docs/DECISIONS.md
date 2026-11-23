# DECISIONS.md — Décisisions implémentées dans λ̄

Une phrase simple par décision.

- λ̄  est un Animat apprenant dans un environnement 2D simulé.
- 2D Vue de coté. Orientation de l'animat (sens de la marche et du regard) à droite ou à gauche. C'est le `facing`.
- Simulation Pymunk avec ressorts, friction au sol et gravité. Des plateformes statiques et une balle rebondissante. Tick de simulation physique stable: 60 Hz.
- Un actuateur pour chaque jambe (consignes d'angle et de longueur) et pour la queue (angle uniquement).
- Les actuateurs suivent des valeurs effectives de consignes après lissage (50 % de l'écart résiduel par période policy).
- Référentiel égocentré des angles de membres : positif vers l'extérieur, orienté vers la marche.
- Retournement de l'orientation de l'animat: `facing` **flip**. Réorientation du référentiel. Réinitialisation du lisseur. Purge de l'historique WM. Reconstruction immédiate de la salve courante.
- Une IA à 2 modèles : Un Transformer "Word Model" et un MLP "Politique".
- Le World Model prédit soit le futur, soit la conséquence d'une action, soit une action par anticipation du dénouement.
- La Politique décide la prochaine action
- La Politique prend en entrée le latent intermédiaire du WM.
- Apprentissage continu obtenu par un cycle d'éveil (inférence et enrichissement du dataset) et de sommeil (apprentissage).
- 2 Modes d'éveil. Mode auto / autonome : inférence des consignes par l'IA vs. Mode manuel / supervisé : L'utilisateur pilote manuellement les consignes de l'animat (l'enregistrement produit de nouvelles données d'entraînement).
- Boucle principale de l'éveil : physics step → smoother → facing-flip → sleep check → brain_tick → record_tick.
- L'animat perçoit par Proprioception, Extéroception, Intéroception.
- Proprioception : angles, extensions, vitesses, forces d'actuateurs et contacts au sol.
- Extéroception : vision en cône rétinien, flux optique sur les solides mobiles, toucher, audition et suivi de la balle.
- Intéroception : la fatigue accumule l'effort. La souffrance accumule la douleur. Intégrateurs lents avec Récupération encore plus lente.
- Circuit de récompense inné: effort, douleur, courbature, instabilité, vertige, confort (5 coûts positifs = pénalité, 1 coût négatif = récompense), d'autres en gestation
- hiérarchie des coûts par pondération : effort ×1, douleur ×4, courbature ×2, instabilité ×2, vertige ×3, confort ×1.
- effort = quantité de mouvement * force exercée, 0 si immobile
- douleur = force appliquée / surface de contact, lissée par une enveloppe asymétrique (attaque instantanée, décroissance exponentielle).
- courbature = immobilité prolongée
- instabilité = distance du centre de masse hors du polygone de sustentation (les 2 pieds) : Grandit quand le centre de masse sort et s'éloigne du support.
- vertige = |accélération de la tête|
- confort = posture proche d'une posture instinctive, actuellement le tronc vertical, sans douleur (ni assis ni couché)
- Filtrage IIR sélectif : seuls les signaux dérivés bruités (accélérations, forces, flux optique) sont lissés, les signaux physiques directs passent sans retard.
- Capteur balle avec vitesses radiale et angulaire.
- Enroulement continu de l'angle du tronc sur [-15°, 345°].
- Actions encodées sans échelle physique, toujours dans l'intervalle [0,1] ; sigmoid en sortie de policy.
- Normalisation de signaux (instabilité, vertige...) dans l'intervalle [0, 1) par saturation (raw / (raw + 1)) 
- Salve dense de 16 tokens (13 états + 3 actions) produite à chaque tick du World Model. L'action répond à l'état.
- Token dense de 29 flottants : type(1) + modalité one-hot(6, bit 6 dédié au dénouement EOS) + canal one-hot(6, ids réutilisés entre modalités) + 16 signaux. Le préfixe structurel n'est jamais prédit en rollout (copié des templates) ni appris (loss sur les 16 signaux seuls).
- PE sinusoïdal non appris keyed by indice de salve (tous les tokens d'une salve partagent la position) ; le token EOS a un PE dédié `seq_steps+1` (hors fenêtre, token de synthèse). Dans la conséquence rétro-causale, les positions temporelles restent décroissantes (26→14).
- Token d'état = le champ sensoriel dont les coûts
- Tokens d'action = consigne pour les actuateurs.
- Agencement causal dans la salve : perception → coûts → confort → intéroception, coupant tout raccourci intra-salve.
- Séquence = fenêtre glissante de salves consécutives
- `seq_steps` 26 salves par séquence, paramétrable. Les n_ctx premières salves de la séquence (hors décision) forment le contexte de décision. L'action à n_ctx est la décision. Les salves suivantes forment la fenêtre de conséquence.
- Le token EOS siège à `eos_pos` (premier token après le contexte), modalité dédiée. Il porte le dénouement : intégrale par canal de l'écart au coût de la décision `r[j] = Σ_{k>d} eos_decay^(k−d)·(c[k][j]−c[d][j])·p_k`, dernier état `p_k = L−d` (prolongation fictive du coût final), `slot = 0.5 − 0.5·tanh(r)`.
- Le taux de soulagement `sequence_relief` est un score basé sur le dénouement. >0 amélioration; <0 détérioration.
- Journalisation linéaire des salves. Journal segmenté sur les discontinuités physiques (retournement, reset).
- Le WM est un transformer causal prédisant l'état et l'action suivants.
- Prédiction conjointe de la salve complète état + action (`predict_next_salve`).
- Latent intermédiaire capturé par hook sur la couche médiane, normalisé par LayerNorm apprenable, et fourni à la policy.
- Dans la phase d'éveil. Inférence WM basé sur un historique des dernières salves vécues (taille du contexte pour l'apprentissage de la policy). L'historique est purgé quand nécessaire (`clear_history`).
- KV-cache incrémental pour l'inférence, invalidé à chaque mise à jour des poids.
- Pondération de l'entraînement WM par l'impact de la séquence, c'est à dire abs(sequence_relief) + plancher: contraste recherché impact/neutre x3.
- Boost ×16 du gradient sur les canaux de récompense et les slots EOS terminaux.
- Passes d'entraînement WM par batch : chaque batch subit une passe causale ET une passe lookahead (plus d'alternance 1:1 par epoch).
- mode causale: le World Model apprend à prédire l'action et la conséquence à partir d'un contexte.
- mode lookahead: le World Model apprend à prédire l'action conditionnée par le dénouement.
- En passe causale: 
  * Zero-padding de l'intéroception (fatigue/souffrance) sauf EOS.
  * Causalité stricte : masque par groupe empêchant les tokens d'une même salve de se voir entre eux.
- En passe lookahead: Masque d'attention lookahead : le token EOS ne voit que le contexte de décision; et l'action décision voit le token EOS et le contexte.
- 5 epochs WM par sommeil = tout le dataset parcouru en batchs. Coreset trié. Addendum shufflé à chaque epoch.
- Trois profils matériels configurables : CUDA accéléré, CPU portable, CPU rapide (MKL-DNN).
- Masquage strict à zéro des slots de padding invalides dans les prédictions.
- La Policy est un MLP recevant les scalaires bruts et le latent du World Model.
- Scalaires et latent Policy recalculé à chaque tick policy plutôt qu'au dernier tick WM
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
- Coreset (mémoire long terme consolidée) + Addendum (mémoire court terme) + Journal (veille linéaire).
- Chaque séquence porte des métadonnées : origin, vivacity, relief_score, learn_count, causal_score, lookahead_score, dissonance, surprise, latent (vecteur EOS), pol_relief_score.
- Métadonnées calculées en début de sommeil: passes causale et lookahead ; passe politique ; passe causale post contexte+action politique.
- Vidage du journal d'expérience entre la phase de sommeil et celle de veille.
- Pipeline de sommeil en étapes ordonnées : extraction, remise en jeu de l'oubli actif, métadonnées, déduplication, entraînement WM sur le Coreset, filtrage par surprise, entraînement WM sur l'Addendum, entraînement de la policy, consolidation, éviction.
- Oubli actif : 33 % du Coreset remis en jeu à chaque cycle, sélectionné stochastiquement (Efraimidis-Spirakis pondéré par la vivacité).
- Décroissance de la vivacité (-0,01 par cycle).
- Tri du Coreset si plateau WM (ratio loss après/avant > 0.95) : min-flou pondéré
  des valeurs min-max normalisées — déterminisme=(lookahead+causal)/2 (poids `tri_w_det`),
  consonance=1/(1+dissonance) (`tri_w_cons`), vivacité^`tri_vivacity_pow` (`tri_w_viv`).
  Les moins pertinentes en premier. Tri bruité par de l'aléa.
- Éviction en cas de saturation du Coreset selon l'orde du tri.
- Déduplication géométrique intra-Addendum par clustering hiérarchique (conservation du médoïde).
- Déduplication croisée Addendum ↔ Coreset (ε-net) avec ravivage de la vivacité des souvenirs retrouvés.
- Distance combinée : RMSE pondérée par sensibilité EOS (saliency, utilisée seulement quand le WM a convergé) + distance cosinus entre latents de séquences (latent_weight=1.0).
- Filtrage cognitif par surprise : les séquences familières (erreur sous le seuil) sont éliminées avant entraînement. Les séquences neuves de l'Addendum (learn_count==0) sous le seuil sont supprimées dès la première epoch.
- Ratio loss WM avant/après sommeil persisté ; plateau WM si ratio > 0.95.
- Compteur d'apprentissage par séquence : +1 par epoch WM. Surprise = loss causale du dernier batch.
- Quotas d'époques identiques pour le Coreset et l'Addendum (5).
- Critère d'arrêt dynamique : quota minimum d'époques ET loss cible, avec prolongation possible.
- Persistance disque des trois checkpoints (buffer, WM, policy) avec chargement résilient et `--reset`.
- Échantillonnage équiprobable des fenêtres glissantes par sommes préfixes et recherche dichotomique.
- Éviction O(1) des salves les plus anciennes à saturation de capacité.
- Extraction des fenêtres avec stride (actuellement 1)
- En mode brain-off, décalage action/conséquence à l'enregistrement : paire (a_n ; s_{n-k}), k paramétrable (défaut 2) ; brain-on = k forcé à 0.
- En mode supervisé: Pilotage manuel par manettes
- Bascule du mode supervisé sans à-coup par synchronisation des manettes avec la policy.
- Reset de la simulation sur commande
- Sortie de la manette: continuité polaire avec saturation à ±(π − 10°) au lieu d'un wrap ±π.
- Mode accélération du temps réel de 0,25x à 8x.
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
- Configuration TOML centralisée (`config.toml`) pour toutes les options structurelles
- Dossier `.tmp/` gitignoré pour les scripts temporaires.

## Idées abandonnées / en cours d'abandon

- Entraînement alternatif de la policy sur option `--no-eos-lookahead`: entraînement basé sur l'imagination.
   * Apprentissage supervisé basé sur la meilleure action (d'après le futur prédit par le WM) entre 3 candidats: [action historique telle qu'enregistrée dans la séquence] [action générée par la politique courante] [exploration gaussienne σ = 0,30 autour de la politique courante].
   * Pour chaque action candidate: Imagination WM multi-futurs. On évalue l'action par son meilleur futur ("optimisme de Bellman").
   * 2 futurs: [prolongation Kalman amortie de l'action candidate] et [application de la policy sur les ticks suivants].
   * Prolongation Kalman avec clamping de vitesse physique (±0,20 rad/pas, ±0,15 extensions).
- NON: quantification des signaux de token.
- NON: entraînement REINFORCE de l'action inférée à partir du gain prédit par le WM
- NON: Générateur de dataset d'amorçage headless (`--bootstrap-dataset`) avec auto-reset sur chute. Câblé dans le code (`main.py`, `dataset.py`) mais jamais testé et non voulu.
- NON: Compilation optionnelle `torch.compile` du pas d'entraînement WM (graphe fusionné, passe arrière incluse).
- Avant, le token EOS était le dernier token de la dernière salve. C'était un token terminal qui se subtituait au token 12 de salve.