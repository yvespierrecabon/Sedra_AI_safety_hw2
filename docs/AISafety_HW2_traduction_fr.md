# AI Safety – Concepts and Techniques — Devoir 2 (traduction française)

> Traduction du sujet officiel `docs/AISafety_HW2.pdf` (7 pages). Les schémas (réseau de la Partie 1, exemples KMNIST/EMNIST) ne sont pas reproduits ici — voir le PDF.

## Devoir 2

**À rendre le 15 décembre 2026 à 12h00, heure d'Allemagne (strict)**

### Consignes :

- **Rapport écrit**
  - Envoyer **un fichier ZIP par groupe** à rohan.menon@uol.de, contenant :
    - **Rapport** : le rapport écrit en PDF, incluant la déclaration d'utilisation de GenAI
    - **Note visée pour ce devoir, en PDF** : par exemple « Partie A vise 1,0, Partie 2 vise 2,0 ». Si certains membres visent moins (ils ne veulent pas investir), l'écrire explicitement.
    - **Dépôt** : un fichier texte contenant le lien vers le dépôt du groupe, accessible à l'équipe pédagogique.
    - **Notebook** : préparer chaque partie de l'expérience sous forme de notebook Jupyter (.ipynb). Inclure le notebook avec son code et ses sorties enregistrées.
    - **Export** : un export HTML ou PDF du notebook exécuté, avec code, figures et résultats visibles.
  - Envoyer aussi la **feuille de contribution** (une personne, une feuille) individuellement à rohan.menon@uol.de
- **Présentation orale (16 ou 17 décembre)**
  - Inscrire le groupe à un créneau de 45 min où tout le groupe est présent pour discuter de ses résultats avec l'enseignant.
    - https://docs.google.com/spreadsheets/d/1Kda9G4cP1b4wODJr_UzzmCj8oxiyYsEgYVZsLEO3rFo/edit?usp=sharing
  - Comme tout le monde sera appelé au hasard, et que vous pouvez vous répartir les tâches, il est bon que le membre concerné prépare une fiche d'aide (cheat sheet) et informe les autres des réponses à donner.
    - Quand on vous appelle, vous répondez d'abord, puis vous pouvez demander à un coéquipier de compléter votre réponse.

Pour la présentation, vous vous tenez juste à côté de l'écran et **n'avez pas le droit d'utiliser de fiche ni de lire les slides**. Il faut retenir les choses par cœur ou refaire la déduction au tableau avec craie. Si deux personnes du groupe ne savent pas expliquer le concept, l'enseignant regardera le travail produit par GenAI et **la note de toute l'équipe sera diminuée**.

### Utilisation de GenAI pour préparer la soumission

Identifier clairement quelles parties ont utilisé une assistance GenAI. Inclure dans le rapport une courte déclaration d'utilisation GenAI listant l'outil/modèle, les sections du rapport, figures, cellules du notebook ou fichiers de code concernés, à quoi l'outil a servi, et comment vous avez vérifié et corrigé sa sortie. Rendre le code assisté identifiable dans le notebook ou le dépôt. Si aucune assistance GenAI n'a été utilisée, le déclarer explicitement. Chaque membre reste responsable de comprendre et d'expliquer le travail soumis.

### Collaboration de groupe

Sur les feuilles de contribution individuelles, noter la contribution de chaque membre à cette discussion et cette revue.

### Critères de notation

Des réponses de base correctes à chaque tâche cœur donnent droit à un 3,0. Le 1,0 exige un travail cœur constamment bien étayé et une réponse argumentée à la question ouverte. La réponse à la question ouverte seule ne distingue pas 1,0 de 2,0.

1,0 aux devoirs = 90 points ; 2,0 = 80 points ; 3,0 = 70 points ; 4,0 = 60 points.

### Partie 1 : 25 % — Vérification formelle

Pour cette partie, on pourra vous demander de refaire la dérivation au tableau, pour les points « a) » à « c) ».

A. Pour le réseau de neurones, utiliser la **propagation de bornes** (bound propagation) pour dériver les valeurs min et max de chaque neurone.
B. Pouvez-vous **prouver que y1 > y2** par propagation de bornes ?

C. Pour l'encodage MILP, pour le neurone « a », écrire la **somme pondérée linéaire** et l'**encodage big-M du ReLU** ; utiliser **M = 1 000 000**.
D. [Programmation] Encoder maintenant **tout le réseau en MILP**, et calculer la **valeur minimale de y1 − y2**.
E. **y1 > y2 est-il vrai** d'après la méthode exacte (MILP) ?

[Schéma du réseau — voir PDF page 3] : entrées x1, x2, x3 avec **X = [-1, 1] × [0, 1] × [0, 1]** ; h = ReLU(im_h) = max(0, im_h).

### Partie 2 : 50 % — Techniques de robustesse

A. Prendre le jeu de données **MNIST** et entraîner un **classifieur DNN simple** qui reconnaît les chiffres, en n'utilisant que **ReLU et linéaire en alternance**, avec un **minimum de 5 couches** (linéaire → ReLU → linéaire → ReLU → linéaire).

a. Pour chaque valeur de pixel, la mettre à l'échelle pour être entre **-1 et 1**.
b. Faire un **shuffling correct** et utiliser **l'ID de votre groupe comme graine aléatoire**.
c. Selon votre groupe, montrer l'image n° **(GROUP_ID·8·i)** et la prédiction résultante, pour i = {1, 2, ..., 15}.

B. Prendre **5 images** qui ont une prédiction correcte du DNN et leur appliquer **FGSM** (Fast Gradient Sign Method) avec différentes intensités (à partir de **0,001** et au-delà) jusqu'à ce que le DNN change de classe.

a. Pour chaque image, quelle est **l'intensité FGSM nécessaire pour faire basculer la classe** ?
b. À la présentation orale, vous devez être capable d'expliquer **l'astuce d'attaque de FGSM**.

C. [Pour 1,0] Calculer la **constante de Lipschitz de chaque couche**. Pour le changement d'intensité en FGSM, quelle est la **variation maximale de sortie** d'après l'analyse de Lipschitz ?

D. Maintenant, prendre le DNN et la même entrée perturbée par FGSM, et appliquer le **randomized smoothing** (lissage aléatoire), avec **100 échantillons** à chaque application, sur les **5 mêmes images perturbées**, avec **trois intensités de bruit gaussien** (0,01, 0,1, 0,5).

a. Pour chaque image, la prédiction basée sur le randomized smoothing **s'améliore-t-elle** ?

E. [Pour 1,0] Quantifier maintenant l'entrée pour que toutes les valeurs de pixels réelles dans [-1, 1] soient **discrétisées en un ensemble limité de valeurs possibles** ; la façon de discrétiser est à votre choix, aussi simple qu'une partition régulière, par ex. {-1, -0,5, 0, 0,5, 1}.

a. Entraîner un **autre DNN** avec la même architecture, mais cette fois en n'utilisant en entrée que des valeurs discrétisées.
b. Pouvez-vous atteindre les **mêmes performances** avec le **même nombre d'époques** ?
c. Appliquer FGSM à nouveau, et voir s'il est possible de **faire basculer les classes** ?
d. D'après c), développer une **meilleure méthode de partitionnement des entrées**, et donner votre **justification**.

### Partie 3 : 25 % — Calibration & technique OoD

[Schéma — voir PDF page 5]

A. Mesurer le DNN en créant le **reliability diagram** (diagramme de fiabilité) pour les données MNIST.

[Calibration en pratique — voir PDF page 6]

B. Prendre le **même classifieur DNN** entraîné sur MNIST, et lui donner maintenant les images de **Fashion-MNIST**.

a. Selon votre groupe, montrer l'image n° **(GROUP_ID·8·i)**, la prédiction résultante et sa **probabilité de sortie (après softmax)**, pour i = {1, 2, ..., 10}.
b. Donner **100 images aléatoires de Fashion-MNIST**, et obtenir la probabilité de sortie de chaque prédiction ; tracer l'**histogramme de la distribution des probabilités de sortie**, avec des bins de taille 0,1.
c. Prendre **une méthode donnée en cours**, et vérifier si l'on peut **filtrer avec succès les données OoD** (prendre 50 échantillons), tout en pouvant **accepter les données ID**.
d. [Pour 1,0] Prendre **une autre méthode donnée en cours**, et comparer leurs performances.
e. [Pour 1,0] Pensez-vous que votre **évaluation des performances est équitable** ? Pourquoi ?
f. [Pour 1,0] Essayer maintenant votre détecteur OoD avec **Kuzushiji-MNIST (KMNIST)** et **EMNIST** (alphabets uniquement), et rapporter les performances.

[Exemples d'images KMNIST et EMNIST — voir PDF page 7]
