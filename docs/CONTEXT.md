# Contexte Persistant - Projet AI Safety (Devoir 2 : vérification formelle, robustesse, calibration/OoD)

## **Informations Générales**
- **Projet** : [Sedra_AI_safety_hw2](https://github.com/yvespierrecabon/Sedra_AI_safety_hw2)
- **Répertoire Local (utilisateur)** : à confirmer (projet séparé du Devoir 1)
- **Répertoire Sandbox** : `/workspace/github__yvespierrecabon__Sedra_AI_safety_hw2`
- **Objectif** : aider une étudiante en 5ᵉ année d'école d'ingénieurs (option cyber) à comprendre et réaliser le **Devoir 2 du cours « AI Safety – Concepts and Techniques »** (Université d'Oldenbourg), afin de pouvoir lui expliquer ensuite. Travail en groupe de 4 étudiants ; accompagnement en parallèle sur l'ordinateur d'Yves. Le Devoir 1 a été fait dans le dépôt `Sedra_IA_safety` (même groupe).
- **Utilisateur** : yvespierrecabon (accompagne l'étudiante, veut comprendre pour expliquer).
- **Échéance stricte** : 15 décembre 2026, 12h00 heure d'Allemagne.
- **Oral** : 16 ou 17 décembre 2026, créneau de 45 min pour tout le groupe (inscription : feuille Google partagée dans le sujet). Interrogation aléatoire de chaque membre.

---

## **Le devoir (sujet officiel : `docs/AISafety_HW2.pdf`, 7 pages)**
Soumission en groupe : un ZIP à **rohan.menon@uol.de** contenant le **rapport PDF** (avec déclaration GenAI), la **note visée par partie en PDF**, un **fichier texte avec le lien du repo du groupe**, le **notebook Jupyter exécuté** (.ipynb, code + sorties) et son **export HTML/PDF**. **Feuille de contribution individuelle** envoyée séparément par chaque membre au même adresse.

### Barème
- 1,0 = 90 pts · 2,0 = 80 pts · 3,0 = 70 pts · 4,0 = 60 pts.
- **3,0** = réponses de base correctes à toutes les tâches cœur. **1,0** = travail constamment bien étayé + réponse argumentée à la question ouverte (les tâches marquées « [For 1.0] » ne distinguent pas seules 1,0 de 2,0).

### Partie 1 (25 %) — Vérification formelle
Réseau donné sur le schéma (entrées **x1 ∈ [-1,1], x2 ∈ [0,1], x3 ∈ [0,1]**) :
- a = ReLU(+x1 + 2·x2 − x3),  b = ReLU(−x1 − 2·x2 + x3)  (z_b = −z_a : a et b sont les parties positive/négative de la même pré-activation z_a)
- c = ReLU(2·a + b + 1),  d = ReLU(a + 2·b + 1)
- **y1 = c + 2**,  **y2 = d**

- **A.** Bound propagation : dériver les min/max de chaque neurone sur le domaine d'entrée X = [-1,1]×[0,1]×[0,1].
- **B.** Peut-on prouver y1 > y2 par bound propagation ?
- **C.** Encodage MILP pour le neurone « a » : somme pondérée linéaire + encodage big-M du ReLU, avec **M = 1 000 000**.
- **D. [Programmation]** Encoder tout le réseau en MILP et calculer le **minimum de y1 − y2**.
- **E.** y1 > y2 est-il vrai selon la méthode exacte (MILP) ?
- À l'oral, derivation au tableau possible pour A–C.

### Partie 2 (50 %) — Techniques de robustesse (MNIST)
- **A.** Entraîner un DNN simple (uniquement couches linéaires et ReLU en alternance, **minimum 5 couches** : linear → ReLU → linear → ReLU → linear) :
  - a. pixels mis à l'échelle **[-1, 1]**
  - b. shuffling correct, **graine aléatoire = ID du groupe**
  - c. montrer l'image n° **(GROUP_ID·8·i)** et sa prédiction, pour i = 1..15
- **B.** 5 images correctement prédites + attaque **FGSM** à intensités croissantes (à partir de 0,001) jusqu'au changement de classe :
  - a. intensité nécessaire par image pour « flip » la classe
  - b. savoir expliquer l'astuce de FGSM à l'oral
- **C. [1,0]** Constante de Lipschitz de chaque couche ; variation max de sortie pour l'intensité FGSM d'après l'analyse de Lipschitz.
- **D.** **Randomized smoothing** (100 échantillons, bruits gaussiens 0,01 / 0,1 / 0,5) sur les 5 mêmes images perturbées : la prédiction s'améliore-t-elle ?
- **E. [1,0]** Quantifier les entrées (ex. {-1, -0,5, 0, 0,5, 1}) : a. réentraîner un DNN identique sur entrées discrétisées ; b. mêmes performances au même nombre d'époques ? ; c. FGSM peut-il encore flip des classes ? ; d. proposer une meilleure méthode de partitionnement + justification.

### Partie 3 (25 %) — Calibration & technique OoD
- **A.** **Reliability diagram** du DNN sur MNIST (diagramme de fiabilité : confiance vs fréquence réelle).
- **B.** Même DNN, entrées **Fashion-MNIST** :
  - a. image n° (GROUP_ID·8·i), prédiction + probabilité softmax, i = 1..10
  - b. 100 images Fashion-MNIST aléatoires → histogramme des probabilités de prédiction (bins de 0,1)
  - c. une méthode OoD vue en cours pour filtrer 50 échantillons OoD tout en acceptant les ID
  - d. [1,0] une deuxième méthode vue en cours, comparer les performances
  - e. [1,0] l'évaluation est-elle équitable ? Pourquoi ?
  - f. [1,0] tester le détecteur OoD sur **KMNIST** et **EMNIST** (alphabets seuls), rapporter les performances

---

## **Concepts Clés (détaillés dans `docs/AISafety_HW2_explications_concepts.md`)**
- **Bound propagation** : arithmétique d'intervalles couche par couche — propager des bornes [min, max] de chaque neurone (linéaire : bornes via signes des poids ; ReLU : max(0, ·)). Approximation par le haut (sound) : si min(y1−y2) > 0, alors y1 > y2 prouvé ; sinon on ne conclut pas.
- **MILP big-M pour ReLU** : z = somme pondérée + biais ; a = ReLU(z) encodé avec variable binaire δ et grand M : a ≥ 0, a ≥ z, a ≤ z + M(1−δ), a ≤ Mδ, z ≤ Mδ, z ≥ −M(1−δ). Le solveur MILP donne la valeur **exacte** de min(y1−y2).
- **FGSM** : attaque « one-step » white-box : x' = x + ε·sign(∇ₓ J(θ, x, y)) ; ε = intensité. Un pas dans la direction du gradient de la perte.
- **Constante de Lipschitz** : pour une couche linéaire, c'est la norme spectrale ‖W‖₂ ; le produit des constantes des couches borne la variation de sortie. Compare à l'amplitude ε de FGSM.
- **Randomized smoothing** : prédire par vote majoritaire sur N échantillons bruités x + N(0, σ²I) ; donne un classifieur avec rayon certifié.
- **Quantification des entrées** : discrétiser [-1,1] en une grille finie, réentraîner dessus ; l'attaque FGSM continue de marcher sur la grille mais la défense repose sur l'entraînement robuste à la grille.
- **Calibration** : reliability diagram (bins de confiance vs accuracy réelle), ECE ; un modèle calibré a confiance ≈ fréquence réelle de justesse.
- **OoD** : détecter les entrées hors distribution (Fashion-MNIST, KMNIST, EMNIST) malgré des softmax confiants — méthodes classiques : MSP (max softmax), energy score, ODIN, Mahalanobis (choisir celles vues en cours).

---

## **Environnement Technique**
- **Langage** : Python 3.12, venv `.venv` (créé par `setup/install_env.sh`).
- **Bibliothèques installées (sandbox)** : torch 2.15.0+cpu, torchvision 0.29.2+cpu, numpy, scipy, matplotlib, pandas, scikit-learn, jupyter/nbformat/nbconvert/ipykernel.
- **MILP** : `scipy.optimize.milp` (solveur HiGHS embarqué) — aucune dépendance externe.
- **Données** : MNIST, Fashion-MNIST, KMNIST, EMNIST via `torchvision.datasets` (téléchargement dans `data/`, git-ignoré).
- **Notebooks** : un par partie (P1, P2, P3), exécutés avec sorties enregistrées + export HTML.

---

## **Structure du Dépôt**
- `docs/` : sujet officiel `AISafety_HW2.pdf`, traduction FR `AISafety_HW2_traduction_fr.md`, concepts `AISafety_HW2_explications_concepts.md`, `CONTEXT.md`, `AVANCEMENT.md`
- `notebooks/` : notebooks Jupyter exécutés (Parties 1–3)
- `scripts/` : outils réutilisables (ex. encodage MILP)
- `report/` : rapport et figures
- `setup/` : `install_env.sh` ; `requirements.txt`, `.gitignore` à la racine

---

## **Règles de Travail**
1. **Persistance** : lire ce fichier **à chaque nouvelle session**.
2. **Synchronisation systématique** : en début de session, `git fetch --all && git pull origin main`.
3. **Mises à jour** : mettre à jour `docs/AVANCEMENT.md` après chaque tâche significative ; garder `CONTEXT.md` cohérent.
4. **Rôle pédagogique** : l'utilisateur veut **comprendre pour expliquer** à l'étudiante. Privilégier :
   - **expliquer ce qui est demandé et comment on va y répondre AVANT de proposer le code** ;
   - explications conceptuelles claires et interprétation des résultats ;
   - guider les choix expérimentaux (graines, métriques) sans faire le travail à la place — chaque membre doit pouvoir tout expliquer à l'oral.
5. **Clarté** : sections `##`/`###`, listes à puces pour les éléments dynamiques.

---

## **Points d'Attention**
- **ID du groupe (β) = 5** (confirmé au Devoir 1, même groupe) → **graine 5** ; images n° **5·8·i = 40·i** (i=1..15 en Partie 2 ; i=1..10 en Partie 3). À re-confirmer si l'équipe pédagogique a changé la numérotation.
- **Déclaration GenAI obligatoire** : outil/modèle, sections/figures/cellules concernées, usage, comment la sortie a été vérifiée/corrigée ; code assisté identifiable dans le notebook/repo. Sinon, le déclarer explicitement.
- **Oral** : interrogation aléatoire → cheat sheet commune ; chacun doit pouvoir refaire la dérivation de la Partie 1 (A–C) au tableau et expliquer FGSM.
- **Reproductibilité** : graine = ID du groupe, notebook exécutable avec sorties enregistrées, export HTML/PDF.
- **M = 1 000 000** imposé pour l'encodage big-M du ReLU (Partie 1-C).

---

*Dernière modification : session initiale de mise en contexte (Devoir 2 AI Safety)*
