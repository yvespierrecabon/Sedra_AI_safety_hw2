# Devoir 2 — Explications concrètes des concepts, par partie

> Objectif pédagogique : comprendre chaque concept **avant** d'écrire le code, pour pouvoir l'expliquer à l'oral (interrogation aléatoire de chaque membre). Les schémas du sujet sont dans `docs/AISafety_HW2.pdf`.

---

## Partie 1 (25 %) — Vérification formelle d'un petit réseau

### Le réseau à vérifier
Entrées : **x1 ∈ [-1, 1], x2 ∈ [0, 1], x3 ∈ [0, 1]** (le domaine d'entrée X est un pavé).
- a = ReLU(+x1 + 2·x2 − x3)
- b = ReLU(−x1 − 2·x2 + x3)  →  z_b = −z_a, donc **a·b = 0** (un seul des deux est non nul, sauf si z_a = 0)
- c = ReLU(2·a + b + 1)
- d = ReLU(a + 2·b + 1)
- **y1 = c + 2**, **y2 = d**

Question : sur tout le domaine X, a-t-on toujours **y1 > y2** ?

### A. Bound propagation (propagation de bornes = arithmétique d'intervalles)
Principe : au lieu de propager des valeurs ponctuelles, on propage des **intervalles [min, max]** pour chaque neurone.
- Entrées : [x1] = [-1, 1], [x2] = [0, 1], [x3] = [0, 1].
- Couche linéaire z = Σ wᵢ·xᵢ + b : le min de z s'obtient en prenant, pour chaque poids wᵢ, le min de xᵢ si wᵢ > 0 et le max de xᵢ si wᵢ < 0 (et inversement pour le max de z).
- ReLU : [ReLU(z)] = [max(0, min z), max(0, max z)].
On obtient ainsi des bornes pour a, b, c, d, puis y1, y2.

### B. Peut-on prouver y1 > y2 ?
On regarde **min(y1 − y2) = min(y1) − max(y2)** (borne inférieure de la différence). Si cette borne est **> 0**, alors y1 > y2 est prouvé partout. Sinon, la bound propagation **ne conclut pas** — ce n'est pas une preuve de l'inverse : les bornes sont une sur-approximation (les min et max de y1 et y2 ne sont pas atteints au même point d'entrée).

### C. Encodage MILP du neurone « a » (big-M, M = 1 000 000)
Somme pondérée (pré-activation) : **z_a = +x1 + 2·x2 − x3 + 0** (biais nul) ; z_b = −z_a.
ReLU exact par variable binaire δ ∈ {0,1} et grand M :
- a ≥ 0
- a ≥ z_a
- a ≤ z_a + M·(1 − δ)
- a ≤ M·δ
- z_a ≤ M·δ
- z_a ≥ −M·(1 − δ)

Si δ = 1 : z_a ≥ 0 et a = z_a (ReLU actif). Si δ = 0 : z_a ≤ 0 et a = 0. Le solveur MILP choisit δ de façon optimale — d'où l'**exactitude**.

### D–E. MILP complet et conclusion exacte
On encode de même b, c, d (chaque ReLU avec son propre δ), les sorties y1 = c + 2, y2 = d, et on **minimise y1 − y2** sous les contraintes de bornes d'entrée. Si min(y1 − y2) > 0 → **y1 > y2 est vrai exactement** ; si ≤ 0, la propriété est fausse (et le solveur fournit un contre-exemple).

Implémentation prévue : `scipy.optimize.milp` (solveur HiGHS embarqué, aucune installation externe).

### Vérification à la main (résultats attendus — à confirmer dans le notebook)
Propagation de bornes, neurone par neurone (z_a = x1 + 2x2 − x3 ∈ [−2, 3], z_b = −z_a ∈ [−3, 2]) :

| neurone | min | max |
|---|---|---|
| a = ReLU(z_a) | 0 | 3 |
| b = ReLU(z_b) | 0 | 2 |
| c = ReLU(2a + b + 1) | 1 | 9 |
| d = ReLU(a + 2b + 1) | 1 | 8 |
| y1 = c + 2 | 3 | 11 |
| y2 = d | 1 | 8 |

- **B.** Borne inf de y1 − y2 par bound propagation : min(y1) − max(y2) = 3 − 8 = **−5 < 0** → **on ne peut pas prouver y1 > y2** (la propagation traite a et b comme indépendants, alors que a·b = 0 : sur-approximation).
- **D–E.** En réalité, comme a·b = 0, on a partout **y1 − y2 = z_a + 2 = x1 + 2x2 − x3 + 2**, donc min(y1 − y2) = min(z_a) + 2 = −2 + 2 = **0**, atteint au coin (x1, x2, x3) = (−1, 0, 1). Conclusion exacte : **y1 > y2 est FAUX** (égalité y1 = y2 = 5 en ce point). Le MILP doit retrouver min = 0 exactement.

**Pour l'oral** : savoir refaire la propagation de bornes d'une couche au tableau, et écrire l'encodage big-M d'un ReLU.

---

## Partie 2 (50 %) — Robustesse d'un classifieur MNIST

### A. Le DNN de base
- Architecture : uniquement **Linear → ReLU** en alternance, **≥ 5 couches** (ex. 784 → 128 → ReLU → 128 → ReLU → 10).
- Pixels mis à l'échelle dans **[-1, 1]** (au lieu de [0, 1]).
- **Shuffle avec graine = ID du groupe (β = 5)** avant l'entraînement (reproductibilité).
- Montrer les images n° **40·i** (i = 1..15) du jeu (de test, après shuffle) avec la prédiction du modèle.

### B. FGSM (Fast Gradient Sign Method)
Attaque **white-box en un seul pas** : pour une image x de vraie classe y,
**x' = x + ε · sign(∇ₓ J(θ, x, y))**
où J est la perte de classification et ε l'**intensité** (le sujet démarre à 0,001 et augmente jusqu'au changement de classe). L'astuce : un petit pas dans la direction du **signe du gradient de la perte** augmente la perte le plus vite possible pour un budget ε donné. On cherche, pour chacune des 5 images, le **ε minimal qui fait basculer la classe**.

### C. [1,0] Constante de Lipschitz par couche
Une fonction f est L-Lipschitz si ‖f(x) − f(x')‖ ≤ L·‖x − x'‖. Pour une couche linéaire x ↦ Wx + b, la constante de Lipschitz optimale est la **norme spectrale ‖W‖₂** (plus grande valeur singulière). Pour le réseau complet (composition), le produit des constantes borne la variation de sortie : **‖f(x') − f(x)‖ ≤ (∏ Lᵢ)·‖x' − x‖**. Avec FGSM, ‖x' − x‖∞ = ε, donc ‖x' − x‖₂ ≤ ε·√d (d = 784) : on compare cette borne à la variation de sortie observée.

### D. Randomized smoothing (lissage aléatoire)
Au lieu de prédire f(x) directement, on prédit la classe majoritaire parmi **N = 100** prédictions sur des échantillons bruités **x + N(0, σ²I)**, pour σ ∈ {0,01, 0,1, 0,5}. Principe : le bruit gaussien « moyenne » les pics adversariaux locaux ; on teste si la prédiction sur les 5 images FGSM redevient correcte.

### E. [1,0] Quantification des entrées
Discrétiser chaque pixel de [-1, 1] en une **grille finie** (ex. {-1, -0,5, 0, 0,5, 1} → 5 niveaux). Réentraîner un DNN identique sur ces entrées quantifiées. Questions : mêmes performances ? FGSM peut-il encore flip ? Si oui, améliorer le partitionnement (grille plus fine, ou partition non uniforme adaptée à la distribution des pixels de MNIST) et justifier.

---

## Partie 3 (25 %) — Calibration & détection OoD

### A. Reliability diagram (diagramme de fiabilité)
On regroupe les prédictions de test MNIST en **bins de confiance** (ex. 10 bins de largeur 0,1 sur la probabilité softmax max). Pour chaque bin, on trace **confiance moyenne vs fréquence réelle de justesse**. Un classifieur **calibré** a sa courbe sur la diagonale. Écart moyen pondéré = **ECE** (Expected Calibration Error).

### B. OoD avec Fashion-MNIST
Même DNN (entraîné sur MNIST), entrées Fashion-MNIST (vêtements) :
- images n° 40·i (i = 1..10) avec prédiction + proba softmax (souvent très confiantes à tort → le problème OoD) ;
- histogramme des probas softmax sur 100 images Fashion-MNIST (bins de 0,1) ;
- **une méthode OoD vue en cours** pour séparer ID (MNIST) et OoD (50 échantillons Fashion-MNIST) — candidats classiques : **MSP** (max softmax probability, baseline), **energy score**, **ODIN**, **Mahalanobis** (choisir celles du cours) ;
- [1,0] une **deuxième méthode** + comparaison (ex. AUROC, FPR à 95 % de TPR) ;
- [1,0] équité de l'évaluation : seuil réglé sur un jeu OoD marche-t-il sur un autre ? biais de seuil, etc. ;
- [1,0] généralisation à **KMNIST** (caractères japonais) et **EMNIST** (lettres) : mêmes performances ?

---

## Paramètres du groupe
- **GROUP_ID = β = 5** → graine aléatoire 5 ; images n° **5·8·i = 40·i** (i = 1..15 en Partie 2 ; i = 1..10 en Partie 3).
- M = **1 000 000** pour le big-M (Partie 1-C).
- Randomized smoothing : **100 échantillons**, σ ∈ {0,01, 0,1, 0,5}.
- Histogramme des probas OoD : **100 images**, bins de **0,1**.
