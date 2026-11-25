# Historique d'Avancement - Projet AI Safety : Devoir 2 (Vérification formelle, Robustesse, Calibration/OoD)

> Suit le projet **Sedra_AI_safety_hw2** : Devoir 2 du cours « AI Safety – Concepts and Techniques » (Université d'Oldenbourg), sujet `docs/AISafety_HW2.pdf`. À mettre à jour régulièrement, en cohérence avec `docs/CONTEXT.md`. Le Devoir 1 était suivi dans le dépôt `Sedra_IA_safety` (même groupe, β = 5).

---

## **État Actuel du Projet**
*Snapshot de l'avancement global*

### **Informations à confirmer**
- [x] **ID du groupe (β) = 5** ✅ (confirmé au Devoir 1, même groupe) → graine **5**, images n° **40·i**
- [ ] **Notes visées** par partie (déclarées dans le ZIP)
- [ ] **Découpage des tâches** entre les 4 étudiants (Sedra + 3 coéquipiers)
- [ ] Inscription au **créneau d'oral** (16 ou 17 déc. 2026, feuille Google du sujet)

### **Partie 1 (25 %) — Vérification formelle**
- [x] Sujet extrait et traduit (`docs/AISafety_HW2_traduction_fr.md`), réseau relu depuis le schéma
- [ ] **A.** Bound propagation : min/max de chaque neurone (a, b, c, d, y1, y2) sur X = [-1,1]×[0,1]×[0,1]
- [ ] **B.** y1 > y2 prouvable par bound propagation ?
- [ ] **C.** Encodage MILP du neurone « a » (somme pondérée + big-M ReLU, M = 10⁶)
- [ ] **D. [Prog]** MILP du réseau complet → minimum de y1 − y2 (`scipy.optimize.milp`, HiGHS)
- [ ] **E.** Conclusion exacte : y1 > y2 tient-il ?

### **Partie 2 (50 %) — Robustesse (MNIST)**
- [ ] **A.** DNN ≥ 5 couches (linear/ReLU alternés) : pixels [-1,1], shuffle graine 5, images 40·i (i=1..15) + prédictions
- [ ] **B.** FGSM sur 5 images correctes, intensités dès 0,001 jusqu'au flip ; intensité de flip par image
- [ ] **C. [1,0]** Constante de Lipschitz par couche (norme spectrale) ; variation max de sortie pour l'intensité FGSM
- [ ] **D.** Randomized smoothing (100 éch., σ = 0,01 / 0,1 / 0,5) sur les 5 images perturbées
- [ ] **E. [1,0]** Quantification des entrées (ex. {-1,-0,5,0,0,5,1}) : réentraînement, mêmes perfs ?, FGSM flip ?, meilleure partition + rationale

### **Partie 3 (25 %) — Calibration & OoD**
- [ ] **A.** Reliability diagram du DNN sur MNIST
- [ ] **B.** Fashion-MNIST dans le même DNN : images 40·i (i=1..10) + proba softmax ; histogramme des probas sur 100 images (bins 0,1) ; méthode OoD du cours (50 échantillons OoD filtrés, ID acceptés)
- [ ] **D–F. [1,0]** deuxième méthode OoD + comparaison ; équité de l'évaluation ; KMNIST + EMNIST (alphabets)

### **Livrables**
- [ ] Notebooks exécutés (code + sorties) + exports HTML/PDF
- [ ] Rapport PDF avec déclaration GenAI
- [ ] Note visée par partie (PDF)
- [ ] Fichier texte avec le lien du repo
- [ ] ZIP envoyé à rohan.menon@uol.de
- [ ] Feuilles de contribution individuelles
- [ ] Cheat sheet commune pour l'oral

---

## **Objectifs Actuels**
- [ ] 🔍 **Priorité 1** : confirmer que β = 5 s'applique aussi au Devoir 2 (même groupe) et que la convention « image n° GROUP_ID·8·i » est comprise de la même façon
- [ ] **Priorité 2** : lancer la **Partie 1** (100 % local, CPU, rapide : bound propagation à la main + MILP avec scipy/HiGHS) — bon candidat pour démarrer
- [ ] **Priorité 3** : Partie 2 (entraînement DNN MNIST en CPU, puis FGSM/smoothing/quantification)
- [ ] **Priorité 4** : Partie 3 (calibration + OoD sur Fashion-MNIST/KMNIST/EMNIST)

---

## **Arborescence du Dépôt**
```
Sedra_AI_safety_hw2/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── AISafety_HW2.pdf                        # sujet officiel
│   ├── AISafety_HW2_traduction_fr.md            # traduction FR du sujet
│   ├── AISafety_HW2_explications_concepts.md   # concepts expliqués par partie
│   ├── CONTEXT.md                              # contexte persistant
│   └── AVANCEMENT.md                           # ce fichier
├── notebooks/                                  # notebooks exécutés (P1–P3)
├── scripts/                                    # outils (ex. MILP)
├── report/                                     # rapport et figures
└── setup/
    └── install_env.sh                          # venv .venv + PyTorch CPU
```
💡 Données (MNIST, Fashion-MNIST, KMNIST, EMNIST) téléchargées dans `data/`, hors repo (git-ignoré).

---

## **Points d'Attention**
- **Échéance stricte** : 15 décembre 2026, 12h00 heure d'Allemagne (ZIP à rohan.menon@uol.de).
- **Oral 16–17 déc.** : interrogation aléatoire de chaque membre → cheat sheet commune ; dérivation Partie 1 (A–C) et explication de FGSM à savoir refaire au tableau.
- **GenAI** : déclaration précise obligatoire dans le rapport (outil, sections/cellules, vérifications).
- **Reproductibilité** : graine = 5 (ID groupe), notebooks exécutés avec sorties, export HTML/PDF.
- **Partie 1** : M = 1 000 000 imposé pour le big-M ; la bound propagation peut ne pas conclure même si la propriété est vraie → le MILP tranche exactement.

---

## **Journal des Actions**
*(Ordre chronologique inverse)*

---

### **Session 1 — Mise en contexte du Devoir 2**

#### **Actions Réalisées**
- Extraction du sujet officiel `docs/AISafety_HW2.pdf` (7 pages, OCR) : consignes, barème, Parties 1–3.
- Lecture du schéma de la Partie 1 : entrées x1∈[-1,1], x2∈[0,1], x3∈[0,1] ; a=ReLU(−x1+2x2−x3), b=ReLU(x1+x2+x3) ; c=ReLU(2a+b+1), d=ReLU(a+2b+1) ; y1=c+2, y2=d.
- Création de `docs/CONTEXT.md` (contexte persistant) et de ce fichier ; `README.md`, `.gitignore`, `requirements.txt`, `setup/install_env.sh`.
- Rédaction de la traduction FR du sujet : `docs/AISafety_HW2_traduction_fr.md`.
- Rédaction des explications de concepts par partie : `docs/AISafety_HW2_explications_concepts.md`.
- Environnement sandbox : venv `.venv` (Python 3.12) avec torch 2.15.0+cpu, torchvision 0.29.2+cpu, numpy, scipy, matplotlib, pandas, scikit-learn, jupyter/nbconvert. MILP via `scipy.optimize.milp` (HiGHS). Réseau PyPI accessible ; données MNIST via torchvision à tester.
- Hypothèse retenue : **β = 5** (ID groupe confirmé au Devoir 1) → graine 5, images 40·i.

#### **Prochaines Actions Immédiates**
- Après `git pull` de l'utilisateur : décider par quoi commencer (proposition : Partie 1, 100 % local).
- Tester le téléchargement des datasets torchvision (MNIST d'abord).

---

## **Calendrier des Sessions**
- **Session 1** : mise en contexte Devoir 2 ; traduction + concepts + environnement sandbox.

---

## **Tags Utilisés**
- ✅ : Tâche terminée
- 🔍 : En cours
- ❌ : Bloquée
- 💡 : Contexte important
- ⚠️ : Problème / attention
- 👀 : Observation
- 📝 : Documentation

---

*Dernière modification : Session 1 — mise en contexte Devoir 2*
