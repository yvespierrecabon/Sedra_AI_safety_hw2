# Sedra_AI_safety_hw2

Projet de soutien pour le **Devoir 2 du cours « AI Safety – Concepts and Techniques »** (Université d'Oldenbourg) — groupe de 4 étudiants (Sedra, 5ᵉ année, option cyber, + 3 coéquipiers), accompagné en parallèle.

- **Échéance stricte : 15 décembre 2026, 12h00 heure d'Allemagne**
- **Oral : 16 ou 17 décembre 2026**, créneau de 45 min (inscription via feuille Google partagée)
- **Partie 1 (25 %)** : vérification formelle d'un petit réseau — bound propagation + encodage MILP (big-M ReLU)
- **Partie 2 (50 %)** : robustesse d'un DNN MNIST — FGSM, constante de Lipschitz, randomized smoothing, quantification des entrées
- **Partie 3 (25 %)** : calibration & détection OoD — reliability diagram, Fashion-MNIST / KMNIST / EMNIST

Sujet officiel : [docs/AISafety_HW2.pdf](docs/AISafety_HW2.pdf) · Traduction FR : [docs/AISafety_HW2_traduction_fr.md](docs/AISafety_HW2_traduction_fr.md) · Concepts expliqués : [docs/AISafety_HW2_explications_concepts.md](docs/AISafety_HW2_explications_concepts.md)

Documentation : [docs/CONTEXT.md](docs/CONTEXT.md) (contexte persistant) · [docs/AVANCEMENT.md](docs/AVANCEMENT.md) (suivi détaillé).

## Livrables (ZIP à rohan.menon@uol.de)

- Rapport PDF (avec déclaration d'utilisation GenAI)
- Note visée par partie, en PDF
- Fichier texte avec le lien du repo du groupe
- Notebook Jupyter (.ipynb) exécuté, code + sorties enregistrées
- Export HTML ou PDF du notebook exécuté
- Feuille de contribution individuelle (chacun, séparément, à rohan.menon@uol.de)

## État d'avancement (résumé)

Voir le tableau détaillé dans [docs/AVANCEMENT.md](docs/AVANCEMENT.md).

| Partie | Contenu | État |
|---|---|---|
| P1 (25 %) | bound propagation, MILP, min(y1−y2) | ⬜ non commencé |
| P2 (50 %) | DNN MNIST, FGSM, Lipschitz, smoothing, quantification | ⬜ non commencé |
| P3 (25 %) | calibration, OoD (Fashion-MNIST, KMNIST, EMNIST) | ⬜ non commencé |
| Livrables | notebook + export, rapport PDF + GenAI, notes visées, ZIP | ⬜ non commencés |

## Structure du repo

```
docs/       Sujet officiel (PDF), traduction FR, explications de concepts, CONTEXT.md, AVANCEMENT.md
notebooks/  Notebooks Jupyter exécutés (une partie par notebook ou un notebook par partie)
scripts/    Scripts auxiliaires (ex. encodage MILP réutilisable)
report/     Rapport et figures
setup/      install_env.sh (création du venv .venv, PyTorch CPU)
```

## Mise en place

```bash
git clone https://github.com/yvespierrecabon/Sedra_AI_safety_hw2.git
cd Sedra_AI_safety_hw2
bash setup/install_env.sh          # venv .venv + PyTorch CPU + requirements.txt
source .venv/bin/activate
```

Données MNIST / Fashion-MNIST / KMNIST / EMNIST téléchargées automatiquement par `torchvision` dans `data/` (git-ignoré).

## Conventions

- Lire `docs/CONTEXT.md` à chaque nouvelle session ; synchroniser le dépôt (`git pull origin main`) en début de session
- Mettre à jour `docs/AVANCEMENT.md` après chaque tâche significative
- Le notebook doit être exécutable et reproductible (graine = ID du groupe) avec les sorties enregistrées
- Toute utilisation de GenAI est tracée (outil, sections/cellules, vérifications) pour la déclaration obligatoire du rapport
- Travail directement sur `main` (convention du projet)
