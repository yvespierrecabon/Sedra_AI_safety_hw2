#!/usr/bin/env bash
# Installation de l'environnement HW2 (profil CPU par défaut)
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m ipykernel install --user --name aisafety-hw2 --display-name "Python 3 (aisafety-hw2)"

echo "OK — active l'environnement avec : source .venv/bin/activate"
