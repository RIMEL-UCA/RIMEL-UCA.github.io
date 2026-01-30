#!/bin/bash
set -e

echo "=== Pipeline d'analyse automatique ==="
echo ""
REPLAY=0
if [ "$1" = "--replay" ] || [ "$1" = "-r" ]; then
    REPLAY=1
    echo "Mode replay: utilisation des SHA dans repos_url.csv"
fi
echo "Construction de l'image Docker..."
# Supprimer les warnings bruyants de docker-compose
docker-compose build --quiet >/dev/null 2>&1 || true
echo -e "	Image prête"
echo ""

echo "=== Etape 0/1: Analyse SonarQube + commits + contributeurs ==="
if [ "$REPLAY" -eq 1 ]; then
    bash 1-qualite/scan_repos.sh --replay
else
    bash 1-qualite/scan_repos.sh
fi
echo ""

echo "=== Etape 1/3: Analyse des types de commits ==="
docker-compose run --rm analysis python 3-activite-contributeurs/get_commits_types.py 2>/dev/null
echo ""

if [ -f "3-activite-contributeurs/data/commits_other_for_ml.csv" ]; then
    echo "=== Etape ML: Classification des commits non reconnus ==="
    docker-compose run --rm analysis python 3-activite-contributeurs/train_and_apply_commit_classifier.py 2>/dev/null
    echo ""
else
    echo -e "	Aucun dataset ML annoté trouvé — étape ML ignorée"
fi

echo "=== Etape 2/3: Generation des graphiques ==="
echo "  -> Graphiques qualite (question 1)..."
docker-compose run --rm analysis python 1-qualite/generate-graphs.py 2>/dev/null
docker-compose run --rm analysis python 1-qualite/generate-violin-graph.py 2>/dev/null
echo "  -> Graphiques qualite par groupe (question 2)..."
docker-compose run --rm analysis python 2-nombre-contributeurs/compute_repo_groups.py 2>/dev/null
docker-compose run --rm analysis python 2-nombre-contributeurs/generate_quality_graphs.py 2>/dev/null
echo "  -> Graphiques activite (question 3)..."
docker-compose run --rm analysis python 3-activite-contributeurs/generate_graphs.py 2>/dev/null
echo ""

echo "=== Pipeline termine avec succes! ==="
echo ""
echo "Resultats:"
echo "  - 1-qualite/outputs/summary.csv"
echo "  - 1-qualite/outputs/reports/*.csv"
echo "  - 1-qualite/outputs/*.png (question 1)"
echo "  - 2-nombre-contributeurs/data/contributors.csv"
echo "  - 2-nombre-contributeurs/graphs/*.png (question 2)"
echo "  - 3-activite-contributeurs/data/commits_types.csv"
echo "  - 3-activite-contributeurs/outputs/graphs/*.png (question 3)"