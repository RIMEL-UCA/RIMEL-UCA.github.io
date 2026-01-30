#!/bin/bash
# =============================================================================
# reproduce.sh - Reproductibilité des graphiques SQ1
# =============================================================================
# Ce script recrée tous les graphiques de la sous-question 1
# Pré-requis:
#   - Python 3.10+
#   - Fichier kaggle_state.json (session Kaggle authentifiée)
#   - Dossier out/leaderboards/ contenant les .zip téléchargés depuis Kaggle
# =============================================================================

set -e  # Arrêter en cas d'erreur

echo "=============================================="
echo " SQ1 - Reproduction des graphiques"
echo "=============================================="

# -----------------------------------------------------------------------------
# 0) Installation des dépendances Python
# -----------------------------------------------------------------------------
echo "[0/7] Installation des dépendances..."
pip install pandas matplotlib playwright --quiet
playwright install chromium --with-deps || true

# -----------------------------------------------------------------------------
# 1) Graphique 1 & 2: Timeline yuanzhezhou (compétitions/an + score)
#    -> yuanzhezhou_competitions_per_year.png
#    -> yuanzhezhou_performance_score_per_year.png
# -----------------------------------------------------------------------------
echo "[1/7] Scraping timeline yuanzhezhou..."
python sq1_scrape_timeline_user.py \
    --username yuanzhezhou \
    --state kaggle_state.json \
    --out_raw out/user_competitions_raw.csv \
    --out_figdir out/figures_sq1 \
    --headless \
    --max_scrolls 40

# -----------------------------------------------------------------------------
# 2) Graphique 3 & 4: Solo vs Equipe timeline (yuanzhezhou)
#    -> solo_vs_team_counts.png
#    -> solo_vs_team_ratios.png
# -----------------------------------------------------------------------------
echo "[2/7] Analyse solo vs équipe (timeline yuanzhezhou)..."
python sq1_solo_vs_team_timeline.py \
    --csv out/user_competitions_raw.csv \
    --state kaggle_state.json \
    --outdir out/figures_sq1 \
    --headless

# -----------------------------------------------------------------------------
# 3) Graphique 5: Heatmap des coéquipiers par année
#    -> teammates_heatmap_topN.png
# -----------------------------------------------------------------------------
echo "[3/7] Heatmap coéquipiers yuanzhezhou..."
python sq1_teammates_heatmap.py \
    --csv out/user_competitions_raw.csv \
    --state kaggle_state.json \
    --outdir out/figures_sq1 \
    --top_n 20 \
    --headless

# -----------------------------------------------------------------------------
# 4) Graphiques 6 & 7: Force équipe vs performance
#    -> team_strength_vs_perf_scatter.png
#    -> team_strength_vs_perf_boxplot.png
# -----------------------------------------------------------------------------
echo "[4/7] Force équipe vs performance..."
python sq1_team_strength_vs_performance.py \
    --competitions-csv out/user_competitions_raw.csv \
    --teammates-csv out/user_teammates_raw.csv \
    --state kaggle_state.json \
    --cache out/teammate_global_ranks.json \
    --outdir out/figures_sq1

# -----------------------------------------------------------------------------
# 5) Graphique 8: Solo vs Équipe par décile (agrégé sur plusieurs compétitions)
#    -> solo_vs_team_by_decile.png
#    Utilise les leaderboards .zip dans out/leaderboards/
# -----------------------------------------------------------------------------
echo "[5/7] Solo vs équipe par décile (offline, depuis ZIPs)..."
python sq1_solo_vs_team_deciles_from_csv.py \
    --leaderboards_dir out/leaderboards \
    --outdir out/figures_sq1 \
    --min_leaderboard_size 50

# -----------------------------------------------------------------------------
# 6) Graphique 9: Heatmap Collaboration Index - Fast Risers
#    -> fast_risers_collab_index_heatmap.png
# -----------------------------------------------------------------------------
echo "[6/7] Heatmap collaboration index (fast risers)..."
python sq1_compare_fast_risers_collab_heatmap.py \
    --users yuanzhezhou jsday96 daiwakun tomoon33 HarshitSheoran sayoulala \
    --state kaggle_state.json \
    --outdir out/figures_sq1 \
    --workdir out/fast_risers \
    --cache_ranks out/teammate_global_ranks.json \
    --headless \
    --max_scrolls 40

# -----------------------------------------------------------------------------
# 7) Graphique 10: Heatmap Top 25 collaboration profile (depuis ZIPs)
#    -> top25_collab_metrics_heatmap_from_zips_filtered.png
# -----------------------------------------------------------------------------
echo "[7/7] Heatmap Top 25 collaboration profile (depuis ZIPs)..."
python sq1_top25_collab_from_leaderboard_zips.py \
    --users yuanzhezhou tascj0 christofhenkel cnumber hydantess jeroencottaar \
            wowfattie jsday96 cpmpml daiwakun takoihiraokazu arc144 conjuring92 \
            aerdem4 tomoon33 mathurinache harshitsheoran dc5e964768ef56302a32 \
            philippsinger chenxin1991 asalhi sayoulala ren4yu brendanartley \
    --competitions_dir out/top10 \
    --leaderboards_dir out/leaderboards \
    --max_competitions_per_user 60 \
    --outdir out/figures_sq1 \
    --workdir out/top10 \
    --highlight yuanzhezhou \
    --min_zips_present 1 \
    --min_row_coverage 0.0

# -----------------------------------------------------------------------------
# 8) Copie des graphiques vers assets/images/ (pour content.md)
# -----------------------------------------------------------------------------
echo "[8/8] Copie des graphiques vers assets/images/..."
mkdir -p ../images
cp out/figures_sq1/*.png ../images/ 2>/dev/null || true

echo ""
echo "=============================================="
echo " Terminé!"
echo "=============================================="
echo ""
echo "Graphiques dans out/figures_sq1/ ET ../images/ :"
ls -la ../images/*.png 2>/dev/null || echo "(aucun fichier PNG trouvé)"
