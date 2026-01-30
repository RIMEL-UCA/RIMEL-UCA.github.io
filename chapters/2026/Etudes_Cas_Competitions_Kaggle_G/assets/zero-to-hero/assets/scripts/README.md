# Guide de Reproductibilité - SQ1 : Zero to Hero

Ce guide explique comment reproduire l'analyse de la sous-question 1 : "Comment le numéro 1 du leaderboard global est-il passé de 0 à héros ?"

---

## Pré-requis

### 1. Environnement Python

- **Python 3.10+** requis
- Système d'exploitation : Windows, macOS ou Linux

### 2. Compte Kaggle

Vous devez avoir un **compte Kaggle actif** pour :
- Vous authentifier sur le site (scraping des profils et leaderboards)
- Télécharger les leaderboards en fichiers ZIP

---

## Installation

### Étape 1 : Cloner le dépôt et accéder au dossier

```bash
cd zero-to-hero/assets/scripts
```

### Étape 2 : Installer les dépendances Python

```bash
pip install -r requirements.txt
```

### Étape 3 : Installer Playwright et Chromium

Playwright est utilisé pour le scraping automatisé des pages Kaggle.

```bash
playwright install chromium
```

---

## Configuration de l'authentification Kaggle

### Pourquoi c'est nécessaire ?

Les scripts de scraping ont besoin d'accéder à des pages Kaggle qui requièrent une authentification :
- Profils utilisateurs (onglet Competitions)
- Leaderboards des compétitions
- Rangs globaux des coéquipiers

### Comment générer `kaggle_state.json` ?

Le fichier `kaggle_state.json` stocke votre session Playwright authentifiée sur Kaggle.

#### Option A : Script automatique (recommandé)

```bash
python 02_login_save_state.py
```

Ce script :
1. Ouvre un navigateur Chromium
2. Vous redirige vers la page de connexion Kaggle
3. **Vous devez vous connecter manuellement** (email/password ou Google/GitHub)
4. Une fois connecté, le script sauvegarde la session dans `kaggle_state.json`

> **Note** : Le navigateur reste ouvert ~30 secondes pour vous laisser le temps de vous connecter.

#### Option B : Connexion manuelle

Si le script ne fonctionne pas :
1. Lancez `python 02_login_save_state.py --no-headless` pour voir le navigateur
2. Connectez-vous manuellement sur Kaggle
3. Attendez que le script détecte la connexion et sauvegarde l'état

### Vérification

Après l'exécution, vous devriez avoir un fichier `kaggle_state.json` dans le dossier courant.

```bash
ls -la kaggle_state.json
```

> **Sécurité** : Ce fichier contient vos cookies de session Kaggle. **Ne le partagez jamais** et ajoutez-le à `.gitignore`.

---

## Téléchargement des Leaderboards (fichiers ZIP)

Certaines analyses (déciles solo/équipe, profil de collaboration du top 25) nécessitent les **leaderboards téléchargés manuellement** depuis Kaggle.

### Pourquoi ?

Les leaderboards contiennent des informations détaillées (composition des équipes, noms des membres) qui ne sont pas accessibles via scraping simple.

### Comment télécharger un leaderboard ?

1. Allez sur une page de compétition Kaggle, ex: https://www.kaggle.com/competitions/optiver-trading-at-the-close/leaderboard
2. Cliquez sur le bouton **"Download"** (icône de téléchargement)
3. Sauvegardez le fichier ZIP dans `out/leaderboards/`

### Structure attendue

```
out/
└── leaderboards/
    ├── optiver-trading-at-the-close.zip
    ├── jane-street-market-prediction.zip
    ├── asl-signs.zip
    └── ... (autres compétitions)
```

> **Important** : Le nom du fichier ZIP doit correspondre au **slug** de la compétition (partie de l'URL après `/competitions/`).

### Compétitions recommandées

Pour reproduire l'analyse complète, téléchargez au moins 5-10 leaderboards de compétitions récentes et populaires.

---

## Exécution de l'analyse

### Option 1 : Script complet (recommandé)

```bash
chmod +x reproduce.sh  # Linux/macOS uniquement
./reproduce.sh
```

Sur Windows :
```bash
bash reproduce.sh
```

Ce script exécute toutes les étapes dans l'ordre :
1. Scraping de la timeline de yuanzhezhou
2. Analyse solo vs équipe
3. Heatmap des coéquipiers
4. Force d'équipe vs performance
5. Analyse par décile (depuis les ZIPs)
6. Comparaison des "fast risers"
7. Profil de collaboration du top 25

### Option 2 : Exécution étape par étape

Si vous voulez exécuter les scripts individuellement :

```bash
# 1. Timeline yuanzhezhou
python sq1_scrape_timeline_user.py --username yuanzhezhou --state kaggle_state.json --headless

# 2. Solo vs équipe
python sq1_solo_vs_team_timeline.py --csv out/user_competitions_raw.csv --state kaggle_state.json --headless

# 3. Heatmap coéquipiers
python sq1_teammates_heatmap.py --csv out/user_competitions_raw.csv --state kaggle_state.json --headless

# 4. Force équipe vs performance
python sq1_team_strength_vs_performance.py --state kaggle_state.json

# 5. Déciles (depuis ZIPs locaux)
python sq1_solo_vs_team_deciles_from_csv.py --leaderboards_dir out/leaderboards

# 6. Fast risers
python sq1_compare_fast_risers_collab_heatmap.py --users yuanzhezhou jsday96 daiwakun --state kaggle_state.json --headless

# 7. Top 25
python sq1_top25_collab_from_leaderboard_zips.py --users yuanzhezhou tascj0 christofhenkel --competitions_dir out/top10 --leaderboards_dir out/leaderboards
```

---

## Fichiers générés

Après exécution, les graphiques sont générés dans `out/figures_sq1/` :

| Fichier | Description |
|---------|-------------|
| `yuanzhezhou_competitions_per_year.png` | Nombre de compétitions par an |
| `yuanzhezhou_performance_score_per_year.png` | Score de performance par an |
| `solo_vs_team_counts.png` | Évolution solo vs équipe (barplot) |
| `solo_vs_team_ratios.png` | Proportion solo vs équipe (line plot) |
| `teammates_heatmap_topN.png` | Heatmap des 20 principaux coéquipiers |
| `team_strength_vs_perf_scatter.png` | Scatter plot force équipe vs performance |
| `team_strength_vs_perf_boxplot.png` | Boxplot force équipe vs performance |
| `solo_vs_team_by_decile.png` | Répartition solo/équipe par décile |
| `fast_risers_collab_index_heatmap.png` | Collaboration index des fast risers |
| `top25_collab_metrics_heatmap_from_zips_filtered.png` | Profil de collaboration du top 25 |

---

## Dépannage

### Erreur : "Not authenticated"

→ Votre `kaggle_state.json` est expiré ou invalide. Relancez `python 02_login_save_state.py`.

### Erreur : "Timeout on profile page"

→ Kaggle peut être lent. Augmentez le timeout ou réessayez plus tard.

### Erreur : "No ZIP files found"

→ Téléchargez des leaderboards dans `out/leaderboards/`.

### Les graphiques sont vides ou incorrects

→ Vérifiez que le scraping a bien fonctionné en inspectant les fichiers CSV dans `out/`.

---

## Structure des fichiers

```
zero-to-hero/assets/scripts/
├── README.md                           # Ce guide
├── requirements.txt                    # Dépendances Python
├── reproduce.sh                        # Script de reproduction complet
├── 02_login_save_state.py              # Authentification Kaggle
├── sq1_scrape_timeline_user.py         # Timeline compétitions
├── sq1_solo_vs_team_timeline.py        # Analyse solo/équipe
├── sq1_teammates_heatmap.py            # Heatmap coéquipiers
├── sq1_team_strength_vs_performance.py # Force équipe vs perf
├── sq1_solo_vs_team_deciles_from_csv.py# Déciles depuis ZIPs
├── sq1_compare_fast_risers_collab_heatmap.py # Fast risers
└── sq1_top25_collab_from_leaderboard_zips.py # Top 25
```

---

## Contact

Pour toute question sur la reproductibilité :
- **Auteur** : Sacha Chantoiseau
- **Email** : sacha.chantoiseau@etu.univ-cotedazur.fr
