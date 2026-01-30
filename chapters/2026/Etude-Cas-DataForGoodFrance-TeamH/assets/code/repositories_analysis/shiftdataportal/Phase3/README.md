# DataForGood - Phase 3: Data Preparation History Analysis

Ce dossier contient des scripts d'analyse de l'historique des scripts de préparation de données pour le projet **Shift Data Portal**.

## Contenu du dossier

### Scripts

#### 1. `track_data_preparation_history.py`

**Objectif**: Extraire et analyser l'historique des modifications apportées aux scripts de préparation de données.

**Fonctionnement**:

- Parcourt l'historique complet des commits du dépôt Git
- Identifie toutes les modifications dans le dossier `data-preparation/`
- Filtre les fichiers pertinents (`.py`, `.ipynb`, `.sh`) en excluant les fichiers de configuration et de test
- Extrait pour chaque modification:
  - Date du commit
  - Hash du commit (court format)
  - Auteur
  - Chemin du fichier
  - Type de changement (ADD, MODIFY, DELETE, RENAME)
  - Nombre de lignes ajoutées/supprimées
  - Message du commit

**Usage**:

```bash
python track_data_preparation_history.py <chemin_vers_shiftdataportal>
```

**Exemple**:

```bash
python track_data_preparation_history.py /path/to/shiftdataportal
```

**Résultats**:

- Génère un fichier CSV: `results/data_preparation_history.csv`
- Affiche un résumé dans la console avec:
  - Nombre total de changements
  - Nombre de fichiers uniques modifiés
  - Nombre de contributeurs
  - Plage de dates couverte

#### 2. `plot.py`

**Objectif**: Générer des visualisations et statistiques détaillées basées sur l'historique extrait.

**Visualisations générées**:

1. **Weekly Activity Plot** (`data_preparation_activity_weekly.png`)

   - Graphique d'activité hebdomadaire par type de changement (ADD, MODIFY, DELETE, RENAME)
   - Graphique du volume total d'activité avec courbe de tendance
   - Montre l'intensité du développement au fil du temps

2. **Code Churn Plot** (`data_preparation_code_churn.png`)

   - Lignes ajoutées vs supprimées (hebdomadaire)
   - Code churn total (somme des ajouts et suppressions)
   - Indicateur de l'activité de refactorisation

3. **Top Files Plot** (`data_preparation_top_files.png`)

   - Top 15 des fichiers les plus modifiés
   - Classement par nombre de modifications

4. **Statistiques détaillées** (affichées dans la console)
   - Répartition par type de changement
   - Total de lignes de code ajoutées/supprimées
   - Top 5 des contributeurs
   - Nombre de fichiers touchés

**Usage**:

```bash
python plot.py
```

**Prérequis**:

- Le fichier `results/data_preparation_history.csv` doit exister (généré par `track_data_preparation_history.py`)

**Résultats**:

- Génère 3 fichiers PNG dans le dossier `results/`
- Affiche les statistiques dans la console

## Workflow complet

### Étape 1: Extraire l'historique

```bash
python track_data_preparation_history.py /path/to/shiftdataportal
```

### Étape 2: Générer les visualisations

```bash
python plot.py
```

## Sortie attendue

### Dossier `results/`

```
results/
├── data_preparation_history.csv          # Données brutes
├── data_preparation_activity_weekly.png  # Activité par semaine
├── data_preparation_code_churn.png       # Churn de code
└── data_preparation_top_files.png        # Top 15 fichiers
```

### Fichier CSV

Le fichier `data_preparation_history.csv` contient les colonnes suivantes:

- **date**: Date du commit (ISO format: YYYY-MM-DD)
- **commit**: Hash court du commit (8 caractères)
- **author**: Nom de l'auteur du commit
- **file**: Chemin du fichier modifié
- **change_type**: Type de modification (ADD, MODIFY, DELETE, RENAME)
- **lines_added**: Nombre de lignes ajoutées
- **lines_removed**: Nombre de lignes supprimées
- **msg**: Premier ligne du message de commit (max 100 caractères)

## Cas d'usage

- **Analyse de la qualité du projet**: Identifier les scripts critiques (plus modifiés)
- **Suivi de la productivité**: Visualiser les pics d'activité
- **Gestion du code**: Déterminer quels fichiers ont besoin de refactorisation
- **Évaluation des contributeurs**: Identifier les contributeurs clés
- **Tendances du développement**: Comprendre l'évolution du projet

## Configuration

### Paramètres dans `track_data_preparation_history.py`

```python
DATA_PREPARATION_DIR = "data-preparation"  # Dossier cible
EXTENSIONS = {".py", ".ipynb", ".sh"}      # Extensions à tracker
IGNORE_EXACT_NAMES = {...}                 # Fichiers à ignorer
IGNORE_KEYWORDS = [...]                    # Mots-clés d'exclusion
```

### Paramètres dans `plot.py`

```python
START_DATE = "2024-01-01"  # Date de début de l'analyse
```

Modifiez ces paramètres pour adapter l'analyse à vos besoins.

## Dépendances

- `pydriller`: Extraction de l'historique Git
- `pandas`: Manipulation de données
- `matplotlib`: Génération de graphiques
- `numpy`: Calculs numériques

Installation:

```bash
pip install pydriller pandas matplotlib numpy
```

## Notes importantes

- Les scripts ignorent automatiquement les fichiers de configuration (`__init__.py`, `requirements.txt`, etc.)
- Les chemins avec "utils" ou "test" sont exclus de l'analyse
- L'analyse se limite au dossier `data-preparation/`
- Les dates sont normalisées au format ISO (YYYY-MM-DD)
- Le dossier `results/` est créé automatiquement s'il n'existe pas

## Exécution rapide

```bash
# Récupérer l'historique
python track_data_preparation_history.py /path/to/repo

# Générer les graphiques
python plot.py

# Visualiser les résultats
# Ouvrir les fichiers PNG dans results/
```

---

**Créé en January 2026 | Phase 3 - Data Preparation Analysis**
