# DataForGood - Phase 2: Data Access Analysis

Ce dossier contient des scripts pour analyser l'accès aux données et l'historique d'introduction des sources de données dans un dépôt Git.

## Contenu du dossier

### Scripts

#### 1. `build_access_inventory.py`

**Objectif**: Classifier les fichiers de données en fonction de leur source et mode d'accès.

**Fonctionnement**:

- Prend en entrée le fichier `data_files.csv` généré par Phase 1
- Classe chaque fichier selon:
  - **Source**: D'où viennent les données (dossier data/, data-preparation/, server/data/, etc.)
  - **Mode d'accès**: Comment les données sont utilisées (raw_data, pipeline, notebook, embedded_db, etc.)
- Génère un inventaire complet des accès

**Usage**:

```bash
python build_access_inventory.py <path_to_data_files.csv> [results_dir]
```

**Exemple**:

```bash
python build_access_inventory.py results/data_files.csv
```

**Résultats**:

- **`access_inventory_shiftdataportal.csv`**
  - Colonnes: `path`, `extension`, `size_bytes`, `source`, `access_mode`
  - Affiche aussi un résumé des sources et modes d'accès trouvés

#### 2. `access_churn.py`

**Objectif**: Analyser l'évolution mensuelle de l'accès aux données par catégorie.

**Fonctionnement**:

- Parcourt l'historique Git complet
- Groupe les commits par mois
- Identifie quelles catégories de fichiers ont été modifiées chaque mois
- Génère un graphique d'activité par catégorie

**Catégories suivies**:

- `raw_data_files`: Fichiers de données brutes (CSV, XLSX, JSON, etc.)
- `data_preparation_scripts`: Scripts de préparation (Python, Jupyter, Shell)
- `backend_api`: Code backend/API (TypeScript, JavaScript, GraphQL)
- `frontend_client`: Code frontend (React, TypeScript, CSS)
- `infrastructure_deployment`: Configuration Docker, CI/CD, YAML
- `documentation`: Fichiers README et documentation Markdown

**Usage**:

```bash
python access_churn.py <chemin_vers_repo>
```

**Exemple**:

```bash
python access_churn.py /path/to/shiftdataportal
```

**Résultats**:

- **`access_churn_shiftdataportal.csv`**
  - Colonnes: `month` + chaque catégorie
  - Montre combien de commits ont touché chaque catégorie par mois
  - Exemple:
    ```
    month,raw_data_files,data_preparation_scripts,backend_api,...
    2024-01,5,12,8,...
    2024-02,3,15,10,...
    ```

#### 3. `first_seen.py`

**Objectif**: Identifier quand chaque source de données et mode d'accès a été introduit pour la première fois.

**Fonctionnement**:

- Parcourt l'historique Git complet
- Charge l'inventaire d'accès (`access_inventory.csv`)
- Trouve le premier commit contenant chaque source/mode d'accès
- Génère deux fichiers CSV avec les dates d'introduction

**Usage**:

```bash
python first_seen.py <repo_path> <access_inventory.csv> <results_dir>
```

**Exemple**:

```bash
python first_seen.py /path/to/shiftdataportal results/access_inventory_shiftdataportal.csv results/
```

**Résultats**:

- **`first_seen_sources.csv`**

  - Colonnes: `source`, `first_commit_hash`, `first_commit_date`, `first_commit_month`
  - Quand chaque source de données a été introduite

- **`first_seen_modes.csv`**
  - Colonnes: `access_mode`, `first_commit_hash`, `first_commit_date`, `first_commit_month`
  - Quand chaque mode d'accès a été introduit

#### 4. `plot_first_seen.py`

**Objectif**: Générer 3 visualisations complémentaires de la chronologie d'introduction des sources et modes d'accès.

**Fonctionnement**:

- Charge les données de `first_seen_sources.csv` et `first_seen_modes.csv`
- Génère automatiquement 3 graphiques différents:

**Visualisation 1 - Timeline Gantt**

- Vue d'ensemble style Gantt avec groupement par section
- Montre toutes les sources et modes d'accès sur une timeline horizontale
- Lignes reliant chaque élément à sa date d'introduction
- Idéal pour voir l'ordre et la densité des introductions

**Visualisation 2 - Timeline Séparée**

- Deux graphiques superposés (sources et modes d'accès)
- Chaque ligne représente un élément avec sa date d'apparition
- Permet de comparer indépendamment l'évolution des sources vs modes
- Vue plus lisible avec moins d'éléments par graphique

**Visualisation 3 - Histogramme mensuel**

- Distribution du nombre de nouvelles sources/modes par mois
- Histogrammes séparés pour sources et modes d'accès
- Permet d'identifier les mois de forte activité
- Vue agrégée et synthétique

**Usage**:

```bash
python plot_first_seen.py
```

**Prérequis**:

- Fichiers `results/first_seen_sources.csv` et `results/first_seen_modes.csv`

**Résultats**:

- **`first_seen_timeline_gantt.png`** - Vue Gantt complète
- **`first_seen_timeline_separate.png`** - Deux graphiques séparés
- **`first_seen_timeline_histogram.png`** - Distribution mensuelle

#### 5. `visual_charn.py`

**Objectif**: Visualiser le churn d'accès mensuel en graphique en aires empilées.

**Fonctionnement**:

- Charge le fichier `access_churn_shiftdataportal.csv`
- Crée un graphique en aires empilées montrant:
  - Évolution mensuelle de l'activité par catégorie
  - Répartition de l'effort de développement
  - 6 couleurs distinctes pour les catégories

**Catégories affichées**:

- Données brutes
- Scripts de préparation
- API Backend (GraphQL)
- Frontend (React)
- Infra / CI-CD
- Documentation

**Usage**:

```bash
python visual_charn.py
```

**Prérequis**:

- Fichier `results/access_churn_shiftdataportal.csv`

**Résultats**:

- **`shiftdataportal_churn_stacked_area.png`** - Graphique en aires empilées

#### 6. `visual_churn_dual.py`

**Objectif**: Créer une double visualisation du churn d'accès (linéaire + empilée).

**Fonctionnement**:

- Charge le fichier `access_churn_shiftdataportal.csv`
- Crée deux graphiques complémentaires:
  1. **Graphique linéaire**: Évolution mensuelle avec une ligne par catégorie
     - Permet de voir les tendances individuelles
     - Points de marquage pour chaque valeur
  2. **Graphique empilé**: Aires empilées par catégorie
     - Montre la proportion relative de chaque catégorie
     - Vue d'ensemble de la répartition de l'effort

**Usage**:

```bash
python visual_churn_dual.py
```

**Prérequis**:

- Fichier `results/access_churn_shiftdataportal.csv`

**Résultats**:

- **`shiftdataportal_churn_analysis.png`** - Double graphique (lignes + areas)

## Workflow complet

### Étape 1: Construire l'inventaire d'accès

```bash
# Prérequis: vous devez avoir exécuté Phase 1 d'abord
python build_access_inventory.py ../1/results/data_files.csv
```

### Étape 2: Analyser le churn d'accès

```bash
python access_churn.py /path/to/shiftdataportal
```

### Étape 3: Identifier les premières apparitions

```bash
python first_seen.py /path/to/shiftdataportal results/access_inventory_shiftdataportal.csv results/
```

### Étape 4: Visualiser la timeline (optionnel)

```bash
python plot_first_seen.py
# Génère automatiquement 3 graphiques PNG
```

### Étape 5: Visualiser le churn mensuel (optionnel)

```bash
# Graphique simple en aires empilées
python visual_charn.py

# Double graphique (lignes + areas)
python visual_churn_dual.py
```

## Sortie attendue

### Dossier `results/`

```
results/
├── access_inventory_shiftdataportal.csv
├── access_churn_shiftdataportal.csv
├── first_seen_sources.csv
├── first_seen_modes.csv
├── first_seen_timeline_gantt.png
├── first_seen_timeline_separate.png
├── first_seen_timeline_histogram.png
├── shiftdataportal_churn_stacked_area.png (optionnel)
└── shiftdataportal_churn_analysis.png (optionnel)
```

## Classification des sources

Les sources identifiées incluent:

- `data_root`: Fichiers dans `data/`
- `data_bmo`: Données BMO (Business Maintained Objects)
- `data_rpls`: Données RPLS (logements sociaux)
- `data_preparation`: Scripts/données de préparation
- `server_embedded_db`: Base de données embarquée du serveur
- `other`: Autres sources

## Classification des modes d'accès

- `raw_data`: Données brutes
- `data_pipeline`: Pipeline de données
- `data_transformation`: Scripts de transformation
- `notebook`: Jupyter notebooks
- `embedded_db`: Base de données embarquée
- `api_backend`: API backend
- `frontend`: Code frontend
- `config`: Fichiers de configuration
- `shell_script`: Scripts shell
- `utility`: Fichiers utilitaires
- `other`: Autres

## Dépendances

```
pydriller
pandas
seaborn
matplotlib
```

Installation:

```bash
pip install pydriller pandas seaborn matplotlib
```

---

**Créé en January 2026 | Phase 2 - Data Access Analysis**
