# DataForGood - Phase 1: Repository Exploration

Ce dossier contient des scripts pour explorer et analyser la structure globale d'un dépôt Git.

## Contenu du dossier

### Scripts

#### `explore_repo.py`

**Objectif**: Scanner le dépôt pour explorer sa structure et identifier les fichiers de données.

**Fonctionnement**:

- Parcourt le dépôt récursivement
- Construit une arborescence complète du projet
- Identifie les fichiers de données basés sur:
  - Extensions de fichier (`.csv`, `.json`, `.parquet`, `.xlsx`, `.sql`, `.ipynb`)
  - Mots-clés dans les chemins (`data`, `dataset`, `raw`, `processed`, `etl`, `ingestion`, etc.)
- Classe les fichiers par extension
- Calcule les ratios de données par rapport au total

**Usage**:

```bash
python explore_repo.py <chemin_vers_repo>
```

**Exemple**:

```bash
python explore_repo.py /path/to/shiftdataportal
```

**Résultats générés dans `results/`**:

1. **`repo_structure.json`**

   - Arborescence complète du projet en format JSON
   - Utile pour visualiser la hiérarchie des fichiers

2. **`file_distribution.csv`**

   - Distribution des fichiers par extension
   - Colonnes: `extension`, `count`

3. **`data_files.csv`**

   - Liste de tous les fichiers de données identifiés
   - Colonnes: `path`, `extension`, `size_bytes`

4. **`repo_profile.json`**

   - Profil global du dépôt
   - Total de fichiers, ratio de données, etc.

5. **`repo_stats_summary.json`**
   - Statistiques résumées:
     - `data_file_ratio`: Pourcentage de fichiers de données
     - `data_raw_ratio`: Ratio de fichiers de données brutes
     - `data_code_ratio`: Ratio de fichiers Python
     - `notebooks_ratio`: Ratio de notebooks Jupyter

## Exemple de sortie

### repo_profile.json

```json
{
  "total_files": 1523,
  "data_files_count": 287,
  "data_file_ratio": 0.189,
  "extensions": {
    ".py": 450,
    ".ts": 320,
    ".csv": 156,
    ".json": 89,
    ".ipynb": 42,
    ...
  }
}
```

### repo_stats_summary.json

```json
{
  "data_file_ratio": 0.189,
  "data_raw_ratio": 0.087,
  "data_code_ratio": 0.295,
  "data_formats_count": 4,
  "notebooks_ratio": 0.028
}
```

## 🔧 Éléments détectés

**Extensions de données**:

- Données brutes: `.csv`, `.json`, `.xlsx`, `.parquet`, `.sql`
- Notebooks: `.ipynb`

**Mots-clés recherchés**:

- `data`, `dataset`, `datasets`, `raw`, `processed`, `scraping`, `etl`, `ingestion`

## 🚀 Exécution rapide

```bash
python explore_repo.py /path/to/repo
# Résultats générés dans results/
```

## 📁 Exemple de fichier data_files.csv

```
path,extension,size_bytes
data/bmo/bmo_2024.csv,.csv,2048576
data/rpls/logements_sociaux.xlsx,.xlsx,5242880
data-preparation/utils/config.py,.py,4096
notebooks/analysis.ipynb,.ipynb,1048576
```

---

**Créé en January 2026 | Phase 1 - Repository Exploration**
