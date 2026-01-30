Ce dossier contient un ensemble d'outils Python conçus pour explorer la structure de dépôts Git, identifier les actifs liés aux données et visualiser des métriques comparatives entre différents projets.

---

### **Fichiers**

* **`explore_repo.py`** : Un script qui scanne un répertoire local pour analyser les extensions de fichiers, identifier les fichiers de données (basé sur des extensions comme `.csv`, `.parquet` ou des mots-clés comme "etl", "scraping") et calculer des ratios structurels.
* **`compare_all_repos.py`** : Un script de visualisation qui génère un graphique "Radar" (diagramme en araignée) comparant les profils de structure de données de plusieurs projets en utilisant des métriques normalisées.

---

### **Utilisation**

#### **1. Analyser un dépôt**

Exécutez le script d'exploration sur un chemin de dépôt local spécifique. Cela créera un dossier `results` contenant des statistiques détaillées.

```bash
python explore_repo.py <chemin_vers_le_repo_local>

```

#### **2. Comparer les profils**

Exécutez le script de comparaison pour générer la visualisation radar.
*Note : Les données des projets dans ce script sont actuellement codées en dur.*

```bash
python compare_all_repos.py
```

---

### **Résultats (Outputs)**

Après avoir exécuté `explore_repo.py`, les fichiers suivants sont générés dans le dossier `results/` :

* **`repo_stats_summary.json`** : Métriques de haut niveau (ex: `data_raw_ratio`, `notebooks_ratio`) utilisées pour le profil de comparaison.
* **`repo_structure.json`** : Une représentation complète de l'arborescence du dépôt.
* **`file_distribution.csv`** : Le décompte de chaque extension de fichier trouvée.
* **`data_files.csv`** : Une liste des fichiers de données spécifiques trouvés, incluant leurs chemins et tailles.
* **`repo_profile.json`** : Un résumé incluant le nombre total de fichiers et les dictionnaires d'extensions spécifiques.
