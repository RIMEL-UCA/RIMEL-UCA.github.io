# Phase 4 : Analyse Quantitative des Contributions

Scripts pour analyser l'historique git, inférer les rôles des contributeurs (Backend, DevOps, etc.) et visualiser les phases du projet dans le temps.

## Installation

1. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt

   ```
Rappel, n'oubliez pas de mettre votre token github en variable d'environement (.env a la racine du repo)

## Pipeline d'Utilisation

Exécutez les scripts dans cet ordre précis.

### 1. Extraction des Données

Extrait l'historique des commits et les métadonnées des profils.

```bash
python phase4_contributions.py <chemin_vers_repo_git_local>
```

* **Sortie :** `results/commits_detailed.csv`, `results/contributors_profiles.csv`

### 2. Inférence des Profils

Déduit le rôle technique (ex: Data Scientist, Frontend) selon les fichiers modifiés.

```bash
python phase4_contributor_inferrence.py

```

* **Sortie :** `results/contributors_jobs_inferred.csv`

### 3. Visualisation : Chronologie des Contributeurs

Génère un graphique de flux montrant qui intervient quand, avec son rôle.

```bash
# Avec les jobs inférés 
python phase4_contributions_vizualisation.py results/contributors_jobs_inferred.csv

# OU avec un fichier manuel
python phase4_contributions_vizualisation.py results/contributors_jobs.csv

```

A noté que `results/contributors_jobs.csv` doit être remplit manuellement avec le vrai métier des contributeurs (via linkedIn, portfolio par exemple)

Pour plus de détails sur l'inférence des profils, voir [utils](/utils/)

* **Sortie :** `results/contributions_timeline_jobs.png`

### 4. Visualisation : Phases du Projet

Génère un diagramme en barres montrant l'activité dominante par mois.

```bash
# Avec les jobs inférés ou manuels
python phase4_vizualisation_project_state.py <file>

```

* **Sortie :** `results/project_steps_jobs_bar.png`

## Structure de Dossier Requise

Votre dossier doit respecter cette structure pour les imports :

```text
.
├── phase4_contributions.py
├── phase4_contributor_inferrence.py
├── phase4_contributions_vizualisation.py
├── phase4_vizualisation_project_state.py
├── results/       # Créé automatiquement

```
