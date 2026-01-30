# Script d’exploration du repository

## Objectif

Ce script produit une vue statique automatisée d’un repository pour la phase 1 du protocole RIMEL : exploration du projet.
Il permet d’identifier la structure du projet, la répartition des fichiers et la place occupée par les éléments liés à la donnée.


## Fonctionnement général

Le script parcourt récursivement l’ensemble du repository à partir d’un chemin racine fourni en entrée.
Il classe les fichiers par extension, détecte les fichiers liés à la donnée à l’aide de règles simples, et génère plusieurs fichiers de sortie exploitables pour l’analyse, stockés dans un répertoire /results.

## Hypothèses et règles de détection

### Définition d’un “fichier data”

Un fichier est considéré comme lié à la donnée s’il vérifie au moins un des critères suivants :

#### 1. Extension “data brute”

Les extensions suivantes sont considérées comme représentant des données ou des artefacts analytiques :

* `.csv`, `.json`, `.parquet`, `.xlsx`
* `.sql`
* `.ipynb`

#### 2. Mots-clés dans le chemin

Un fichier est également classé comme “data” si son chemin contient l’un des mots-clés suivants :

* `data`, `dataset(s)`
* `raw`, `processed`
* `scraping`, `etl`, `ingestion`

Cette règle permet d’inclure les scripts de collecte, de transformation ou d’analyse, même lorsqu’ils sont écrits en Python ou dans d’autres langages.

## Outputs générés

### `repo_structure.json`

Représentation arborescente complète du repository, utilisé pour analyser la structure globale et repérer les zones clés du projet.

### `file_distribution.csv`

Répartition du nombre de fichiers par extension.
Le fichier permet d’évaluer la nature dominante du projet (code, data, frontend, documentation).

### `data_files.csv`

Liste des fichiers détectés comme liés à la donnée, avec :

* chemin relatif
* extension
* taille du fichier

Ce fichier sert de base pour les analyses des phases 2 et 3, en répertoriant une partie des fichiers qui touchent aux données.

### `repo_profile.json`

Synthèse des indicateurs clés :

* nombre total de fichiers
* nombre et ratio de fichiers data
* distribution des extensions
* présence de dossiers structurants (`data`, `backend`, `frontend`, `scripts`, `notebooks`)

Le `data_file_ratio` correspond à la proportion de fichiers liés à la donnée par rapport au nombre total de fichiers du repository.

### `repo_stats_summary.json`

Le fichier repo_stats_summary.json contient un ensemble d’indicateurs structurels synthétiques, calculés automatiquement afin de permettre la comparaison inter-repositories lors de la phase d’exploration.

Chaque indicateur est normalisé par le nombre total de fichiers du repository.

Contenu du fichier :
`{
  "data_file_ratio": 0.335,
  "data_raw_ratio": 0.058,
  "data_code_ratio": 0.382,
  "data_formats_count": 6,
  "structuring_dirs_count": 5,
  "notebooks_ratio": 0.033
}`

### Définition des indicateurs

`data_file_ratio` 

Proportion de fichiers liés à la donnée au sens large.

Calcul :
(fichiers data bruts + scripts data + notebooks) / total des fichiers

`data_raw_ratio`

Proportion de fichiers de données brutes.

Extensions prises en compte :
.csv, .json, .xlsx, .parquet, .sql

`data_code_ratio`

Proportion de code orienté data.

Hypothèse : tous les fichiers .py sont considérés comme du code data.

`data_formats_count`

Nombre de formats de données distincts présents dans le repository.

Formats comptés : .csv, .json, .xlsx, .parquet, .sql, .ipynb


`structuring_dirs_count`

Nombre de dossiers structurants détectés parmi : data, backend, frontend, scripts, notebooks

`notebooks_ratio`

Proportion de notebooks (.ipynb) dans le repository.
Indicateur du caractère exploratoire du projet.


## Limites

* Analyse purement statique pour cette phase (pas de dimension temporelle).
* Détection basée sur des heuristiques simples.
* Les résultats visent une comparabilité inter-repositories, non une vérité sémantique exhaustive.