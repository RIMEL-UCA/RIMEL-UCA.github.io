# Phase 1

## Contexte et généralités

Le projet "Dans Mon Eau" est un outil cartographique interactif développé par Générations Futures et Data For Good.
Il permet aux utilisateurs de consulter la qualité de l'eau potable en France par adresse, en visualisant les concentrations de 5 catégories de polluants chimiques : pesticides, PFAS, nitrates, CVM, perchlorate.

Pas de fichiers CSV/JSON volumineux dans le repo (données téléchargées dynamiquement).
Pas de notebooks Jupyter utilisés (dossier analytics mentionnés comme "non utilisés depuis avril 2025", les notebooks de la pipeline ne servent qu'à aider à développer les pipelines python).
Pipeline CI/CD pour ingestion mensuelle des nouvelles données.


## Structure du projet

- `pipelines/` : Consolidation et préparation des données
- `dbt_/` : Modèles de données et transformations
- `analytics/` : Analyses et notebooks (n'est plus utilisé depuis la fin de la saison)
- `webapp/` : Site web Next.js interactif
- `database/` : Base de données DuckDB et fichiers de cache


## Flux de données

1. Récupération et traitement
    - Pipelines Python pour ingérer les données brutes (CSV/JSON de sources ouvertes), les nettoyer et les stocker dans DuckDB

2. Transformation
    - Modèles DBT (SQL) organisés en couches (staging, intermediate, website) pour agréger et optimiser les données (bilans annuels, seuils de pollution).

3. Diffusion
    - Génération de PMTiles pour la cartographie interactive, et base DuckDB réduite pour les stats du site web.
    - Liaison entre la base DuckDB et s3 afin 