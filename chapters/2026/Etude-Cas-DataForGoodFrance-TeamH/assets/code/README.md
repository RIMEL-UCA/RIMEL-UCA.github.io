# Analyse de Repositories

Ce projet permet d'auditer la structure de plusieurs dépôts de code (Code vs Data, Notebooks, etc.) et de générer un tableau de bord comparatif visuel.

## Installation

Toutes les dépendances nécessaires à l'exécution des scripts sont listées dans `requirements.txt`.

1. Installez les librairies Python :
    ```bash
    pip install -r requirements.txt
    ```

2. Créez un fichier `.env` à la racine du projet pour votre token github nécessaire à certains scripts :
    ```bash
    GITHUB_TOKEN=votre_token_github_ici
    ```

## Structure du Projet

L'architecture est organisée pour séparer les outils communs des résultats d'analyse spécifiques :

```text
.
├── common/                     # Scripts et étapes communes à tous les projets
├── repositories_analysis/      # Dossier contenant l'analyse de chaque repo (organisé par step)
│   ├── [NOM_DU_REPO]/
│   │   └── results/            # Fichiers JSON/CSV générés par l'exploration
├── recap-repo.py               # Script principal de génération du dashboard
├── requirements.txt            # Liste des dépendances
├── .env                        # Configuration (Token API)
└── multi_repo_dashboard.png    # (Output) Le tableau de bord final généré

```
