# Suivi de l'Historique des Modules d'Enrichissement

Ce kit analyse l'historique Git d'un dépôt pour suivre l'évolution des modules d'enrichissement (scripts Python situés dans `back/scripts/enrichment`). Il permet de visualiser comment le nombre de scripts d'enrichissement actifs a augmenté au fil du temps.

## Vue d'ensemble

Le kit se compose de deux scripts Python :

1.  **`track_enrichment_history.py`** :
    * Scanne l'historique Git du dépôt.
    * Suit chaque ajout (`ADD`) et suppression (`DELETE`) de fichiers dans `back/scripts/enrichment`.
    * Ignore automatiquement les fichiers utilitaires (`utils`) et les classes de base (`base_enricher.py`).
    * Génère un journal CSV détaillé de tous les changements.

2.  **`plot_enrichment_history.py`** :
    * Lit le journal CSV.
    * Rejoue l'historique pour calculer le **nombre total** de fichiers d'enrichissement à une date donnée.
    * Génère un graphique temporel montrant la croissance de votre logique d'enrichissement.

## Prérequis

Assurez-vous d'avoir Python 3 installé. Installez les dépendances requises :

```bash
pip install -r requirements.txt
```

## Utilisation

### Étape 1 : Extraire l'Historique

Exécutez le script de suivi en pointant vers votre dépôt local.

```bash
# Syntaxe : python track_enrichment_history.py <chemin_vers_le_repo>
python track_enrichment_history.py ../chemin/vers/votre/repo/

```

* **Entrée :** Scanne `back/scripts/enrichment` dans le dépôt cible.
* **Sortie :** Génère le fichier `results/enrichment_history.csv`.

### Étape 2 : Visualiser l'Évolution

Exécutez le script de tracé pour générer le graphique.

```bash
python plot_enrichment_history.py
```

* **Entrée :** Lit le fichier `results/enrichment_history.csv`.
* **Sortie :** Génère le graphique `results/enrichment_count_evolution.png`.
