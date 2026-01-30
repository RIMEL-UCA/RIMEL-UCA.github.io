# Suivi de l'Évolution des Sources de Données

Cet ensemble d'outils analyse l'historique Git d'un dépôt pour suivre l'évolution du nombre de **sources de données externes** (fichiers CSV, URL, API) référencées dans votre fichier de configuration au fil du temps.

Il se compose de deux scripts principaux :
1.  **Extraction :** Scanne l'historique du dépôt et extrait chaque référence trouvée dans `back/config.yaml`.
2.  **Visualisation :** Reconstruit l'état du projet au fil du temps et trace le nombre total de sources de données présentes.

## Utilisation

### 1. Extraire les Données Historiques

Exécutez le script d'extraction en pointant vers votre dépôt local. Cela scanne chaque commit où `back/config.yaml` a été modifié.

```bash
# Syntaxe : python count_data_sources.py <chemin_vers_le_repo>
python count_data_sources.py ../chemin/vers/votre/repo/

```

* **Entrée :** Localise automatiquement `back/config.yaml` à l'intérieur du dépôt cible.
* **Sortie :** Génère un fichier `results/datasource_references_history.csv`.

### 2. Visualiser l'Évolution

Exécutez le script de visualisation pour générer le graphique.

```bash
python vizualize_data_source_history.py

```

* **Entrée :** Lit le fichier CSV généré à l'étape 1.
* **Logique :** Utilise une approche "Machine à États". Il rejoue l'historique commit par commit pour calculer exactement combien de sources de données existaient dans le fichier à une date donnée (en filtrant pour l'activité à partir du 1er Janvier 2025).
* **Sortie :** Génère un graphique `results/datasource_state_evolution.png`.

## Exemple de Résultat

L'outil génère un **Graphique en Escaliers** (Step Chart) qui montre le nombre exact de sources de données présentes dans le code au fil du temps.

* **Lignes plates :** Périodes où aucune source de données n'a été ajoutée ou supprimée.
* **Marches (montée/descente) :** Jours spécifiques où un commit a ajouté ou supprimé une source.