## Script `build_access_inventory.py`  

Commande pour le lancer : 

`python build_access_inventory.py ../../results/data_files.csv results`

Ce script génère un inventaire des accès aux données à partir du fichier `data_files.csv` produit en phase 1.
Chaque fichier lié à la donnée est automatiquement classé selon **sa source** (ex. Allociné, CNC, Mubi, prédictions ML, initialisation DB, etc.) et selon son **mode d’accès** (fichier brut, scraping, extraction, seed de base de données, migration, notebook).

Le script produit un fichier `access_inventory.csv` contenant, pour chaque artefact data :

* le chemin du fichier,
* son extension et sa taille,
* la source de données associée,
* le mécanisme par lequel la donnée est accédée.

Cet inventaire constitue une base structurée pour analyser l’évolution des pratiques d’accès aux données au fil du temps et pour interpréter les analyses temporelles réalisées dans les étapes suivantes.



## Script `first_seen.py` 

Commande pour le lancer :

`
python first_seen.py ../../13_reveler_inegalites_cinema results/access_inventory.csv results
`

Ce script analyse l’historique Git du projet afin d’identifier la date d’apparition initiale des différentes sources de données et des mécanismes d’accès associés. À partir de l’inventaire des accès (access_inventory.csv), il parcourt les commits dans l’ordre chronologique et détecte le premier commit introduisant un fichier lié à chaque source (Allociné, CNC, Mubi, prédictions ML, etc.) ainsi qu’à chaque mode d’accès (fichier brut, scraping, extraction, migration de base de données, seed, notebook).

Le script produit deux fichiers CSV récapitulant, pour chaque source et chaque mode d’accès, le hash du premier commit, la date d’introduction et le mois correspondant. Ces résultats permettent de reconstituer une timeline d’évolution de l’accès aux données et d’analyser la mise en place progressive de mécanismes de structuration visant à améliorer la fiabilité et la reproductibilité du projet.


## Script `plot_first_seen.py` 

Produit un graphique de la chronologie d’introduction des sources de données et des modes d’accès.

Commandes à lancer :

`pip install pandas seaborn matplotlib`
`python plot_first_seen.py`

## Script `access_churn.py`

Commande à lancer : 

`python access_churn.py /chemin/vers/le/repo`
