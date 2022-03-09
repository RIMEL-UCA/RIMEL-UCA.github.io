# rimel-F-Tools

## Comment utiliser les différents outils

### [cppstats](https://www.se.cs.uni-saarland.de/projects/cppstats/)
Pour utiliser cppstats sur les différents projets à tester suivre les instruction d'installation sur leur [site](https://www.se.cs.uni-saarland.de/projects/cppstats/) ainsi que sur le repertoire github.
Ensuite lancer la commande :

```
$ cppstats --kind featurelocations
```

cette commande va générer les fichiers cppstats_featurelocation.csv et listoffeatures.csv dont vous aurez besoin pour générer les graphiques.

### Générer des graphiques avec compute_results.py

pour générer des graphiques avec le pourcentage de variabilité pour chaque type placer les fichiers cppstats_featurelocation.csv et listoffeatures.csv dans le dossier courant de compute_results.py puis lancer la commande.
vous aurez besoin pour lancer les scripts de python3 ainsi que télécharger les dépendance pour matplotlib.
Ensuite vous n'aurez qu'a lancer le script de génération de graphique

```
$ python3 compute_results.py  
```
Vous aurez un première graphique avec le pourcentage de variabilité de chaque type
Puis une fois avoir fermer cette première fenetre la seconde apparait avec le graphique du pourcentage de constante de chaque type

### Génerer le fichier csv de la variabilité par constante

Pour générer ce fichier il n'y a pas de version de python minimal requise. il suffit de lancer la commande suivant

```
$ python compute_feature.py > <fichier_cible>  
```

