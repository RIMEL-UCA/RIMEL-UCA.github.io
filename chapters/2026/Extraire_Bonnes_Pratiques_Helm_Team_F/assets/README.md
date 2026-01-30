# Étude des charts HELM
# Où trouver nos résultats
Les résultats, graphes, tableurs sont présents dans le dossier `results`. 
Si vous souhaitez les reproduire vous-même, la section suivante récapitule les étapes nécessaires.

# Comment obtenir les graphes et résultats
Les instructions se basent sur notre repo
[helm-best-practices](https://github.com/CamilleAntonios/helm-best-practices). Vous devez avant tout cloner le repo et aller dedans :
```
git clone https://github.com/CamilleAntonios/helm-best-practices.git
cd helm-best-practices
```


<span style="color: red;">**Première étape : [cliquez ici pour télécharger le fichier `repos_charts.zip`.](https://unice-my.sharepoint.com/:u:/g/personal/camille_antonios_etu_unice_fr/IQDtdRmt7FUtQZjRQlw504x_AYE6U5M9Cmqi1-caMQrPTnw?e=0rRrg6)**</span>
Décompressez-le à la racine du projet, c'est-à-dire qu'une fois décompressé, il faut obtenir cette structure :
```
helm-best-practices/
├── repos_charts/              
│   ├── argo-helm/             
│   ├── authorino-operator/    
│   ├── [other-chart-dirs]/    
```

(Ce dossier n'est pas à confondre avec le dossier `charts`, utilisé pour nos études qui ne varient pas en fonction du temps. Le dossier `repos_charts` contient lui les repos git des charts.)

**Seconde étape : exécutez le fichier `make_graphs.sh`**
```
chmod +x make_graphs.sh && ./make_graphs.sh
``` 

Nous avons parfois eu des soucis d'exécution sur Windows que nous n'avons pas réussi à régler, qui semblent liés à l'encodage de certains fichiers. Le script est tout en cas fonctionnel sur Linux.

Le script se base sur le fait que vous disposez de la commande `python`, et que vous disposez de python au minimum dans sa version `3.10.14`. Vous aurez besoin de `matplotlib` et `tomli` que vous possédez potentiellement déjà avec une installation typique de Python.

## Structure du repo

### Fichiers de calculs de graphs
#### `compute_graphs` 
Le fichier fait les graphs au sens qu'on avait prévu : faire varier le ratio de code smells par lignes de code, sur toutes nos variables.

#### `compute_graphs_peak_excluded`
Même chose qu'avant, mais exclue les valeurs qui dépassent le 95% des autres, que ce soit en ordonnée ou en abscisse.

#### `compute_graphs_ratio_per_file`
Comme `compute_graphs`, mais au lieu d'utiliser le pourcentage de codes smells par lignes de code, utilise le ratio de code smells par fichiers YAML.

#### `code_smells_calculator`
Utilisé par les autres fichiers pour comptabiliser les présences de mauvaises pratiques dans les charts.

#### `compute_mean_evolution`
Permet d'évaluer l'évolution du ratio de mauvaises pratiques au fil du temps.

#### `find_repo_tags`
Utilisé pour obtenir les commits à vérifier lors de l'étude de l'évolution au fil du temps.

### Dossiers des graphes produits
#### `graphs`
Contient les graphes "par défaut", portant sur l'étude de tous nos critères, par rapport au nombre de lignes. 

#### `graphs_ratio_per_file`
Comme `graphs`, mais effectue le ratio par rapport au nombre de fichiers plutôt que par rapport au nombre de lignes.

#### `graphs_with_peaks_excluded`
Comme `graphs`, mais exclue les valeurs dont l'abscisse est plus grande que 95% des autres abscisses ou dont l'ordonnée est plus grande que 95% des autres ordonnées, afin d'exclure les valeurs extrêmes.

### Fichiers CSV
#### `chart_infos.csv`
Contient les infos qu'on a noté sur Google Drive : dernières deliveries, provenance de la chart ...

#### `code_smells_report.csv` (généré par les scripts)
Résultat des codes smells

#### `final_report.csv` (généré par `aggregate_csv.py`)
Fait le lien entre `chart_infos.csv` et `code_smells_report.csv`, pour mettre dans un unique fichier toutes les infos nécessaires pour calculer les graphs. Le join se fait sur le nom de la chart, et ce doit donc être STRICTEMENT le même dans les deux fichiers CSV sources.

### Dossiers des charts
#### `charts`
Contient des dossiers contenant exactement les charts.

#### `repos_charts`
Contient les repos liés aux charts. Ajouté après le premier dossier `charts`, par nécessité d'utiliser les commits pour itérer dans le temps.
