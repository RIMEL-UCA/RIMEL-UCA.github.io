# Étude des charts HELM
# Où trouver nos résultats
Les résultats, graphes, tableurs sont présents dans le dossier `RESULTATS_ETUDE`. 
Si vous souhaitez les reproduire vous-même, la section suivante récapitule les étapes nécessaires.

# Comment obtenir les graphes et résultats
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

Le script se base sur le fait que vous disposez de la commande `python`, et que vous disposez de python au minimum dans sa version `3.10.14`. Vous aurez besoin de `matplotlib` et `tomli` , `pandas`, `pyyaml` et `python-dateutil`

## Structure du repo

### Fichiers de calculs de graphs
#### `compute_graphs` 
Le fichier fait les graphs pour répondre à la question 1 : faire varier le ratio de code smells par lignes de code, sur toutes nos variables (date de dernier release, origine du projet, taille de la chart, nombre d'étoile sur ArtifactHub). 

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

#### `stacked_ratio_by_chart.py`
Script générant un graphique en barres empilées représentant, pour chaque chart la répartition des différentes mauvaises pratiques exprimée en ratio de mauvaises pratiques par ligne.

Il produit également des statistiques globales :
- charts les plus et les moins concernées,
- pratiques jamais détectées,
- pratiques les plus fréquentes,
- analyse de Pareto,
- nombre moyen de pratiques par chart.

#### `generate_graphs_over_time.py`
Script utilisé pour répondre à la Question 2.
Il analyse l’évolution du ratio de mauvaises pratiques par ligne,et par fichier YAML,pour une chart donnée, en comparant plusieurs commits successifs d’un dépôt Git.
Les commits analysés sont sélectionnés à intervalles réguliers (en mois).
Le script génère :
- un graphique de l’évolution du ratio par ligne,
- un graphique de l’évolution du ratio par fichier.

#### `generate_practice_evolution.py`
Script dédié à l’analyse détaillée de l’évolution des différentes mauvaises pratiques chacune individuellement au fil du temps pour une chart donnée.
Il produit :
- un graphique par mauvaise pratique,
- un graphique empilé représentant leur évolution conjointe,
- un fichier texte de synthèse indiquant :
  - les pratiques présentes au début et à la fin,
  - leur tendance globale (↗ / ↘ / =),
  - les pratiques nouvellement introduites.




### Dossiers des graphes produits
#### `graphs`
Contient les graphes "par défaut", portant sur l'étude de tous nos critères, par rapport au nombre de lignes. 

#### `graphs_ratio_per_file`
Comme `graphs`, mais effectue le ratio par rapport au nombre de fichiers plutôt que par rapport au nombre de lignes.

#### `graphs_with_peaks_excluded`
Comme `graphs`, mais exclue les valeurs dont l'abscisse est plus grande que 95% des autres abscisses ou dont l'ordonnée est plus grande que 95% des autres ordonnées, afin d'exclure les valeurs extrêmes.

#### `graphs_practices_over_time`
Graphes détaillant l’évolution des différentes mauvaises pratiques au fil du temps pour une chart donnée.

#### `graphs_stacked`
Graphes en barres empilées représentant la répartition des mauvaises pratiques par chart.


### Pipeline GitHub Actions – Analyse de l’évolution des pratiques
#### Objectif du pipeline

Ce pipeline GitHub Actions automatise l’analyse de l’évolution des mauvaises pratiques Helm au fil du temps pour une chart donnée.
Il permet de reproduire de manière fiable et contrôlée l’analyse temporelle décrite dans la Question 2 de l’étude, sans dépendre d’un environnement local.

Le pipeline :
- clone dynamiquement un dépôt Helm cible,
- parcourt son historique Git à intervalles réguliers,
- calcule le ratio de mauvaises pratiques par ligne pour chaque commit sélectionné,
- génère des graphiques illustrant l’évolution globale et détaillée des pratiques,
- publie ces graphiques comme artefacts téléchargeables.

#### Déclenchement du pipeline

Le pipeline est déclenché automatiquement lors d’un push sur l’un des fichiers suivants :
```
on:
  push:
    paths:
      - '.github/workflows/analysis-practice-over-time.yml'
      - 'graph-analyze-practice.toml'
      - 'generate_practice_evolution.py'
      - 'code_smells_calculator.py'
```

Ainsi, toute modification :

de la configuration de l’analyse,du script d’évolution des pratiques, ou des règles de détection des mauvaises pratiques,entraîne automatiquement une nouvelle exécution du pipeline.

Le dépôt Helm cible n’est pas codé en dur.
Son URL est lue dynamiquement depuis le fichier de configuration : graph-analyze-practice.toml


il génère :

- un graphique par mauvaise pratique,
- un graphique empilé global,
- un fichier texte de synthèse.

#### Publication des résultats

Les graphiques générés sont publiés comme artefacts GitHub Actions `uses: actions/upload-artifact@v4`

Ils peuvent être téléchargés directement depuis l’interface GitHub Actions après l’exécution du pipeline.

#### Comment lancer le pipeline

⚠️ Important : ce pipeline nécessite un fork du dépôt.

GitHub Actions ne peut pas être déclenché sur un dépôt externe sans droits suffisants.
Pour exécuter le pipeline :
- Forkez le dépôt d’analyse sur votre compte GitHub.
- Activez GitHub Actions sur votre fork (si ce n’est pas déjà fait).
- Modifiez au besoin le fichier graph-analyze-practice.toml pour cibler la chart souhaitée.
- Effectuez un push sur l’un des fichiers surveillés.

Le pipeline se lance automatiquement.
Les résultats sont ensuite accessibles dans l'onglet artifacts.



### Fichiers CSV
#### `chart_infos.csv`
Contient les infos qu'on a noté sur Google Drive : dernières deliveries, provenance de la chart ...

#### `code_smells_report.csv` (généré par les scripts)
Résultat des codes smells

#### `final_report.csv` (généré par `aggregate_csv.py`)
Fait le lien entre `chart_infos.csv` et `code_smells_report.csv`, pour mettre dans un unique fichier toutes les infos nécessaires pour calculer les graphs. Le join se fait sur le nom de la chart, et ce doit donc être STRICTEMENT le même dans les deux fichiers CSV sources.


#### `final_report.csv` (généré par les scripts)
fichier généré par code_smells_calculator.py.
Il contient le détail des mauvaises pratiques par chart et par type de pratique.
Ce fichier est notamment utilisé par :
stacked_ratio_by_chart.py pour générer le graphique de répartition des mauvaises pratiques par chart, les analyses statistiques globales (fréquence des pratiques, pratiques rares ou dominantes, analyse de Pareto).

### Dossiers des charts
#### `charts`
Contient des dossiers contenant exactement les charts.

#### `repos_charts`
Contient les repos liés aux charts. Ajouté après le premier dossier `charts`, par nécessité d'utiliser les commits pour itérer dans le temps.



