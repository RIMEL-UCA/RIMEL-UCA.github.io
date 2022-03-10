---
layout: default
title: Quelle est la relation entre les algorithmes, préprocesseurs et leur utilisation dans les séries temporelles ?
date: 2022-02-25 18:00:00 +0100
---

## Auteurs

Nous sommes 5 étudiants en dernière année à Polytech Nice Sophia, dans la spécialité **Architecture Logicielle** :

* Eric Boudin (<eric.boudin@etu.univ-cotedazur.fr>)
* Clément Monestier (<clement.monestier@etu.univ-cotedazur.fr>)
* Florian Naud (<florian.naud@etu.univ-cotedazur.fr>)
* Lucas Rakotomalala (<lucas.rakotomalala@etu.univ-cotedazur.fr>)
* Loïc Rizzo (<loic.rizzo@etu.univ-cotedazur.fr>)

## I. Contexte

**OpenML** est un environnement collaboratif en ligne pour le *machine learning* où les chercheurs et les praticiens peuvent partager des ensembles de données (*datasets*), des flux de travail (*workflows*) et des expériences. **OpenML** est particulièrement utilisé pour la recherche en méta-apprentissage (*meta-learning*) ; en étudiant un grand nombre d'expériences passées, il devrait être possible d'apprendre la relation entre les données et le comportement de l'algorithme.

Ainsi, dans ce projet, nous voulons extraire des connaissances sur les workflows de traitement des séries temporelles. En effet, nous voulons connaître les liens qui peuvent exister entre les données et comment elles sont traités. On pourrait trouver une relation entre
celles-ci qui nous amèneraient à mieux comprendre le meilleur contexte d'utilisation des 
algorithmes. Concrètement, cela permettrait d'utiliser l'algorithme adéquat étant donné un *dataset* qui traite de série temporelle. Pour cela, il faudrait établir des corrélations (on pense ici à [`Kendall`, `Spearman` ou `Pearson`](https://datascience.stackexchange.com/questions/64260/pearson-vs-spearman-vs-kendall/)) entre les données et les résultats de l'algorithme.


## II. Observations & Question générale

La découverte d'**OpenML** a fait émerger une question en nous :

**Quelle est la relation entre les algorithmes, préprocesseurs et leur utilisation dans les séries temporelles ?**

Cette question est d'autant plus intéressante que les secteurs des séries temporelles est très vaste, puisqu'ils vont de la météoriologie aux prix en bourse en passant par la démographie. En effet, elles permettent notamment de trouver des tendances ou encore de prédire des données futures.

Ainsi, dans le cas où nous arrivons à mettre en avant des algorithmes, il serait intéressant de pouvoir les optimiser, grâce à des `SAT Solver` ou encore en utilisant la `théorie des graphes` **si cela est possible**. Beaucoup de branches de métier seraient donc intéressées par nos trouvailles.

Finalement, pour nous aider à répondre à cette question, nous l'avons découpée en plusieurs sous-questions, à savoir :

1. Quels sont les principaux types de tâches sur les données issues des datasets de séries temporelles sur **OpenML** (Exemples: *classification*, *clustering*, *détection d'anomalies*) ?
2. Quels sont les algorithmes et prétraitements les plus fréquemment utilisés ? Peut-on identifier des sous-workflows, des occurrences conjointes des mêmes algorithmes ?
3. Existe-t-il des algorithmes qui ne sont utilisés que sur les séries temporelles ?

## III. Collecte d'informations

Nous extrayons les tâches (*tasks*), les runs  (*runs*), les flux (*flows*) sur différents datasets grâce à l’[**API Python d'OpenML**](https://docs.openml.org/Python-API/).

## IV. Hypothèses & Expériences

### 1. Quels sont les principaux types de tâches sur les données issues des datasets de séries temporelles sur **OpenML** (Exemples: *classification*, *clustering*, *détection d'anomalies*) ?

#### Hypothèse

Lors de la découverte du sujet et de sa lecture, nous nous sommes rapidement dit qu'au vue du petit nombre de *datasets* et de *tâches* disponibles sur **OpenML**, il y a de fortes chances que seuls quelques types de *tâches* soient utilisés.

On cherche donc à démontrer qu'uniquement certaines *tâches* sont effectuées sur des séries temporelles. Cela nous permettra d'avoir un point de départ pour la recherche des algorithmes les plus fréquemment utilisés pour les séries temporelles.

#### Expérimentation

Dans un premier, nous avons regardé manuellement un *dataset*, à savoir [JapaneseVowels](https://www.openml.org/d/375), pour chercher les types de tâche les plus utilisés. On remarque que les *tâches* où il y a plus de *runs* sont les tâches de type *Découverte de sous-groupes*.

Dans un second temps, nous avons utilisé l'**API Python d'OpenML** pour extraire les *tâches* du dataset [JapaneseVowels](https://www.openml.org/d/375) et les afficher sous forme de diagramme en bâtons.

Finalement, nous avons utilisé l'**API Python d'OpenML** pour extraire les *tâches* de tous les datasets fournis et afficher sous forme de diagramme en bâtons les *tâches* les plus réalisées. Pour cela, nous avons repris le code existant développé pour le dataset [JapaneseVowels](https://www.openml.org/d/375) et nous l'avons adapté pour extraire les *tâches* de tous les datasets.

### <a name="4HE"></a> 2. Quels sont les algorithmes et prétraitements les plus fréquemment utilisés ? Peut-on identifier des sous-workflows, des occurrences conjointes des mêmes algorithmes ?

#### Hypothèse

On s’attend à trouver des algorithmes plus utilisés que d’autres, et des sous-workflows apparaissant plusieurs fois ensemble pour construire une chaîne d'algorithmes. 

#### Expérimentation

Nous avons utilisé l'**API Python d'OpenML** pour extraire les `flows` des dataset considéré comme des séries temporelles. Pour cela on a d'abord récupéré toutes tentatives de runs effectués sur toutes les tâches des *datasets*. Puis nous avons trié les `flows` de façon à ne pas avoir d’utilisation doublon, c'est-à-dire lorsqu’une même tâche est effectuée par le même auteur avec le même `flow`. De cette manière, nous pouvons obtenir les occurrences de `flow` sans compter les simples changements de paramètre.

On construit deux types de graphiques, un premier diagramme en bâton montrant simplement les `flows` les plus utilisés. Puis un graphe valué permettant de visualiser les liens et l'occurrence de chaque `flows`, pour cette dernière visualisation, nous trions les `flows` à afficher pour plus de lisibilité, autrement dit, ne pas afficher les `flows` à occurrence très faible (`< 20`) et sans lien à d’autres `flows`.

### 3. Existe-t-il des algorithmes qui ne sont utilisés que sur les séries temporelles ?

#### Hypothèse

On s'attend à trouver des algorithmes qui ne sont utilisés que sur les séries temporelles. Cela nous permettra de mettre en évidence de bonnes pratiques pour le traitement de séries temporelles.

#### Expérimentation

Pour savoir si notre hypothèse est correcte, nous avons pris des datasets qui **ne sont pas des séries temporelles** parmi les plus utilisés sur **OpenML**. Nous avons ensuite comparé les occurrences de `flows` (et donc d'algorithmes) entre les *datasets* de séries temporelles et ceux qui ne le sont pas, dans le but de trouver des algorithmes étant davantage utilisés pour le traitements des séries temporelles et peu utilisés dans l'autre cas. Pour cela, nous avons utilisé le même code qu'à la sous-question précédente afin d'obtenir le même type de résultat sur l'occurrence des `flows`.

Finalement, on a comparé les résultats obtenus avec ceux de la sous-question précédente.

## V. Analyse des résultats & Conclusion

### 1. Quels sont les principaux types de tâches sur les données issues des datasets de séries temporelles sur **OpenML** (Exemples: *classification*, *clustering*, *détection d'anomalies*) ?

Sur le dataset JapaneseVowels le type de tache majoritaire est la **découverte de sous-groupe**. 

Voici les tâches par type les plus réalisées sur le dataset [JapaneseVowels](https://www.openml.org/d/375) :
![Occurrence des tâches par type pour JapaneseVowels](../assets/OpenMLAnalysis/Occurrence%20des%20tâches%20par%20type%20pour%20JapaneseVowels.png "Occurrence des tâches par type pour JapaneseVowels")

Sur l’ensemble des *datasets* de séries temporelles les types de tâches les plus représentés sont la **classification supervisée** et la **découverte de sous-groupe**. 

Voici les tâches par type les plus réalisées sur les *datasets* de séries temporelles sur **OpenML** :
![Occurrence des tâches par type](../assets/OpenMLAnalysis/Occurrence%20des%20tâches%20par%20type.png "Occurrence des tâches par type")

Les types de travaux sont majoritairement restreints à deux tâches, mais cela peut différer d’un dataset à l’autre. Comme le montre les résultats sur le dataset [JapaneseVowels](https://www.openml.org/d/375) qui présente peu de runs sur les tâches de type classification supervisée. 

### 2. Quels sont les algorithmes et prétraitements les plus fréquemment utilisés ? Peut-on identifier des sous-workflows, des occurrences conjointes des mêmes algorithmes ?

Voici les `flows` les plus utilisés sur les *datasets* de séries temporelles sur **OpenML** :

![Occurrence des flows](../assets/OpenMLAnalysis/Flows%20les%20plus%20fréquemment%20utilisés.png "Occurrence des flows")

Parmis les cinq algorithmes les plus utilisés, on remarque que **DecisionStump** est présent deux fois dans le classement, à la première et la troisième place. Seule sa version diffère, cela fait de lui l'algorithme le plus utilisé pour les tâches sur les datasets de séries temporelles disponibles sur **OpenML**. 
Parmi ces quatre algorithmes, on a :

- **DecisionStump**, qui est un modèle d'apprentissage automatique composé d'un arbre de décision à un niveau. C'est-à-dire qu'il s'agit d'un arbre de décision avec un nœud interne (la racine) qui est immédiatement connecté aux nœuds terminaux (ses feuilles). Une souche de décision fait une prédiction basée sur la valeur d'une seule caractéristique d'entrée.

- **J48 (C4.5)** est un algorithme de classification supervisé, publié par Ross Quinlan. Il est basé sur l'algorithme ID3 auquel il apporte plusieurs améliorations. Il a pour but de produire un modèle de type arbre de décision à partir d'un échantillon d'apprentissage.

- **RepTree** est une méthode pour générer un arbre de décision à partir d'un ensemble de données donné. Il est considéré comme une extension de C4.5 en améliorant la phase d'élagage en utilisant l'élagage à erreur réduite.

- **ZeroR** est la méthode de classification la plus simple qui s’appuie sur la cible et ignore tous les prédicteurs. Le classificateur ZeroR prédit simplement la catégorie majoritaire (classe). Bien qu'il n'offre aucune fonctionnalité de prédiction, il est utile pour déterminer une performance de base servant de référence pour d'autres méthodes de classification.

On constate logiquement que le type de tâche majoritaire sur les séries temporels influe sur les algorithmes utilisés. En effet, on retrouve des algorithmes permettant de réaliser de la **classification supervisée** parmi les plus utilisés. 
De plus, on remarque une forte utilisation d’**arbre de décision** pour le traitement des séries temporelles.

On s'intéresse maintenant au(x) lien(s) entre les algorithmes.
Sur les graphiques suivants on a représenté chaque algorithme par un point de plus en plus gros suivant son nombre d’utilisations, et les liens entre deux algorithmes représentent le nombre d'apparitions ensemble dans la même chaîne de travail. Plus le lien est épais, plus ce nombre est élevé.

<iframe src="../assets/OpenMLAnalysis/graph_occurrence_conjointe.html" width="100%" height="500px"></iframe>
Cliquez [ici](../assets/OpenMLAnalysis/graph_occurrence_conjointe.html){:target="_blank"} pour afficher le graphique en grand.

Sur ce premier graphique, il est difficile de ressortir un quelconque résultat. Nous avons donc effectué un tri, comme expliqué dans la section [Hypothèses & Expériences](#4HE), pour obtenir une meilleure visualisation.

<iframe src="../assets/OpenMLAnalysis/graph_occurrence_conjointe_filtré.html" width="100%" height="500px"></iframe>
Cliquez [ici](../assets/OpenMLAnalysis/graph_occurrence_conjointe_filtré.html){:target="_blank"} pour afficher le graphique en grand.

Dans cette nouvelle visualisation, on distingue différents groupes d'algorithmes. Parmi ces algorithmes utilisés conjointement plusieurs fois avec d’autres algorithmes, créant ainsi des chaînes de travail clairement identifiées, aucune tâche ne se distingue. Ces groupes de *workflows* sont parfois destinés à du prétraitement, ou au traitement pur des données, ou encore à l'évaluation des résultats.

On remarque aussi que les algorithmes les plus utilisés sont souvent isolés. 

Enfin, cette visualisation permet de mettre en évidence qu’il n’y a pas de dépendance entre les différentes chaînes de *workflow*. L’utilisation d’une sous-chaîne ne semble pas influer le choix de l’utilisation d’un algorithme de traitement (comme **DecisionStump**). De la même manière, on remarque aussi que l’utilisation d’un algorithme de traitement ne force pas l’utilisation d’une même sous-chaîne d’algorithmes de prétraitement.

### 3. Existe-t-il des algorithmes qui ne sont utilisés que sur les séries temporelles ?

On a déjà montré l'influence qu'avait une tâche effectuée sur le choix de l’algorithme pour les séries temporelles. Sur des tâches similaires (**Classification supervisée** et **découverte de sous groupe**), parmi les algorithmes les plus utilisés, aucun ne semble être spécifique au traitement des séries temporelles. On obtient quelques résultats par exemple, le flow (id : `135`) **BestFirst**, mais les différences entre les datasets de séries temporelles et ceux qui ne le sont pas sont trop minces. En effet cet écart pourrait être dû à la sélection des datasets pour représenter ceux qui ne sont pas des séries temporelles. 

```json
{
  "135": {
    "nonTimeSeries": 0,
    "timeSeries": 16
  },
  "364": {
    "nonTimeSeries": 7,
    "timeSeries": 20
  },
  "365": {
    "nonTimeSeries": 13,
    "timeSeries": 35
  },
  "375": {
    "nonTimeSeries": 9,
    "timeSeries": 22
  },
  "377": {
    "nonTimeSeries": 20,
    "timeSeries": 55
  },
  "379": {
    "nonTimeSeries": 6,
    "timeSeries": 17
  },
  "380": {
    "nonTimeSeries": 10,
    "timeSeries": 30
  },
  "386": {
    "nonTimeSeries": 10,
    "timeSeries": 25
  },
  "389": {
    "nonTimeSeries": 6,
    "timeSeries": 18
  },
  "390": {
    "nonTimeSeries": 9,
    "timeSeries": 32
  },
  "391": {
    "nonTimeSeries": 8,
    "timeSeries": 19
  },
  "414": {
    "nonTimeSeries": 6,
    "timeSeries": 16
  },
  "421": {
    "nonTimeSeries": 10,
    "timeSeries": 22
  },
  "424": {
    "nonTimeSeries": 7,
    "timeSeries": 16
  },
  "1068": {
    "nonTimeSeries": 42,
    "timeSeries": 98
  },
  "1069": {
    "nonTimeSeries": 19,
    "timeSeries": 49
  },
  "1077": {
    "nonTimeSeries": 20,
    "timeSeries": 48
  },
  "1080": {
    "nonTimeSeries": 7,
    "timeSeries": 16
  },
  "1090": {
    "nonTimeSeries": 50,
    "timeSeries": 113
  },
  "1097": {
    "nonTimeSeries": 12,
    "timeSeries": 30
  },
  "1105": {
    "nonTimeSeries": 7,
    "timeSeries": 17
  },
  "1110": {
    "nonTimeSeries": 10,
    "timeSeries": 20
  },
  "1112": {
    "nonTimeSeries": 7,
    "timeSeries": 17
  },
  "1154": {
    "nonTimeSeries": 14,
    "timeSeries": 34
  },
  "1164": {
    "nonTimeSeries": 12,
    "timeSeries": 26
  },
  "1165": {
    "nonTimeSeries": 5,
    "timeSeries": 17
  },
  "1168": {
    "nonTimeSeries": 10,
    "timeSeries": 27
  },
  "1169": {
    "nonTimeSeries": 10,
    "timeSeries": 27
  },
  "1172": {
    "nonTimeSeries": 6,
    "timeSeries": 17
  },
  "1175": {
    "nonTimeSeries": 10,
    "timeSeries": 21
  },
  "1185": {
    "nonTimeSeries": 7,
    "timeSeries": 16
  },
  "1191": {
    "nonTimeSeries": 8,
    "timeSeries": 18
  },
  "1194": {
    "nonTimeSeries": 7,
    "timeSeries": 17
  },
  "1720": {
    "nonTimeSeries": 64,
    "timeSeries": 31
  },
  "1721": {
    "nonTimeSeries": 38,
    "timeSeries": 20
  },
  "1724": {
    "nonTimeSeries": 52,
    "timeSeries": 24
  },
  "1728": {
    "nonTimeSeries": 27,
    "timeSeries": 16
  },
  "1729": {
    "nonTimeSeries": 62,
    "timeSeries": 30
  },
  "1745": {
    "nonTimeSeries": 47,
    "timeSeries": 24
  },
  "1820": {
    "nonTimeSeries": 37,
    "timeSeries": 23
  },
  "2049": {
    "nonTimeSeries": 30,
    "timeSeries": 16
  },
  "2050": {
    "nonTimeSeries": 31,
    "timeSeries": 16
  },
  "2118": {
    "nonTimeSeries": 48,
    "timeSeries": 24
  },
  "8775": {
    "nonTimeSeries": 34,
    "timeSeries": 16
  },
  "8776": {
    "nonTimeSeries": 34,
    "timeSeries": 16
  },
  "8777": {
    "nonTimeSeries": 34,
    "timeSeries": 16
  },
  "8778": {
    "nonTimeSeries": 46,
    "timeSeries": 22
  },
  "8779": {
    "nonTimeSeries": 46,
    "timeSeries": 22
  },
  "8780": {
    "nonTimeSeries": 46,
    "timeSeries": 22
  },
  "8781": {
    "nonTimeSeries": 46,
    "timeSeries": 22
  },
  "8782": {
    "nonTimeSeries": 46,
    "timeSeries": 22
  }
}
```

Voici les `flows` les plus utilisés sur les datasets qui ne sont pas des séries temporelles sur **OpenML** :

![Occurrence des flows](../assets/OpenMLAnalysis/Flows%20les%20plus%20fréquemment%20utilisés%20sur%20les%20datasets%20qui%20ne%20sont%20pas%20des%20series%20temporelles.png "Occurrence des flows")

Une nouvelle fois, on retrouve la forte utilisation des deux algorithmes **DecisionStump** et **J48**. À fortiori, le choix des algorithmes n’est influencé que par le type de tâches à exécuter, quel que soit le type de dataset sur lequel on lance la tâche.
Finalement, notre étude ne semble pas montrer qu’il existe d’algorithmes dédiés au traitement des séries temporelles.

## VI. Outils & Ressources

Pour répondre à la question générale et aux sous-questions que nous nous sommes posés, nous avons travaillé ensemble sur la plateforme en ligne [Google Colab](https://colab.research.google.com/). Vous trouverez l'accès à notre Notebook [ici](https://colab.research.google.com/drive/1oU2_6NU-mEvBCcgkTCeQFOOK52kMT9fN?usp=sharing). Ce même Notebook est aussi disponible [ici](../assets/OpenMLAnalysis/Notebook_RIMEL_Groupe_H.ipynb).
Nous avons utilisé [`Pandas`](https://pandas.pydata.org/), [`Matplotlib`](https://matplotlib.org/), [`Plotly`](https://plotly.com/) et [`NetworkX`](https://networkx.org/) pour traiter les données et créer des graphiques sur lesquels nous avons basé nos analyses et conclusions.

## VII. Références

[1] V. J. N. Rijn and J. Vanschoren, “Sharing RapidMiner Workflows and Experiments with
OpenML.,” in ceur-ws.org MetaSel@ PKDD/ECML, 2015, pp. 93--103, Accessed: Dec. 17, 2021. Available online [here](ftp://ceur-ws.org/pub/publications/CEUR-WS/Vol-1455.zip#page=98).

[2] H. A. Nguyen, R. Dyer, T. N. Nguyen, and H. Rajan, “Mining preconditions of APIs in large-scale code corpus,” in Proceedings of the ACM SIGSOFT Symposium on the Foundations of Software Engineering, 2014, vol. 16-21-Nove, doi: 10.1145/2635868.2635924.