---
layout: default
title: Quelle est la relation entre les algorithmes, préprocesseurs et leur utilisation dans les séries temporelles ?
date: 2022-01-10 22:00:00 +0100
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

Ainsi, dans le cas où nous arrivons à mettre en avant des algorithmes, il serait intéressant de pouvoir les optimiser, grâce à des `SAT Solver` ou encore en utilisant la `théorie des graphes` **si cela est possible**. Pleins de branches de métier seraient donc intéressées par nos trouvailles.

Finalement, pour nous aider à répondre à cette question, nous l'avons découper en plusieurs sous-questions :

1. Quels sont les principaux types de tâches sur ces données sur **OpenML** (*classification*, *clustering*, *détection d'anomalies*) ?
2. Quels sont les algorithmes et prétraitements les plus fréquemment utilisés ? Peut-on identifier des sous-workflows, des occurrences conjointes des mêmes algorithmes ?
3. En analysant les occurrences communes et l’ordre dans lequel ces algorithmes sont appelés, pouvons-nous identifier des conditions préalables sur les algorithmes ?
4. Existe-t-il des algorithmes qui ne sont utilisés que sur les séries temporelles ?

## III. Collecte d'informations

Nous pouvons extraire les tâches (*tasks*) et les flux (*flows*) terminés grâce à l'**API Python d'OpenML** et [sa documentation](https://docs.openml.org/Python-API/).
Pour cela, on utilise une librairie `Python` existante ([OpenML](https://pypi.org/project/openml/)), et nous les traitons avec `Pandas` et `Matplotlib`, pour en faire des graphes.

## IV. Hypotèses & Expériences

> 1. Il s'agit ici d'énoncer sous forme d' hypothèses ce que vous allez chercher à démontrer. Vous devez définir vos hypothèses de façon à pouvoir les mesurer facilement. Bien sûr, votre hypothèse devrait être construite de manière à _vous aider à répondre à votre question initiale_. Explicitez ces différents points.
> 2. Test de l’hypothèse par l’expérimentation. 
>> 1. Vos tests d’expérimentations permettent de vérifier si vos hypothèses sont vraies ou fausses.
>> 2. Il est possible que vous deviez répéter vos expérimentations pour vous assurer que les premiers résultats ne sont pas seulement un accident.
> 3. Explicitez bien les outils utilisés et comment.
> 4. Justifiez vos choix

We think that it exists some alogrithms and preprocessing that are better used in some conditions than others.  
Our focus will be to find the different approach used by searchers on OpenML.  
Then study the link between data, algorithms, preprocessing, flow and result.  
Finally, we visualize our results.

## V. Analyse des résultats & Conclusion

*Analyse des résultats & construction d’une conclusion : Une fois votre expérience terminée, vous récupérez vos mesures et vous les analysez pour voir si votre hypothèse tient la route.*

## VI. Outils & Ressources

Pour répondre à la questions générale que nous nous sommes posés, nous avons travaillé ensemble sur la plateforme en ligne [Google Colab](https://colab.research.google.com/). Vous trouverez donc l'accès à notre Notebook [ici](https://colab.research.google.com/).

## VII. Références

1. V. J. N. Rijn and J. Vanschoren, “Sharing RapidMiner Workflows and Experiments with
OpenML.,” in ceur-ws.org MetaSel@ PKDD/ECML, 2015, pp. 93--103, Accessed: Dec. 17, 2021. Available online [here](ftp://ceur-ws.org/pub/publications/CEUR-WS/Vol-1455.zip#page=98).