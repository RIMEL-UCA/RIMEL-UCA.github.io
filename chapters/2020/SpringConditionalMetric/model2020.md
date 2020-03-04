---
layout: default
title : Modèle de chapitre pour 2020
date:   2020-01-03 22:00:00 +0100
---

## Authors

We are four students in last year of Polytech' Nice-Sophia specialized in Software Architecture :

* Laura Lopez &lt;laura.lopez1@etu.unice.fr&gt;
* Alexis Gardin &lt;alexis.gardin@etu.unice.fr&gt;
* Hugo Croenne &lt;hugo.croenne@etu.unice.fr&gt;
* Mathieu Paillart &lt;mathieu.paillart@etu.unice.fr&gt;

## I. Research context /Project

Préciser ici votre contexte.

Pourquoi c'est intéressant.


![Figure 1: Logo UCA](../assets/model/UCAlogoQlarge.png)


## II. Observations/General question

1. Commencez par formuler une question sur quelque chose que vous observez ou constatez ou encore une idée émergente. Attention pour répondre à cette question vous devrez être capable de quantifier vos réponses.
2. Préciser bien pourquoi cette question est intéressante de votre point de vue et éventuellement en quoi la question est plus générale que le contexte de votre projet \(ex: Choisir une libraire de code est un problème récurrent qui se pose très différemment cependant en fonction des objectifs\)
Cette première étape nécessite beaucoup de réflexion pour se définir la bonne question afin de poser les bonnes bases pour la suite.


Nous avions tous au sein de l'équipe déjà utilisé Spring, mais aucun de nous n'avais utilisé l'annotation Conditional et nous étions peu qualifiés sur les outils de paramétrage de Spring. La première question émergeant de nos réflexion est donc :

Comment les outils de paramètrage de Spring sont t-il utilisés et à quelle fréquence ?

Cette question d'ordre général a été notre première étape afin d'orienter nos recherches vers d'autres questions plus précises. Au fil de nos découvertes sur ces annotations, nous avons pu cibler un ensemble de nouvelles questions qu'il serait intéressant d'aborder. 

### Comment l'annotation @Conditional est-elle utilisée ? 

Apportant un ensemble de questions sous-jacentes
-> dans quels fichiers ?
-> dans quels projets ?
-> combien de fois par fichiers ? 
-> quelle déclinaison de l'annotation est la plus utilisée ? 

### Comment et à quelle fréquence est-elle testée ? 


### Quelle annotation de configuration est la plus utilisée parmi les existantes ? 

Nous avons pu relever plusieurs annotations de configuration, celle qui nous intéress pour cette étude est @Conditional mais nous avons tout de suite pensé qu'il serait intéressant de la mettre en corrélation avec ses semblables. 

### Pour quel type de configuration est utilisé chaque annotation ? 

### Comment est utilisée l'annotation @Value ? 

Principale concurrente de @Conditional en terme de popularité, il nous paraissait intéressant d'en savoir plus sur son utilisation et son fonctionnement. 

## III. information gathering

Préciser vos zones de recherches en fonction de votre projet,

1. les articles ou documents utiles à votre projet

Afin de mener à bien notre étude nos avons tout d'abord dû 
python + jupyter notebook + github
 
## IV. Hypothesis & Experiences

### Hypothèse n°1 : L'annotation @Conditional est utilisée dans des fichiers de configuration. 

### Hypothèse n°2 : L'annotation @Conditional est particulièrement utilisée dans les frameworks. 

### Hypothèse n°3 : L'annotation Conditional est souvent testée lorsqu'elle est utilisée. (on garde ça ?)

### Hypothèse n°4 : La répartition des annotations de paramétrage n'est pas homogène.

### Hypothèse n°5 : L'annotation @Value sert surtout à établir des règles concernant la valeur des paramètres. 

1. Il s'agit ici d'énoncer sous forme d' hypothèses ce que vous allez chercher à démontrer. Vous devez définir vos hypothèses de façon à pouvoir les _mesurer facilement._ Bien sûr, votre hypothèse devrait être construite de manière à v_ous aider à répondre à votre question initiale_.Explicitez ces différents points.
2. Test de l’hypothèse par l’expérimentation. 1. Vos tests d’expérimentations permettent de vérifier si vos hypothèses sont vraies ou fausses. 2. Il est possible que vous deviez répéter vos expérimentations pour vous assurer que les premiers résultats ne sont pas seulement un accident.
3. Explicitez bien les outils utilisés et comment.
4. Justifiez vos choix.

Le taux de @Value utiliser dans des if
La je vais, par projet, regarder la répartition des Annotation @Conditional et Autre pour essayer de classifier les libs, des projets concrêt
pour voir si on peut dire
Sur les 14 projet analyser le taux de @Conditional était Superieur pour les libs et @Value pour les projet concret

Hypothèses émises : 
Nous avions l'intuition dès le départ que l'annotation @Conditional était utilisée principalement au niveau de fichiers de configuration.
Nous avions l'intuition que l'annotation @Conditional était très utilisée au sein de framework.
Nous avions l'intuition que potentiellement plusieurs @Value identiques étaient utilisées au sein d'un même projet de manière transverse.

Pour démontrer ces hypothèses, nous avons effectuer des expérimentations automtisées sous forme de scripts python et également des expérimentations à la main dans certains cas. 

Tout d'abord, nous avons cherché à savoir quelle annotation @Conditional était la plus utilisée et où. 

Ensuite, nous avons souhaité comparer sa fréquence avec d'autres annotations de configuration.

Nous nous sommes également intéressés aux tests potentiellement effectué sur l'annotation @Conditional afin de voir si celle-ci était souvent testée.

Nous avons ensuite cherché les fichiers au sein desquels l'annotation est utilisée = nom de fichier.

Ensuite création d'un gros dataset fiable.

Pour finir, comparaison de @Value/@Profile/@Conditional/@Ressource


## V. Result Analysis and Conclusion

1. Analyse des résultats & construction d’une conclusion : Une fois votre expérience terminée, vous récupérez vos mesures et vous les analysez pour voir si votre hypothèse tient la route. 


à ne pas oublier : le biais d'utiliser uniquement github (modifié) 

## VI. Tools \(facultatif\)

Précisez votre utilisation des outils ou les développements \(e.g. scripts\) réalisés pour atteindre vos objectifs. Ce chapitre doit viser à \(1\) pouvoir reproduire vos expériementations, \(2\) partager/expliquer à d'autres l'usage des outils.

## VI. References

1. ref1
1. ref2
