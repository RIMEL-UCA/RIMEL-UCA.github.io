---
layout: default
title : Comment est configuré un projet Spring via les @Conditional ?
date:   2020-01-11 22:00:00 +0100
---

## Authors

We are four students in last year of Polytech' Nice-Sophia specialized in Software Architecture :

* Brandon Fontany-Legall &lt;brandon.fontany--legall@etu.unice.fr&gt;
* Thomas Mahe &lt;thomas.mahe@etu.unice.fr&gt;
* Aymeric Valdenaire &lt;aymeric.valdenaire@etu.unice.fr&gt;
* Michel Marmone--Marini &lt;michel.marmone--marini@etu.univ-cotedazur.fr&gt;

## I. Research context /Project

Spring est un framework qui permet de concevoir des applications Java plus simplement. Spring fournit un ensemble d'annotation pour aider le développeur dans la confection de son application. 
Spring nous permet notamment de créer des composant et de les faire communiquer simplement. 

Dans ce context, il est intéressant de voir comment créer et configurer les dit composant dans les projets Spring. 
Nous allons plus spécifiquement étudier comment configurer la création de beans en fonction de conditions au préalable défini.

Dans cette optique, Spring 3.1 a introduit l'annotation @Profile qui permet de créer des beans en fonction des profils qui sont en général défini dans les fichiers de configuration xml.
Dans Spring 4, l'annotation @Profile existe toujours cependant une nouvelle anotation a été introduite qui est @Conditional. Cette annotation permet, tout comme @Profile, de configurer les beans. 
Cependant, @Conditional permet de créer une configuration sans profile préalable, notamment basé sur des stratégies sous format booléennes. Cette nouvelle approche permet une plus grande liberté de création de stratégie, notamment, n'importe quelle statégie peut être utilisée tant qu'elle est testable par valeur booléenne.

Nous allons donc, dans notre étude, nous concentré sur cette nouvelle annotation @Conditional et quels sont les impacts de cette annotation sur les projets Srping.

![Figure 1: Logo UCA](../assets/model/UCAlogoQlarge.png){:height="50px" }


## II. Observations/General question
### Comment sont configurés les projets Spring à l'aide de l'annotation @Conditionnal

Nous nous sommes alors intéressé à ce changement effectué de @Profile vers @Conditional et notamment pourquoi @Profile a été remplacé par @Conditional.
D'après ce que nous avons vu dans nos recherches, ce changement a été possé par une demande de la part de la communauté, il nous parrait donc intéressant quels sont les cas d'usage qui permettent une telle modification.

Pour ce faire, nous nous sommes poser plusieurs sous-questions 

#### 1. Est-ce que le @Conditional à été adopté ?

L'annotation a été poussée par la communauté, comme vu précédemment. 
Il est donc, d'après nous, interessant de savoir si cette dernière à été adopté par la communauté.
En effet, ceci permettrais de savoir si cette nouvelle annotation répond bien aux attentes qui ont entraîné sa création.

Pour se faire, nous pouvons regarder dans les projets Github utilisant Spring le taux d'appartion de l'annotation @Conditional

#### Comparaison de l’utilisation de @Profile et @Conditional

Après l'introduction de la nouvelle annotation, il est interessant de savoir si l'ancienne annotation @Profile est remplacée par @Conditional.

Nous pouvons donc, pour faire ceci, récupérer la liste des git dif dans chaque projet Spring utilisant l'une des deux annotations.
Nous aurons donc la liste des ajouts et retrait de chacune des annotations.

#### Comment sont utilisés les différentes variantes de @Conditional ?

Nous avons observer que l'annotation @Conditional n'a pas été seul. 
En effet, une multitude de variantes ont été introduites qui sont :
- @ConditionalOnProperty : @Conditional qui check si une propriété, qui par défaut est présente dans l'environnement a une valeur spécifique
- @ConditionalOnWebApplication : @Conditional qui est valide dans le cas où l'application courante est une application web. Cette application web peut notamment être de type spécifique, dont le mode servlet ou réactive.
- @ConditionalOnClass : @Conditional qui est valide uniquement si la classe spécifiée est présente dans le classpath.
- @ConditionalOnMissingBean: @Conditional qui est valide uniquement quand aucun autre bean n'est pas déjà instancié.
- @ConditionalOnExpression : @Conditional en basé sur les résultats SpEL (Spring Expression Language)

Il est donc intéressant de savoir comment la communauté Spring utilise ces annotations. 
Pour ce faire, nous allons regarder les occurences dans les différents projets Spring de chaque variantes et voir dans quel contexte avec les raisons d'ajout/remplacement.

#### Quels autres mécanismes peuvent remplacer @Conditional ?

#### Est-ce que @Conditional est vraiment utile et dans quels cas ?  Comment configurer son projet sans @Conditional ?

#### Comparaison de l’utilisation de @Conditional dans le code et dans les tests


## III. information gathering

### La récupération des projets
Pour la base de nos recherches, nous avons utilisé les projets Github que nous allons récupérer par clonnagede façon automatique.
Bien évidemment, tout les projets Github ne sont pas interessant pour nous et c'est pourquoi nous devons mettre des règles de filtrage et notamment en utilisant l'API de Github.
- Pour commencer, le projet doit contenir du java ce qui va permettre d'écarter énormément de projets
- De plus, nous avons, trier par mot clé à l'aide du mot clé "spring"
- Pour finir, nous avons prit les projets avec le plus d'étoiles pour prendre les projets qui ont le plus d'étoile et qui sont donc susceptiblement plus intéressant à étudier

### La récupération de données dans les projets
Une fois les projets récupérés, nous avons utilisé des scripts Python pour récupérer les différentes données.

#### Pydriller
Pydriller est un framework Python qui permet d'analyser les répositories Git.
Nous pouvons faire notamment des analises de commits ou encore de diffs par exemple.

#### Autre scripts
En addition à Pydriller, nous avons effectué de multiples codes python pour analyser le code des différents projets.

## IV. Hypothesis & Experiences

### Première analyse 
Nous avons prit comme hypothèse de départ que l'annotation avait été adoptée.
Pour vérifier cette hypothèse nous avons récupérer des projets et regarder le nombre d'apparition dans ces projets.

Dans cette première expérimentation, nous avons prit 12 projets choisit au préalable. 
![Projet par commits](./assets/projectsByCommits.png)

Nous avons notamment, dans les plus remarquable, BroadleafCommerce qui est un framework de site e-commerce, Spring security qui fournit un service de sécurité pour Spring ou encore Spring IDE qui améliore Eclipse avec des outils pour créer des applications basées sur Spring.

Avec ce dataset, nous avons analysé le nombre d'occurence de @Conditional. 
![Occurences des Conditional](./assets/firstProjectsConditionalOccurences.png)

Nous nous sommes donc apercu que, au niveau du datase choisit, l'utilisation des annotations dépend du projet et donc du contexte.
Nous avons aussi analysé les différentes variantes dans le projet BroadLeafCommerce qui utilise le plus de Conditional de notre dataset.

![Occurences des variantes de Conditional](./assets/firstProjectsConditionalVariants.png)

Nous nous sommes apperçu qu'une annotation en particulière dans ce projet est utilisée (@ConditionalOnTemplating).
Or, cette annotation n'est pas une annotation existante dans Spring mais une annotation créée par BroadLeafCommerce.
Cette fameuse annotation créée utilise une classe de mathcher qui correspond à l'ancienne implémentation via @Profile.
Nous avons donc déjà une première réponse à notre questionnement.

Cependant, ce dataset n'est pas assez grand pour en sortir des conclusions.

### Agrandissement du dataset
 

## V. Result Analysis and Conclusion

> 1. Analyse des résultats & construction d’une conclusion : Une fois votre expérience terminée, vous récupérez vos mesures et vous les analysez pour voir si votre hypothèse tient la route. 

## VI. References

1. https://docs.spring.io/spring-framework/docs/current/javadoc-api/overview-summary.html
2. https://docs.spring.io/spring-framework/docs/current/javadoc-api/org/springframework/context/annotation/Conditional.html 
3. https://github.com
4. https://blog.ippon.fr/2014/01/07/conditional-de-spring-4-au-coeur-de-lauto-configuration-de-spring-boot/
5. https://javapapers.com/spring/spring-conditional-annotation/ 
6. https://github.com/BroadleafCommerce/BroadleafCommerce
7. https://github.com/spring-projects/spring-security
8. https://github.com/spring-projects/spring-ide

