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

> Préciser vos zones de recherches en fonction de votre projet,

1. les articles ou documents utiles à votre projet

Pour commencer nos recherches, nous avons prévu de trouver quelques projets open source (les plus conséquents possible) dans lesquels le framework Spring est utilisé. Nous avons commencé nos recherches de projets sur Github et Gitlab.

2. les outils

 Après avoir trouvé ces projets, nous prévoyons d'appliquer des scripts qui analyseront le code et d'autres qui analyseront des repo git pour trouver des informations telles que des mots clefs dans des fichiers (utilisation de @Conditional, @Resource, @Value) et d'autres mots clefs dans par exemple des messages de commit, titre d'issue pour pouvoir voir dans quels cas et pour quels besoins ces annotations ont été utilisées.
 Nous avons aussi prévu de nous appuyer sur des outils d'analyse statique de code sur les projets trouvés précédemment tel que SonarQube.
 
## IV. Hypothesis & Experiences

> 1. Il s'agit ici d'énoncer sous forme d' hypothèses ce que vous allez chercher à démontrer. Vous devez définir vos hypothèses de façon à pouvoir les _mesurer facilement._ Bien sûr, votre hypothèse devrait être construite de manière à v_ous aider à répondre à votre question initiale_.Explicitez ces différents points.
> 2. Test de l’hypothèse par l’expérimentation. 1. Vos tests d’expérimentations permettent de vérifier si vos hypothèses sont vraies ou fausses. 2. Il est possible que vous deviez répéter vos expérimentations pour vous assurer que les premiers résultats ne sont pas seulement un accident.
> 3. Explicitez bien les outils utilisés et comment.
> 4. Justifiez vos choix

## V. Result Analysis and Conclusion

> 1. Analyse des résultats & construction d’une conclusion : Une fois votre expérience terminée, vous récupérez vos mesures et vous les analysez pour voir si votre hypothèse tient la route. 

## VI. Tools \(facultatif\)

> Précisez votre utilisation des outils ou les développements \(e.g. scripts\) réalisés pour atteindre vos objectifs. Ce chapitre doit viser à \(1\) pouvoir reproduire vos expériementations, \(2\) partager/expliquer à d'autres l'usage des outils.

## VI. References

1. https://docs.spring.io/spring-framework/docs/current/javadoc-api/org/springframework/context/annotation/Conditional.html 
1. ref2

