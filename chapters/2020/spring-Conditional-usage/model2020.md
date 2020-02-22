---
layout: default
title : Comment est configuré un projet Spring via les @Conditional ?
date:   2020-01-11 22:00:00 +0100
---

---

> **Date de rendu finale : Mars 2020 au plus tard**
> - Respecter la structure pour que les chapitres soient bien indépendants
> - Remarques :
>>    - Les titres peuvent changer pour etre en adéquation avec votre étude.
>>    - De même il est possible de modifier la structure, celle qui est proposée ici est là pour vous aider.
>>    - Utiliser des références pour justifier votre argumentaire, vos choix etc.

---

**_janvier 2020_**

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


> 1. Commencez par formuler une question sur quelque chose que vous observez ou constatez ou encore une idée émergente. Attention pour répondre à cette question vous devrez être capable de quantifier vos réponses.
> 2. Préciser bien pourquoi cette question est intéressante de votre point de vue et éventuellement en quoi la question est plus générale que le contexte de votre projet \(ex: Choisir une libraire de code est un problème récurrent qui se pose très différemment cependant en fonction des objectifs\)
> Cette première étape nécessite beaucoup de réflexion pour se définir la bonne question afin de poser les bonnes bases pour la suit.

### Question Générale ...

#### Est ce que conditionnal a été adopté ? 

L'ajout de @Conditional en Spring 4 est la résultante d'une demande de la communauté.
Il est donc intéressant de voir, à postériori, si cette annotation a été adopté par la communauté.

#### Comment et pourquoi sont utilisés les différent type de @Conditional ?

Il existe une multitude de variantes à l'annotation @Conditional
- @ConditionalOnProperty
- @ConditionalOnWebApplication
- @ConditionalOnClass
- @ConditionalOnMissingBean
- @ConditionalOnExpression

Il est intéressant de voir pourquoi autant de variantes existent et dans quelle context elles sont utilisés.

#### Pourquoi @Profile a été remplacé par @Conditional , donc quelles sont leurs différences ?

En version 3.1, Spring a intégré une nouvelle annotation qui est @Profiles. Cette annotation est utilisée pour initialiser des Beans en fonction des variables d'environnements.
Après demande de la communauté, Spring a intégré en version 4 l'annotation @Conditional qui permet au développeurs de définir des stratégies pour ces conditions avec par exemple des conditions booléennes.

Il est donc intéressant de voir pourquoi cette nouvelle annotation a été introduite pour remplacé une annotation déjà existante.

#### Est-ce que @Conditional est vraiment utile et dans quel cas ? Comment configurer son projet sans l'utiliser ?



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

