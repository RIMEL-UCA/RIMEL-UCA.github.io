---
layout: default
title : Modèle de chapitre pour 2020
date:   2020-01-03 22:00:00 +0100
---

## Auteurs

Nous sommes quatre étudiants en dernière année à Polytech Nice Sophia en architecture logicielle :

* Laura Lopez &lt;laura.lopez1@etu.unice.fr&gt;
* Alexis Gardin &lt;alexis.gardin@etu.unice.fr&gt;
* Hugo Croenne &lt;hugo.croenne@etu.unice.fr&gt;
* Mathieu Paillart &lt;mathieu.paillart@etu.unice.fr&gt;

## I. Contexte de la récherche

Spring est un framework open-source qui facilite la mise en place et le développement d'applications Java. Son fonctionnement se base sur un système d'annotations. 

L'annotation de paramétrage *@Conditional* a été introduite dans la version *Spring 5.2.3* afin de répoondre à des problématiques de condition de chargement de classe en fonction d'un contexte. 

bullshit sur la variabilité.
les nouvelles/anciennes annotations 
spring
contexte : 

![Figure 1: Logo UCA](../assets/model/UCAlogoQlarge.png)


## II. Observation générale

1. Commencez par formuler une question sur quelque chose que vous observez ou constatez ou encore une idée émergente. Attention pour répondre à cette question vous devrez être capable de quantifier vos réponses.
2. Préciser bien pourquoi cette question est intéressante de votre point de vue et éventuellement en quoi la question est plus générale que le contexte de votre projet \(ex: Choisir une libraire de code est un problème récurrent qui se pose très différemment cependant en fonction des objectifs\)
Cette première étape nécessite beaucoup de réflexion pour se définir la bonne question afin de poser les bonnes bases pour la suite.


Nous avions tous au sein de l'équipe déjà utilisé Spring lors de projets Polytech, mais aucun de nous n'avais utilisé l'annotation *Conditional* et nous étions peu qualifiés sur les outils de paramétrage de Spring. La première question émergeant de nos réflexions était donc :

**Comment les outils de paramètrage de Spring sont t-il utilisés et à quelle fréquence ?**

Cette question d'ordre général a été notre première étape afin d'orienter nos recherches vers d'autres questions plus précises. Au fil de nos découvertes, nous avons pu cibler un ensemble de nouvelles questions qu'il nous semblait pertinent et intéressant d'étudier.

### Comment l'annotation @Conditional est-elle utilisée ? 

N'étant pas familier avec l'annotation, il était important pour nous dans un premier temps de rassembler le plus d'informations possible sur *@Conditional*. Pour cela, nous nous sommes concentrés sur sa présence au sein des fichiers, des projets et sur sa fréquence d'apparition au sein d'un même fichier.
L'annotation de paramétrage que l'on appelle *@Conditional* possède plusieurs déclinaisons : *@ConditionalOnMissingBean*, 

Nous souhaitons également nous intéresser à la fréquence d'usage de chacune de ces annotations au sein de projets afin d'en déduire qu'elles sont les annotations les plus utilisées et dans quel contexte d'usage. 

Apportant un ensemble de questions sous-jacentes
-> dans quels fichiers ? -> configuration
-> dans quels projets ? -> frameworks
-> combien de fois par fichiers ?  -> rarement seule
-> quelle déclinaison de l'annotation est la plus utilisée ?  -> on missing bean

### Comment et à quelle fréquence est-elle testée ? 

Il nous paraissait d'autre part intéressant d'étudier la fréquence de test de l'annotation étudiée. En effet, cela nous permettrait de nous renseigner sur la facilité de tests de l'annotation, mais également potentiellement en déduire que son utilisation est une bonne pratique et n'affecte pas la couverture de tests d'un projet. D'autre part, nous trouvions également intéressant de savoir si celle-ci était utilisée au sein de fichiers tests dans le but de tester d'autres fonctionnalités afin d'avoir une vue d'ensemble plus large de son utilisation. 

### Quelle annotation de configuration est la plus utilisée parmi les existantes (Conditional, Profile, Value, Ressources) ? 

Nous avons pu remarquer lors de nos recherches, l'existance de plusieurs annotations de configuration autres que *@Conditional*. Nous avons tout de suite pensé qu'il serait intéressant de la mettre en corrélation avec ses équivalents. Les autres annotations sont : 
1
- *@Profile* qui est le prédécesseur de *@Conditional*, cette annotation est apparue dans *Spring 3.1* et est principalement utilisée afin de gérer des profils définit selon des fichiers de configuration (par exemple un fichier *".properties*). Un exemple d'utilisation serait la mise en place de profils de développement (dev), production (prod) et test. Le type de *Bean* chargé pourrait ensuite dépendre du profil sélectionné. 
- *@Ressource* 
- *@Value* cette annotation a pour but d'assigner des valeurs à des variables et des paramètres de méthodes.

Il serait intéressant dans le cadre de cette étude d'étudier la fréquence d'utilisation de chacune de ces annotations, de constater dans quel type de projet elles apparaissent le plus souvent et de constater comment elles sont utilisées au sein de ces projets. 

### Comment est utilisée l'annotation @Value ? 

Lors de nos recherches, une seconde annotation a particulièrement attiré notre attention, il s'agit de l'annotation *@Value*. Celle-ci peut être utilisée à la fois au niveau de méthodes, de paramètres de méthodes, de variables de classes. 

Sa modularité et la diversité de sa portée nous a amenés à nous demander si son mode d'utilisation se confirmait à du *tangling* (signifiant ) ou du *spreading* (signifiant ). 

## III. Récolte d'informations

Afin de 
la doc de spring très important
doc des annotations
le paramétrage
[Mining Implicit Design Templates for Actionable Code Reuse](http://linyun.info/micode/micode.pdf)  
[Are Developers Aware of the Architectural Impact of Their Change](http://www0.cs.ucl.ac.uk/staff/j.krinke/publications/ase17.pdf)  
[Titan: A Toolset That Connects Software Architecture with Quality Analysis](https://www.cs.drexel.edu/~lx52/LuXiao/papers/FSE-TD-14.pdf)  

## IV. Hypothèses émises

### Hypothèse n°1 : L'annotation @Conditional est utilisée dans des fichiers de configuration. 

### Hypothèse n°2 : L'annotation @Conditional est particulièrement utilisée dans les frameworks. 

### Hypothèse n°3 : L'annotation Conditional est souvent testée lorsqu'elle est utilisée. 

reprendre du premier rapport -> on utilise pas @Conditional pour tester @Conditional -> pas démontrable 
résultats pas assez simple à réaliser

### Hypothèse n°4 : La répartition des annotations de paramétrage n'est pas homogène.

### Hypothèse n°5 : L'annotation @Value tangling ou spreading

1. Il s'agit ici d'énoncer sous forme d' hypothèses ce que vous allez chercher à démontrer. Vous devez définir vos hypothèses de façon à pouvoir les _mesurer facilement._ Bien sûr, votre hypothèse devrait être construite de manière à v_ous aider à répondre à votre question initiale_.Explicitez ces différents points.

Le taux de @Value utiliser dans des if
La je vais, par projet, regarder la répartition des Annotation @Conditional et Autre pour essayer de classifier les libs, des projets concrêt
pour voir si on peut dire
Sur les 14 projet analyser le taux de @Conditional était Superieur pour les libs et @Value pour les projet concret

Tout d'abord, nous avons cherché à savoir quelle annotation @Conditional était la plus utilisée et où. 

Ensuite, nous avons souhaité comparer sa fréquence avec d'autres annotations de configuration.

Nous nous sommes également intéressés aux tests potentiellement effectué sur l'annotation @Conditional afin de voir si celle-ci était souvent testée.

Nous avons ensuite cherché les fichiers au sein desquels l'annotation est utilisée = nom de fichier.

Ensuite création d'un gros dataset fiable.

Pour finir, comparaison de @Value/@Profile/@Conditional/@Ressource


## V. Ressources utilisées

Afin de mener à bien notre étude nous avons eu besoin de constituer un dataset de projets utilisant Spring. Pour cela, nous nous sommes servis de la plateforme d'hébergement et de développement, basée sur le gestionnaire de version Git : Github. La plateforme contient des milions de projets qui nous ont permis d'établir un dataset complet et exhaustif. 

[lister en fonction du type de projet]
Les projets utilisés pour cette étude sont les suivants:  
'naver/pinpoint',
 'permazen/permazen',
 'jeremylong/DependencyCheck',
 'micronaut-projects/micronaut-core',
 'zkoss/zk',
 'spring-projects/spring-boot',
 'typetools/checker-framework',
 'debezium/debezium',
 'spring-io/sagan',
 'eclipse/hawkbit',
 'zalando/riptide',
 'spring-projects/spring-framework',
 'eugenp/tutorials',
 'INRIA/spoon',
 'spring-cloud-projects',
 'spring-boot projects'

Afin d'optimiser nos recherches, nous les avons automatisées à l'aide de scripts en Python. Le développement python était effectué sur l'outil Jupyter Notebook. 

[ parler de javalang également ]

### Jupyter Notebook 

Nous avons choisi cette application car elle apportait plusieurs avantages :

* visualisation des résultats
* mémoire en cache des exécutions des scripts, ce qui a permi d'accélérer le développement car les scripts n'ont pas besoin d'être relancés à chaque fois
* scripts partagés avec Jupyter Notebook Collaboration au sein desquels le stockage des projets clonés se fait directement dans le cloud (Google Drive). 

## VI. Expériences réalisées

2. Test de l’hypothèse par l’expérimentation. 1. Vos tests d’expérimentations permettent de vérifier si vos hypothèses sont vraies ou fausses. 2. Il est possible que vous deviez répéter vos expérimentations pour vous assurer que les premiers résultats ne sont pas seulement un accident.
3. Explicitez bien les outils utilisés et comment.
4. Justifiez vos choix.

- 
-


## VII. Analyse des résultats et validation des hypothèses



## VIII. Conclusion

1. Analyse des résultats & construction d’une conclusion : Une fois votre expérience terminée, vous récupérez vos mesures et vous les analysez pour voir si votre hypothèse tient la route. 


à ne pas oublier : le biais d'utiliser uniquement github 




## IX. Références

[Mining Implicit Design Templates for Actionable Code Reuse](http://linyun.info/micode/micode.pdf)  
[Are Developers Aware of the Architectural Impact of Their Change](http://www0.cs.ucl.ac.uk/staff/j.krinke/publications/ase17.pdf)  
[Titan: A Toolset That Connects Software Architecture with Quality Analysis](https://www.cs.drexel.edu/~lx52/LuXiao/papers/FSE-TD-14.pdf)  
[https://github.com/naver/pinpoint](https://github.com/naver/pinpoint)  
[https://github.com/permazen/permazen](https://github.com/permazen/permazen)  
[https://github.com/jeremylong/DependencyCheck](https://github.com/jeremylong/DependencyCheck)  
[https://github.com/micronaut-projects/micronaut-core](https://github.com/micronaut-projects/micronaut-core)  
[https://github.com/zkoss/zk](https://github.com/zkoss/zk)  
[https://github.com/spring-projects/spring-boot](https://github.com/spring-projects/spring-boot)  
[https://github.com/typetools/checker-framework](https://github.com/typetools/checker-framework)  
[https://github.com/debezium/debezium](https://github.com/debezium/debezium)  
[https://github.com/spring-io/sagan](https://github.com/spring-io/sagan)  
[https://github.com/eclipse/hawkbit](https://github.com/eclipse/hawkbit)  
[https://github.com/zalando/riptide](https://github.com/zalando/riptide)  
[https://github.com/spring-projects/spring-framework](https://github.com/spring-projects/spring-framework)  
[https://github.com/eugenp/tutorials](https://github.com/eugenp/tutorials)
[https://github.com/INRIA/spoon](https://github.com/INRIA/spoon)
