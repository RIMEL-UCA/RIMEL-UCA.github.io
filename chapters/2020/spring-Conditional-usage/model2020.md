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

Spring est un framework qui permet de concevoir des applications Java plus simplement. Spring fournit un ensemble d'annotation pour aider le développeur dans la confection de son application et nous permet notamment de créer des composant et de les faire communiquer simplement.

Dans ce context, il est intéressant de voir comment créer est configurer les dit composant dans les projets Spring. 
Nous allons plus spécifiquement étudier comment configurer la création de beans en fonction de conditions au préalable défini.

Dans cette optique, Spring 3.1 a introduit l'annotation @Profile qui permet de créer des beans en fonction des profils qui sont en général défini dans les fichiers de configuration xml.
Dans Spring 4, l'annotation @Profile existe toujours cependant une nouvelle annotation a été introduite qui est @Conditional. Cette annotation permet, tout comme @Profile, de configurer les beans. 
Cependant, @Conditional permet de créer une configuration sans profile préalable, notamment basé sur des stratégies sous format booléennes. Cette nouvelle approche permet une plus grande liberté de création de stratégie, notamment, n'importe quelle statégie peut être utilisée tant qu'elle est testable par valeur booléenne.

Nous allons donc, dans notre étude, nous concentré sur cette nouvelle annotation @Conditional et quels sont les impacts de cette annotation sur les projets Spring.

![Figure 1: Logo UCA](../assets/model/UCAlogoQlarge.png){:height="50px" }


## II. Observations/General question


### Comment sont configurés les projets Spring à l'aide de l'annotation @Conditionnal

Nous nous sommes alors intéressé à ce changement effectué de @Profile vers @Conditional et notamment pourquoi @Profile a été remplacé par @Conditional.
D'après ce que nous avons vu dans nos recherches, ce changement a été posé par une demande de la part de la communauté, il nous parrait donc intéressant de voir quels sont les cas d'usage qui impliquent une telle modification.

Pour ce faire, nous nous sommes posé plusieurs sous-questions auxquels nous pourrons répondre à l'aide de metrics dont l'extraction est automatisable.

#### 1. Est-ce que le @Conditional à été adopté ?

L'annotation a été poussée par la communauté, comme vu précédemment. 
Il est donc, d'après nous, intéressant de savoir si cette dernière à été adoptée par la communauté.
En effet, ceci permettrais de savoir si cette nouvelle annotation répond bien aux attentes qui ont entraîné sa création.

Pour se faire, nous pouvons regarder dans les projets Github utilisant Spring le taux d'appartion de l'annotation @Conditional

#### 2. Comparaison de l’utilisation de @Profile et @Conditional

Après l'introduction de la nouvelle annotation, il est interessant de savoir si l'ancienne annotation @Profile est remplacée par @Conditional.

Nous pouvons donc, pour faire ceci, récupérer la liste des git diff dans chaque projet Spring utilisant l'une des deux annotations.
Nous aurons donc la liste des ajouts et retrait de chacune des annotations.

L'analyse sur les commits nous permet de faire ressortir l'information concernant l'évolution de l'annotation dans le temps.
En effet chaque commit est associés a une date. Avec l'outil PyDriller nous avons pu parcourrir l'ensemble des commit pour chaque projet. Pour chaque commit nous nous sommes intérrèssé a dénombrer chaque ajout dans le code des annotations @Conditional et @Profile. Les données résultantes sont ensuite trié en fonction des années.

#### 3. Comment sont utilisés les différentes variantes de @Conditional ?

Nous avons observer que l'annotation @Conditional n'a pas été seule. 
En effet, une multitude de variantes ont été introduites qui sont :
- @ConditionalOnProperty : @Conditional qui check si une propriété, qui par défaut est présente dans l'environnement a une valeur spécifique
- @ConditionalOnWebApplication : @Conditional qui est valide dans le cas où l'application courante est une application web. Cette application web peut notamment être de type spécifique, dont le mode servlet ou réactive.
- @ConditionalOnClass : @Conditional qui est valide uniquement si la classe spécifiée est présente dans le classpath.
- @ConditionalOnMissingBean: @Conditional qui est valide uniquement quand aucun autre bean n'est pas déjà instancié.
- @ConditionalOnExpression : @Conditional en basé sur les résultats SpEL (Spring Expression Language)

Il est donc intéressant de savoir comment la communauté Spring utilise ces annotations. 
Pour ce faire, nous allons regarder les occurrences dans les différents projets Spring de chaque variantes et voir dans quel contexte avec les raisons d'ajout/remplacement.

#### 4. Quels autres mécanismes peuvent remplacer @Conditional ?

Il existe d'autre mécanisme permettant de configurer son projet Spring. Par exemple, l'annotation @Profile toujours utilisé couvre une partie des fonctionnalités proposé par @Conditional. 

Notre approche est de comparer le nombre d'occurrence de l'annotation par rapport au nombre d'occurrence de l'annotation @Profile, @Value et @Resource, l'objectif étant de déterminer si d'autres mécanismes utilisant ces annotations permettent de remplacer @Conditional.


#### 5. Est-ce que @Conditional est vraiment utile et dans quels cas ? 

Comme nous avons dit précédemment l'annotation @Conditional a été réclamé par la communauté, il est donc intérrêssant de se demande si elle a été adopter. Afin de répondre, nous allons comparer le nombre de projets sans l'annotation @Contional par rapport au nombre de projets avec au moins une occurrence de l'annotation.

La deuxième partie de la question "dans quel cas" est plus complexe à traiter. En effet, nous n'avons pas pu proposer de metric dont l'extraction est automatisable afin de déterminer le context d'utilisation de l'annotation.

#### 6. Comparaison de l’utilisation de @Conditional dans le code et dans les tests ?


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
En addition à Pydriller, nous avons effectué de multiples codes python pour analyser le code des différents projets. Ces scripts nous permettent notament de chercher le nombre d'apparition des annotations dans le code.

## IV. Hypothesis & Experiences
Nos hypothèses correspondent aux questions posées dans la partie II.
* L'annotation @Conditional a été adopté par la communauté au détriment de l'annotation @Profile 
* Les variantes de l'annotation @Conditional sont utilisées pour configurer les projets Spring
* L'annotation @Conditional est préféré aux autres mécanisme de configurations 
### Première analyse 
Nous avons prit comme hypothèse de départ que l'annotation avait été adoptée.
Pour vérifier cette hypothèse nous avons récupérer des projets et regarder le nombre d'apparition dans ces projets.

Dans cette première expérimentation, nous avons prit 12 projets choisit au préalable. 
![Projet par commits](./assets/projectsByCommits.png)

Nous avons notamment, dans les plus remarquable, BroadleafCommerce qui est un framework de site e-commerce, Spring security qui fournit un service de sécurité pour Spring ou encore Spring IDE qui améliore Eclipse avec des outils pour créer des applications basées sur Spring.

Avec ce dataset, nous avons analysé le nombre d'occurrence de @Conditional. 
![Occurrences des Conditional](./assets/firstProjectsConditionalOccurences.png)

Nous nous sommes donc apercu que, au niveau du datase choisit, l'utilisation des annotations dépend du projet et donc du contexte.
Nous avons aussi analysé les différentes variantes dans le projet BroadLeafCommerce qui utilise le plus de Conditional de notre dataset.

![Occurrences des variantes de Conditional](./assets/firstProjectsConditionalVariants.png)

Nous nous sommes apperçu qu'une annotation en particulière dans ce projet est utilisée (@ConditionalOnTemplating).
Or, cette annotation n'est pas une annotation existante dans Spring mais une annotation créée par BroadLeafCommerce.
Cette fameuse annotation créée utilise une classe de mathcher qui correspond à l'ancienne implémentation via @Profile.
Nous avons donc déjà une première réponse à notre questionnement.

Cependant, ce dataset n'est pas assez grand pour en sortir des conclusions.

### Agrandissement du dataset
Suite à notre première analyse il nous a semblé claire que notre dataset n'était pas suffisament large. Nous nous sommes donc aidé de l'API github afin de récupérer un grand nombre projets réspectant les critères de recherche précédement cité. Nous avons récuperer l'ensemble des projets que nous étudions via deux requête à cette API.

Nous avons appliquer les filtres sur nos deux requêtes. Dans la première nous nous intérréssont à l'ensemble des projets github, alors que dans la second seulement au projets appartenant à l'organisation "Spring".

Finalement nous avons un dataset constitué de 214 projets.


### Experience
Notre objectif est de répondre au question posé dans la section II. Nous utiliserons donc les métrics présenté précédement afin d'y répondre.


## V. Result Analysis and Conclusion

### Adoption de l'annotation @Conditional
Nous avons analisé le nombre d'apparission de l'annotation @Conditional dans les 214 projets. Il est important de noter qu'il est possible que certain projet ne soit pas représentatif (tutoriels, ect.) ce qui peut produire un biai dans notre analyse. Toutefois, nous estimons que cela est négligable car nous travaillons sur un grand nombre de projets.
Le graphique ci dessous représente le nombre d'occurrence de l'annotation dans chaque projet. Le graphique est à l'echelle logarithmique afin d'avoir une meilleur représentations des résultats. On remarque que le projet spring-boot se distingue des autres projets. Cette observation n'est pas étonnante car spring-boot fournit des conditions prédéfinies afin que les développeurs n'aient pas à les implémenter eux-mêmes.
![](https://i.imgur.com/CZA2WKQ.png)

Finalement il ressort de cette analyse que uniquement 68 des 214 projets utilisent l'annotation soit environ 32%. Nous nous sommes demandé avant de conclure si il était possible que la date de création du projet soit un facteur pouvant altérer ces résultats. Nous avons donc éfféctué une deuxième analyse du dataset en ne prenant en compte cette fois que les projets créés en 2013 (année de la release de spring 4.0 et donc de l'integration de l'annotation). 
Nous avons constaté cette fois que 57 projets sur les 146 projets débutant en 2013 utilisaient l'annotation soit 39%. Ces résultats semble cohérent, en effet remplacer les anciens mécanismes de configuration par l'annotation @Conditonal peut-être un processus complexe, qui n'est éfféctué que lorsqu'il éxiste un réel besoin.  

Nous héstimons donc que l'annotation a été bien adopté par les développeurs puisque 32% des projets analysés et 39% pour les projets créés en 2013 l'utilisent.  

### Comparaison de l’utilisation de @Profile et @Conditional

Afin de comparer l'utilisation des deux annotations :
- Dans un premier temps nous avons comparé l'utilisations ou non des deux annotations dans chaque projets.
- Puis dans chaque commit nous avons comparer le nombre d'apparission de l'annotation pour voir si nous observons un changement de tendance de la part de l'annotation @Profile au moment de l'introduction de l'annotation @Conditional.

Le tableau ci-dessous nous montre le pourcentage de projets utilisants chaques annotations et le pourcentage de projets créés à partir de 2013. 

|          | Projets | Projets créés aprés 2013 | 
| -------- | -------- |-------- | 
|@Conditional|	32%| 39%|
| @Profile |	23%| 21%|

Nous pouvons voir que l'annotations @Conditional est préfèrée à @Profile, ces résultats ne sont pas étonnant puisque @Conditional est une version de @Profile avec l'ajout d'améliorations comme la possibilité de vérification par exepression booléenne. De plus nous remarquons une diminution de l'utilisation de @Profile contre une augmentation de l'utilisation de @Conditional dans les projets créés à partir de 2013.

A cette analyse peut s'ajouter l'analyse des commits pour chaques projets. Avec l'outil pyDriller nous avons explorer les commits pour chaques projets afin de comparer l'évolution des annotations dans le temps.
![Graphique en aire empilées, occurrence @Conditional dans les commits en fonctions des années](https://i.imgur.com/HrKDSpX.png)

### Utilisation des différentes variantes de @Conditional

Nous avons regarder dans chaque projets la proportions de chaque variante de l'annotation. L'histograme ci-dessous affiche le nombre d'occurrence de chaque annatotions dans chaque projets de notre dataset. 
![](https://i.imgur.com/qnquboL.png)
Le tableau ci-dessous nous montre le nombre de projets utilisant chaques annotations.

| Annotation | Nombre de projets |Pourcentages | 
| -------- | -------- |-------- | 
|@Conditional|	21|10%|
|@ConditionalOnProperty	|42|20%|
|@ConditionalOnMissingBean	|36|17%|
|@ConditionalOnAdmin	|0|0%
|@ConditionalOnBean|	32|15%
|@ConditionalOnClass|45|21%
|@ConditionalOnMissingClass|12|6%

Nous remarquons qu'un projets influence nos résultats (spring-boot pour la même raison qu'énoncé précédement). Nous remarquons que aucuns projets n'utilisent l'annoations @ConditionalOnAdmin. De même il y a trés peu d'occurrence de l'annotations @ConditionalOnMissingClass, cela peut être expliqué par le fait qu'il éxiste l'annotation inverse @ConditionalOnClass et quelle est actuellement la plus utilisé. 
Une autres valeurs intérréssantes est le nombre de projets utilisant l'anntation @Conditional de base, il n'y en a que 21. Cela montre que les variantes définie par spring-boot sont utilisées par les développeurs.

### Autres mécanismes pouvant remplacer @Conditional

Nous sommes intérressé aux autres mécanisme pouvant permettre le parametrage de projets spring. Nous avons donc comparé l'utilisation de @Conditional par rapport à @Profile, @Value et @Ressource. Afin de savoir qu'elles sont les moyens répendu pour configurer un projet spring. Nous avons notamment éfféctué plusieur recherche dans l'outil google trend. Nous avons vue que le sujet "yaml" est souvent associé a cette recherche.  Nous avons donc cherché dans notre dataset toutes les occurrences pour chacune de ces annotations et le nombre de fichier yaml dans les projets. 

|  | Nombre de projet | Pourcentages
| -------- | -------- | -------- | 
|@Conditional|	68| 32%|
|@Profile	|48| 22%|
|@Value	|123| 58%|
|@Resource	|52| 24%|
|Yaml|	97| 45%|

Nous obtenons les résultats présentés dans le tableau ci-dessus. Nous remarquons que l'annotation @value est utilisé dans environ la moitié des projets. Cette annotation permet d'injécter des valeur dans les champs des beans géré par spring.
nous nous plaçons dans les projets ne faisant aucune référence a @Conditional et nous regardont si ces projets utilisent @Resource, @Value, @Profile ou des fichier yaml. L'objectif de cette approche est de déterminer une corélation entre l'absense de l'annotation @Conditional et la présence d'une autre annotation ou de fichier yaml.

Dans un premier temps nous remarquons que 20% des projets n'utilisent aucun de ces mécanismes. Soit ils utilisent d'autres mécanismes de configuration, soit le projet n'a pas besoin de configuration.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg .tg-lboi{border-color:inherit;text-align:left;vertical-align:middle}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-0pky">Annotations</th>
    <th class="tg-0pky" colspan="2">Sans @Conditional</th>
    <th class="tg-0pky" colspan="2">Avec @Conditional</th>
  </tr>
  <tr>
    <td class="tg-lboi"></td>
    <td class="tg-lboi">Nombres projets</td>
    <td class="tg-lboi">Pourcentages</td>
    <td class="tg-lboi">Nombres projets</td>
    <td class="tg-lboi">Pourcentages</td>
  </tr>
  <tr>
    <td class="tg-0pky">@Profile</td>
    <td class="tg-0pky">23</td>
    <td class="tg-0pky">11%</td>
    <td class="tg-0pky">25</td>
    <td class="tg-0pky">12%</td>
  </tr>
  <tr>
    <td class="tg-lboi">@Value</td>
    <td class="tg-lboi">67</td>
    <td class="tg-lboi">31%</td>
    <td class="tg-lboi">56</td>
    <td class="tg-lboi">27%</td>
  </tr>
  <tr>
    <td class="tg-lboi">@Ressource</td>
    <td class="tg-lboi">30</td>
    <td class="tg-lboi">14%</td>
    <td class="tg-lboi">22</td>
    <td class="tg-lboi">10%</td>
  </tr>
  <tr>
    <td class="tg-lboi">Yaml</td>
    <td class="tg-lboi">52</td>
    <td class="tg-lboi">24%</td>
    <td class="tg-lboi">45</td>
    <td class="tg-lboi">21%</td>
  </tr>
</table>

Le tableaux ci-dessus représente le nombre de projets ayant au moins une occurrences de chaque annotations ou au moins un fichier yaml, qui utilise l'annotation @Conditional. Nous remarquons que @Profile est utilisé dans moins de projets sans @Conditonal, alors que tous les autres mécanismes sont utilisés dans plus de projet lorsqu'il n'y a pas de @Conditional.
Ces résultats ne nous permettent pas d'en sortir une correlations.


### L'annotation @Conditional est-elle vraiment utile ?

Au vue des éléments vue précédement nous pouvons répondre positivement a cette question. En effet nous avons observé que l'annotation a été réclamé par la communauté ainsi qu'environ 32% des projets de notre dataset l'utilisent au moins une fois. 

Nous avons cependant certaines pistes d'amélioration pour l'évenir au niveau de nos métriques. En effet, nous nous sommes remarqué lors de l'analyse des git diff que certains commantaire pouvaient impacter nos statisque. De plus, une [recherche en fonction des conditional dans les fichiers de tests](https://regex101.com/r/DTZRmB/1) pourrait être une piste interessante pour les études futur.

## VI. References

1. https://docs.spring.io/spring-framework/docs/current/javadoc-api/overview-summary.html
2. https://docs.spring.io/spring-framework/docs/current/javadoc-api/org/springframework/context/annotation/Conditional.html 
3. https://github.com
4. https://blog.ippon.fr/2014/01/07/conditional-de-spring-4-au-coeur-de-lauto-configuration-de-spring-boot/
5. https://javapapers.com/spring/spring-conditional-annotation/ 
6. https://github.com/BroadleafCommerce/BroadleafCommerce
7. https://github.com/spring-projects/spring-security
8. https://github.com/spring-projects/spring-ide


