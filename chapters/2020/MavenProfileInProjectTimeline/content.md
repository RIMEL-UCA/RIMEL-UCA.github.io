# Quels problèmes résout l’utilisation de profils Maven dans les logiciels Open Source ?

## Auteurs

Nous sommes trois étudiants en dernière année à Polytech Nice Sophia Antipolis, dans la spécialité Architecture Logicielle :

* FORAY Théo &lt;theo.foray@etu.unice.fr&gt;
* SEGURA Alexis &lt;alexis.segura@etu.unice.fr&gt;
* STROBBE Nathan &lt;nathan.strobbe@etu.unice.fr&gt;


## I. Contexte de recherche

Le sujet de notre recherche en rétro-ingénierie se porte sur les profils Maven. 
Maven est un outil d'automatisation et de configuration de production de projets Java, construit sous forme de document XML. La configuration passe notamment par l'utilisation d'un outil propre à Maven appelé _profile_. Un profil sert au moment du _build_ du projet, à paramétrer certains éléments par exemple à spécifier l'utilisation de nouveaux plugins pour ce profil, de nouvelles propriétés ou encore de nouvelles dépendances. Ce mécanisme facilite entre autres l'utilisation d'environnements de test, de développement et de production.

Afin d'avoir une base de projets à étudier, nous avons effectuer nos recherches sur la plateforme GitHub où sont disponibles de nombreux projets Java qui utilisent la technologie Maven (Cf. <a href="#III-Collecte-d’informations">III. Collecte d’informations</a>).

Ce contexte de recherche est particulièrement intéressant pour nous, car nous utilisons majoritairement des technologies de l'éco-système Java, dont Maven. Nous avons donc déjà été amené à utiliser les profils Maven, et avons donc certaines intuitions quant à la question principale de notre sujet. Cela rend donc cette recherche encore plus pertinente afin de voir si ces intuitions se confirment ou non dans des projets Open Source, potentiellement de bien plus grande taille que nos projets. 

De plus, nous arrivons à une période charnière où la méthode DevOps est de plus en plus populaire car elle permet d'avoir un meilleur suivi de projet, et de plus facilement maintenir la stabilité des projets. Or, certains outils utilisés pour appliquer cette méthode DevOps poussent encore plus loin l'idée de configuration et d'automatisation que ce que propose Maven et ses profils. Il nous paraissait donc intéressant d'étudier ces différents points à une échelle plus importante.

## II. Observations et problématique

Au cours de notre formation, nous avons eu l'occasion de rencontrer des Maven Profiles sur plusieurs projets, lors de stage ou à l'école. Ces profils sont utilisés dans des contextes et à des fins variées. Certains permettent de configurer le _build_ d'un projet pour l'utiliser dans un environnement de développement, d'autres pour générer du code ou encore lancer des tests. Ce constat nous a amené à réfléchir sur le *scope* d'un profil, sur sa responsabilité. La mécanique de profils Maven permet de faire beaucoup de choses et nous n'avons trouvé de *guidelines* définissant concrêtement les cas et domaines d'utilisation. <br/>
Ainsi nous nous sommes posés la question : **Quels sont les principaux problèmes résolus par l’utilisation des Maven Profiles ?** Une étude empirique sur un ensemble de projets _Open Source_ pourrait nous permettrait peut être de trouver un ensemble de cas d'utilisation globaux.

Pour conduire cette étude, nous avons découpé la question en trois sous interrogations. 

### Caractérisation des profils
En parcourant des projets Maven, nous avons pu voir la diversité qui existait dans les profils, certains contextuels au projet mais d'autres sont plus communs. Les noms qui viennent tout de suite en tête sont : "test", "prod", "dev" par exemple. Nous avons aussi remarqué par expérience que des profils contextuels à des projets pouvaient utiliser des plugins/dépendances très répandues. (Exemple ?)
En partant de ce constat, nous pensons qu'il serait intéressant de répondre à la question : **Quels sont les différents types de Maven profiles dans les projets Open Source ?**. Ceci nous permettrait de faire inventaire des profils les plus utilisés et de les associer aux plugins/dépendances/propriétés qu'ils configurent.
Cette question constitue la première étape de notre recherche.

### Place des profils dans l'historique des projets

Les profils Maven font parti de la configuration d'un projet. De ce fait, ce n'est pas la partie qui est la plus susceptible d'être modifiée dans un code source. Nous nous sommes donc demandé **Au cours de quels événements les développeurs implémentent-ils des Maven Profiles dans un projet?**

À noter, que nous considérons ici qu'un événement peut être un ajout, modification, suppression de LoC sur un *profile* lors d'un commit.

### Contexte technologique

Enfin, nous avons eu une intuition sur les liens entre les profils Maven et la stack technologique des projets dans lesquels ils sont utilisés. À une époque, les projets java étaient souvent configuré au moment du _build_. Entrainant ainsi l'utilisation des Maven profils. Aujourd'hui, la dockerisation des projets tend à changer les pratiques. Nous avons plus le réflexe de construire une seule image du projet et de venir le configurer au moment de l'exécution via des variables d'environnement ou des configurations sur des serveurs à part. Si ces observations sont vraies alors cela signifie que la dockerisation et les profils dans le pom ont un lien. De même, la mouvance DevOps entraine la mise en place de CI sur de nombreux projets. Les profils peuvent être très utile dans une pipeline d'intégration continu car ils permettent de lancer des phases spécifiques comme les tests, la génération de la javadoc ou encore une analyse de code. <br/>
Cette observation, couplée à notre intuition de départ nous amène à nous poser la question suivante : **L'environnement technique d'un projet influence-t-il l’utilisation des Maven Profiles ?**.


En combinant ces trois questions, nous pensons obtenir des informations de dimensions très différentes et ainsi proposer une réponse à la question générale. La première interrogation concerne la caractérisation des profils, la seconde traite de leur cycle de vie au sein d'un projet et la dernière de la relation entre les profils et le contexte technique d'un projet. 

<!--1. Commencez par formuler une question sur quelque chose que vous observez ou constatez ou encore une idée émergente. Attention pour répondre à cette question vous devrez être capable de quantifier vos réponses.
2. Préciser bien pourquoi cette question est intéressante de votre point de vue et éventuellement en quoi la question est plus générale que le contexte de votre projet \(ex: Choisir une libraire de code est un problème récurrent qui se pose très différemment cependant en fonction des objectifs\)

Cette première étape nécessite beaucoup de réflexion pour se définir la bonne question afin de poser les bonnes bases pour la suit. -->

## III. Collecte d'informations

<!--NATE
Préciser vos zones de recherches en fonction de votre projet,
1. les articles ou documents utiles à votre projet)
2. les outils-->

La zone de recherche se situe dans un ensemble de projet open source. Nous avons décidé de choisir GitHub qui répertorie un grand nombre de projets open source, facile à utiliser et à chercher des dépôts, fichiers spécifiques. GitHub propose également une API permettant de récolter facilement des informations sur des dépôts de manière plus poussée.

### Recherche de POM sur GitHub

Notre objectif est d'obtenir un *dataset* assez conséquent pour pouvoir essayer d'atteindre une analyse complète de nos questions. Il faut donc rechercher sur GitHub des fichiers `pom.xml` contenant une balise `<profile>`. Cette recherche peut s'effectuer manuellement via l'interface de GitHub : 
https://github.com/search?l=Maven+POM&q=profile+filename%3Apom.xml&type=Code
Cette requête peut également se faire de manière automatique grâce à l'API de GitHub et pourra donc être intégré dans un script automatique de récupération de ces POM.

Cette recherche nous donne donc les résultats attendus. Cependant, l'API de recherche GitHub limite la réponse à 1000 résultats dû à la grande quantité de données que pourrait nous renvoyer l'API. Ce qui reste assez bloquant car ne donnant pas un *dataset* assez conséquent.
Pour pallier à cette limite, nous avons expérimenté une manière d'obtenir plus de données (Cf. <a href="#Méthode-de-recherche-sur-GitHub-pour-obtenir-beaucoup-de-résultats">IV. Hypothèses et Expérimentations</a>).

Ce *dataset* résultant sera utilisé pour répondre à notre première sous-question.

### Réduction de l'espace de recherche

Afin de répondre aux sous-questions suivantes, nous avons décidé de nous focaliser sur des projets ayant plus de commits (Cf. <a href="#Recherche-de-statistiques-pour-filtrer-le-dataset">IV. Hypothèses et Expérimentations</a>) en se basant sur notre *dataset* de base. Cela nous a permis de réduire notre espace de recherche pour avoir des résultats plus pertinents relatifs à nos questions.

## IV. Hypothèses et Expérimentations

### Méthode de recherche sur GitHub pour obtenir beaucoup de résultats

Comme indiqué précédemment, l'API de GitHub limite le nombre de résultats à 1000 réponses par requête. Nous avons donc expérimenté une méthode permettant d'avoir plus de résultats. Celle-ci consiste à prendre un nouveau critère pour la recherche, l'appliquer et le faire varier afin d'obtenir plus de résultats. Cette variation est caractérisé par un interval. Nous avons choisi l'interval de taille d'un fichier car les fichiers ne vont pas forcément beaucoup changés en termes de taille. 
Ainsi, lors de notre recherche, nous allons prendre un intervalle de taille de fichiers et effectuer la recherche. Si le nombre de résultats de la recherche est supérieur à 1000, nous allons effectuer une dichotomie et ainsi couper en deux l'intervalle et refaire une requête. Si le nombre de résultats est inférieur à 1000, nous pouvons aggréger les résultats et continuer sur les intervalles restants. Ainsi nous obtenons moins de résultats pour des petits intervalles mais finalement tous les résultats de ces petits intervalles vont donner un grand nombre de résultats.

### Quels sont les principaux problèmes résolus par l’utilisation des Maven Profiles ?

#### Hypothèse

Une première hypothèse que nous établissons est : Les noms des Maven Profile et leur contenu vont permettre de caractériser et de catégoriser un profil afin de pouvoir étudier son usage.

#### Expérimentation

Afin de répondre à cette première sous-question, nous avons constitué une base contenant 18 431 Maven profiles pour les caractériser à l'aide de différents critères. Le premier critère que nous utilisons afin de discriminer les *profiles* est leur nom, ou plus précisemment leur *id*. Étant donné le nombre important de *profiles* que nous analysons et la large variété de ces noms, nous les avons regroupés par catégorie. Nous avons fait le choix d'établir des ensembles disjoints pour ne pas avoir d'ambigüité, c'est-à-dire que si un *profile* est dans une catégorie, il ne peut pas figurer dans une autre.

Ensuite, nous avons examiné le contenu des *profiles* de ces catégories pour comprendre lesquels étaient le plus configuré, et comment. Nous appelons ici configuration, le fait de spécifier des ```properties```, ou d'utiliser des ```dependencies``` ou des ```plugins```. 

#### Outils

- API GitHub : recherche des POMs (Cf. <a href="#Méthode-de-recherche-sur-GitHub-pour-obtenir-beaucoup-de-résultats">Méthode de recherche sur GitHub</a>)
- Téléchargement des sources : pour pallier à la limite de l'API de GitHub, nous avons utilisé un autre outil externe pour télécharger les fichiers sources de GitHub (cf. <a href="#VIII-Annexes">VIII. Annexes</a>)
- Parser XML : pour détecter les balises que nous recherchons dans les POMs, nous utilisons, à l'aide d'une dépendance Maven, une structure de données fournis par le w3c
- Spring Data / Neo4j : pour le stockage de nos données dans la base Neo4j, nous avons utilisé Spring Data qui facilite le stockage et fourni une abstraction du modèle de données pour manipuler les nœuds et les relations.


### Recherche de statistiques pour filtrer le dataset

Afin de réduire l'espace de recherche pour notre premier *dataset*, nous avons décidé d'évaluer certains critères sur les dépôts git afin de déterminer un « bon » filtre de données pour notre *dataset*. Pour cela, nous avons choisis plusieurs critères et récupérer des métriques sur les dépôts pour pouvoir faire des statistiques sur ces critères et en choisir un « bon ». Nous avons choisis ces critères : 
- Nombre de commits
- Nombre de contributeurs au dépôt
- Nombre de releases GitHub

Afin de pouvoir facilement récupérer des statistiques, nous avons utilisé un script en R.

| Critère       | Min | Q1 | Médiane | Moyenne | Q3 | Max   | Écart-type |
|---------------|:---:|:--:|:-------:|:-------:|:--:|:-----:|:----------:|
| Contributeurs |  0  | 1  |    1    | 5.05    | 2  | 100   | 15.87988   |
| Commits       |  0  | 2  |    9    | 477.3   | 48 | 37501 | 2766.255   |
| Releases      |  0  | 0  |    0    | 0.5469  | 0  | 100   | 4.527333   |

Le critère intéressant ici est sur le nombre de commits. En effet, on remarque que la notion de releases beaucoup utilisé sur notre dataset et donc peu pertinent pour nous. Et le nombre de contributeurs varie très peu avec seulement en moyenne 5 contributeurs par projets. Nous allons donc nous concentrer sur le nombre de commits dans un projet.

Cependant, ces statistiques aggrégés ne suffisent pas à correctement filtrer les dépôts par leurs nombre de commits : À partir de quelle limite on peut raisonnablement choisir un dépôt ? Pour répondre à cela, nous avons décidé de calculer le décile sur ce critère : 

| 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70%  | 80%   | 90% | 100%  |
|:--:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:-----:|:---:|:-----:|
| 0  |  0  |  0  |  1  |  3  |  7  | 18  | 49.1 | 188.4 | 407 | 37501 |

Ainsi, nous pouvons voir que 90% de notre dataset comporte plus de 407 commits. Ce qu'ils nous parez raisonnable pour réduire notre dataset, en sachant que celui et à plus de 5000 POM, nous allons donc obtenir 500 poms pour notre dataset réduit.


### Au cours de quels événements les développeurs implémentent-ils des Maven Profiles dans un projet ?

#### Hypothèse

Notre hypothèse de travail pour répondre à cette question est que les *profiles* sont créés ou modifiés lors de situations charnières d'un projet. Ce que nous appelons « situations charnières » correspond par exemple à des commits avec des *tags* de release, ou encore des modifications sur des branches de type *hotfix*. 

#### Expérimentation

Pour vérifier notre hypothèse, nous parcourons les commits d'un dépôt et identifions ceux où les *profiles* sont impactés. Les éléments relevés sont le message de commit et le nom de la branche. De plus, nous inspectons le commit pour récupérer le tag. 

Lors de cette expérimentation, nous avons identifié plusieurs biais non négligeables sur la construction du dataset : 
- Nous nous sommes rendu compte de la difficulté d'analyser les commits touchant les *profiles*. En effet, il est assez complexe de determiner à partir d'un *diff*, ce qui a été modifié dans un *profile* particulier. 
- Nous nous sommes rendu compte que nous n'arrivions à récupérer que des commits provenant de la branche par défaut du repository. 
- Nous nous sommes rendu compte que sur de nombreux commits, nous n'arrivions à récupérer le diff et donc à savoir si un *profile* était impacté.
- Nous nous sommes rendu compte que n'arrivions pas a détecter les commits qui sont des merges (dans le code oui, mais dans la recherche non).
Notre dataset actuel est donc beaucoup plus petit que ce qu'il devrait être. Par exemple, sur certains poms nous n'avons identifié d'événement de modification sur les *profiles* alors qu'il y en a forcément.

#### Outils
- Spring Data / Neo4j
- jgit : bibliothèque nous fournissant les moyens de manipuler l'historique des commits d'un dépôt git et de récupérer les *diffs* de ces commits, ainsi que les autres éléments tels que les *tags* ou les *branchs*.

### L'environnement technique d'un projet influence-t-il l’utilisation des Maven Profiles ?

#### Hypothèse

Du fait de l'utilisation de certains outils externes de différents types, nous pensons que certains vont favoriser ou défavoriser l'utilisation de *profiles*. Par exemple, avec un *stack* technique utilisant Docker où on peut configurer des images, ou la configuration des *beans* dans Spring, cela va défavoriser l'utilisation des Maven *profiles*.

#### Expérimentation

Afin d'évaluer cette *stack* technique, nous parcourons, sur le dernier commit, le nom de tous les fichiers du projet, pour y trouver des noms caractéristiques de certaines technologies notamment "Dockerfile", "docker-compose", ou encore certaines balises qui correspondent à des noms d'ORM dans le pom du projet.

#### Outils

- Réutilisation des outils précédents

<!--1. Il s'agit ici d'énoncer sous forme d' hypothèses ce que vous allez chercher à démontrer. Vous devez définir vos hypothèses de façon à pouvoir les _mesurer facilement._ Bien sûr, votre hypothèse devrait être construite de manière à vous aider à répondre à votre question _initiale_. Explicitez ces différents points.
2. Test de l’hypothèse par l’expérimentation. 
   1. Vos tests d’expérimentations permettent de vérifier si vos hypothèses sont vraies ou fausses.
   2. Il est possible que vous deviez répéter vos expérimentations pour vous assurer que les premiers résultats ne sont pas seulement un accident.
3. Explicitez bien les outils utilisés et comment.
4. Justifiez vos choix -->

## V. Analyse des résultats et conclusion

### Question 1 : Quels sont les différents types de Maven profiles dans les projets Open Source ?

Le diagramme ci-dessous montre le résultat de nos expérimentations, en pourcentage par rapport au 18 431 profiles stockés, leur répartition selon s'ils contiennent les différents termes dans leur nom.

![profiles-categories.png](https://raw.githubusercontent.com/RIMEL-UCA/RIMEL-UCA.github.io/maven-profile-timeline/chapters/2020/assets/MavenProfile_Timeline_profiles-categories.png)

Nous avons choisi d'établir ces catégories en fonction des noms majoritaires des profiles. En effet, ces derniers peuvent avoir des noms très différents, mais certains ne sont en fait que très utilisés peu comparé au nombre total de profiles. Les noms qui ressortaient alors le plus nous ont servi de catégories afin de regrouper par exemple "dev" et "development" ou encore "include-java-sources" et "include-binaries". On peut le constater sur le diagramme original de répartition des noms : 

![profiles-names.png](https://raw.githubusercontent.com/RIMEL-UCA/RIMEL-UCA.github.io/maven-profile-timeline/chapters/2020/assets/MavenProfile_Timeline_Profile-names.png)

Parmi ces catégories, nous remarquons que les différentes phases de développement sont très présentes avec les catégories ```dev```, ```release```, ```test```, et ```prod```. Ces profiles servent donc à configurer les build en fonction du type de phase de développement dans lequel il est utilisé. Ensuite, on remarque que les profiles sont beaucoup utilisés afin d'ajuster le build en fonction de la version de Maven voulait être utilisée. En effet, avec la catégorie ```maven```, nous avons pu regrouper des profiles nommés la plupart du temps "maven3" ou "maven2". Enfin la catégorie la plus importante, ```include``` sert elle à ajouter certaines parties très spécifiques du projet, notamment des bibliothèques, des mappings xml ou encore des fichiers de configuration.

La deuxième expérimentation faite pour répondre à cette question nous a fournis le graphe ci-dessous, qui permet d'identifier les différentes utilisations des contenus des profiles :

![category-config.png](https://raw.githubusercontent.com/RIMEL-UCA/RIMEL-UCA.github.io/maven-profile-timeline/chapters/2020/assets/MavenProfile_Timeline_category-config.png)

On remarque que dans toutes les catégories, sauf ```release```, les ```properties``` sont le moyen de configuration le plus utilisé. Ce type de configuration permet de modifier des éléments internes au projet tandis que les ```dependencies``` et ```plugins``` vont permettre, dans le cas de l'utilisation de ces profiles d'ajouter certains éléménts extérieurs. On voit donc que les profiles sont configurés plus souvent afin de "modifier" l'existant, dans les catégories les plus importantes.

### Question 2 : Au cours de quels événements les développeurs implémentent-ils des Maven Profiles dans un projet ?

*Attention, les résultats présentés dans cet section sont biaisés par les problématiques détaillées dans la section <a href="#Quels-sont-les-principaux-problèmes-résolus-par-l’utilisation-des-Maven-Profiles-">IV. Hypothèses et Expérimentations</a> de cette question.*

Du fait des problèmes rencontrés lors de l'expérimentation, nous ne pouvons pas exploiter les informations sur les branches des commits et sur leur nature (merge ou non merge).
Nous pouvons utiliser le tag et le commit message. Le diff type est aussi exploitable mais nous ne garantissons pas la validité.

ICI DISQUALIFIER ADD/REMOVE/UPDATE

ICI GRAPH 

INTUITION RELEASE (ATTENTION TAG PEUT BIAISER)

DISQUALIFIER LA QUESTION



### Question 3 : L'environnement technique du projet influence-t-il l’utilisation des Maven Profiles ?

Pour rappel, tous les POMs analysés ici possèdent des *profiles*. Nous avons structuré le résultat de cette question en deux parties principales. 

Dans la première partie, nous avons calculé le pourcentage de POMs qui utilise chaque technologie parmi des technologies de CI, Docker, le framework Spring ou au moins un des ORMs importants de l'éco-système Java. Pour avoir les différents ensembles en fonction des usages des technologies, nous avons représenté ces résultats sous forme de diagramme de Venn : 

![Percentage-Pom-using-techno.png](https://raw.githubusercontent.com/RIMEL-UCA/RIMEL-UCA.github.io/maven-profile-timeline/chapters/2020/assets/MavenProfile_Timeline_Percentage-Pom-using-techno.png)

Ici figure donc le détail par technologie seule mais aussi toutes les intersections entre les ensembles. 
Tout d'abord, en additionnant ces pourcentages, on remarque que 65,8% des dépôts que nous analysons utilisent seules ou plusieurs de ces technologies. 
Ensuite contrairement à ce à quoi nous nous attendions,  Docker est la deuxième technologie la plus utilisée dans notre *dataset* selon notre analyse : 22,5% des dépôts possèdent au moins un Dockerfile ou un docker-compose. Cela reste loin de la majorité, mais signifie quand même que plus d'un projet sur cinq a des *profiles* pour configurer son *build* et créer des *containers*/images Docker.
Dans l'ensemble, les outils de type CI sont utilisés par presque un projet sur 2 (48,18%). Une intuition pour expliquer cela peut être que les profiles peuvent par exemple être utilisés dans les *pipelines*.

La deuxième partie des résultats de cette question est le nombre moyen de *profiles* dans les POMs des dépôts utilisants les technologies que nous recherchions. Nous avons ici aussi eu recours à un diagramme de Venn pour visualier tous les ensembles :

![Average-profiles-per-techno.png](https://raw.githubusercontent.com/RIMEL-UCA/RIMEL-UCA.github.io/maven-profile-timeline/chapters/2020/assets/MavenProfile_Timeline_Average-profiles-per-techno.png)

Les résultats de cette analyse vont encore une fois à l'inverse de notre intuition. Car quand Docker est utilisé dans un projet, le nombre de *profiles* moyens est tout à fait comparable aux taux de *profiles* des autres technologies, si ce n'est ORM, qui possède relativement peu de *profiles*.

### Conclusion
Pour la question 1 on s'attendait aux différentes phases de développement (dev,  etc) mais pas aux includes (mauvaise utilisation des profiles ?)

1
2
3

Conclure generale

Lors de ce projet, nous avons beaucoup appris sur la manière dont une étude doit être conduite. Nous nous sommes rendu compte de la nécessité de critiquer nos résultats et de prendre du recul vis à vis de ces derniers pour mieux les analyser.
<!--

1. Analyse des résultats & construction d’une conclusion : Une fois votre expérience terminée, vous récupérez vos mesures et vous les analysez pour voir si votre hypothèse tient la route. -->

## VI. Outillage

Au cours de cette étude, nous avons développé plusieurs outils pour récolter et analyser des projets dotés de Maven Profiles.

- Definition-tool (Java/Spring) : L'outil definition-tool permet de trouver des poms de projets via l'API github. Ces poms sont parsés et analysés (contenu des profils : plugins, dependences et properties), les résultats sont insérés dans une base de données Neo4J.
- Script Analyse (Python) : Ce script utilise les base de données récupérée par l'outil définition-tool afin de récupérer des métriques sur les repositories des projets dont nous avons récupéré les poms.
- Event-search-tool (Java/Spring) : Cet outil clone les projets sélectionnés précédemment avec le script d'analyse Python pour analyser les commits qui ont impactés les profils dans le pom. L'outil permet aussi de définir la stack technologique utilisée par le projet. Les résultats produits par cet outil sont stockés dans une base de données Neo4j.

Nous avons travaillé sur la reproductibilité de notre étude. Ainsi, nous fournissons une documentation compléte permettant de reproduire nos expériences. À noter que les données récupérées sur l'API Github par l'outil "definition-tool" varient. En effet, nous utilisons une fonction de l'API qui renvoi du code indéxé en permanence (lorsque les développeurs commit). Il est donc impossible de reconstruire le même *dataset* que nous. C'est pourquoi nous mettons à disposition nos deux bases de données Neo4J sous forme de volumes docker. Ces dernières peuvent être utilisées via le docker-compose fournit dans le projet.

### Diagramme de Venn

Lors de l'analyse des résultats, nous avons voulu faire un diagramme de Venn pour avoir une répartition de l'utilisation des certains profils pour les quatres technologies étudiées. Nous avons donc effectuées des requêtes Cypher (Cf. <a href="#VIII-Annexes">VIII. Annexes</a>), puis déterminer dans quel ensemble se trouver les données pour le diagramme. C'est pourquoi nous avons utilisé un outil externe<a href="#VII-Références">[3]</a> pour faire cette séparation des données. Enfin, nous avons utilisé un script en R pour visualiser ce diagramme avec les données issues du script.

<!-- Précisez votre utilisation des outils ou les développements \(e.g. scripts\) réalisés pour atteindre vos objectifs. Ce chapitre doit viser à \(1\) pouvoir reproduire vos expériementations, \(2\) partager/expliquer à d'autres l'usage des outils.-->

## VII. Références

1. API GitHub : https://developer.github.com/v3/
2. Neo4j : https://neo4j.com/docs/cypher-refcard/current/
3. Générateur diagramme de Venn : http://bioinformatics.psb.ugent.be/webtools/Venn/

## VIII. Annexes

### Téléchargement des POM

`https://raw.githubusercontent.com/<owner>/<repo>/<branch>/<filepath>`

### Routes pour les requêtes vers l'API GitHub

`https://api.github.com/search/code`
`https://api.github.com/repos/:owner/:repo`
`https://api.github.com/repos/:owner/:repo/stats/contributors`

### Requêtes Cypher (Neo4j)

Pour assurer une meilleure reproductibilité, les requêtes utilisées sur nos bases de données sont disponibles ci-dessous avec de brèves explications.

#### Question 1 : Quels sont les différents types de Maven profiles dans les projets Open Source ?
Les requêtes effectuées pour répondre à cette question ont été faites sur la base de données correspondante.
Les requêtes pour catégoriser les profiles en utilisant leur nom sont de ce type : 
`MATCH (p:Pom)-[r]->(n:MavenProfile) WHERE toLower(n.name) CONTAINS 'release' AND NOT(toLower(n.name) CONTAINS 'dev') AND NOT(toLower(n.name) CONTAINS 'maven') AND NOT(toLower(n.name) CONTAINS 'include') AND NOT(toLower(n.name) CONTAINS 'metrics') AND NOT(toLower(n.name) CONTAINS 'prod') AND NOT(toLower(n.name) CONTAINS 'test') return count(n)`
Cette requête renvoie le nombre de profiles contenus dans la catégorie "release". Pour les autres catégories, il convient de changer ce que doit contenir la variable `n`, de même que ce qu'elle ne doit pas contenir en conséquence pour avoir des ensembles disjoints.

Afin d'obtenir les statistiques de configuration des profiles, nous avons effectué ce type de requêtes : 
`match (n:MavenProfile)-[d:PLUGINS]->() where toLower(n.name) CONTAINS 'metrics' AND NOT(toLower(n.name) CONTAINS 'release') AND NOT(toLower(n.name) CONTAINS 'dev') AND NOT(toLower(n.name) CONTAINS 'maven') AND NOT(toLower(n.name) CONTAINS 'include') AND NOT(toLower(n.name) CONTAINS 'prod') AND NOT(toLower(n.name) CONTAINS 'test') return count(d)`
En faisant varier le type de la variable `d`, de `PLUGINS`, à `DEPENDENCIES` et `PROPERTIES`. De même que pour analyser la catégorie, il faut faire varier le `name` de la variable `n`, et les négations en conséquence, pour conserver des ensembles disjoints.

Nous voulions obtenir le nombre d'utilisation de plugins pour certains *profiles*. Pour simplifier, nous allons indiquer ici `$PROFILE` comme variable, où nous avons cherché tous les différents profils utilisés dans la requête précédente.

`match (n:MavenProfile)-[r]-(a:MavenPlugin) WHERE n.name CONTAINS '$PROFILE' return n.name, sum(r.weight) as usage`

#### Question 2 : Au cours de quels événements les développeurs implémentent-ils des Maven Profiles dans un projet ?

#### Question 3 : L’environnement technique du projet influence-t-il l’utilisation des Maven Profiles ?

Nous avons fait deux requêtes ici pour les quatres technologies donc huit requêtes. Pour simplifier, nous allons indiquer ici `$STACK` comme variable qui peut prendre comme valeur `DOCKER`, `ORM`, `CI`, `SPRING`.

La requête suivante permet de trouver tous les poms qui sont liés à une technologie.

`MATCH (t:Technology)--(p:Pom)--(mp:MavenProfile) WHERE t.stack='$STACK' RETURN DISTINCT p`

La requête suivante permet de trouver les POM qui sont associés à une technologie en particulier et de compter le nombre de profils associés à chaque POM.

`MATCH (t:Technology)--(p:Pom)-[r]-(:MavenProfile) WHERE t.stack = '$STACK' WITH count(DISTINCT r) as nbMavenProfile, p RETURN p as Pom, nbMavenProfile`
