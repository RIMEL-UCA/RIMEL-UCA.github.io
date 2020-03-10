---
layout: default
title : Comment sont utilisés les profils Maven dans les projets Open-Source des grandes entreprises ?
date:   2020-03-10 20:00:00 +0100
---


# Comment sont utilisés les profils Maven dans les projets Open-Source des grandes entreprises ?

## Auteurs

Nous sommes trois étudiants en dernière année à Polytech Nice Sophia en spécialité Architecture Logicielle :

* FERRE Jérémi &lt;jeremi.ferre@etu.unice.fr&gt;
* HENAULT Dorian &lt;dorian.henault@etu.unice.fr&gt;
* JUROSZEK Florian &lt;florian.juroszek@etu.unice.fr&gt;


# Mise en contexte

Les profils maven permettent, à partir d’un projet donné, de créer différentes configurations qui peuvent s’intégrer aux phases du cycle de vie de compilation (packaging, testing, déploiement, validation etc...). On peut ainsi s’adapter à différents environnements (Windows, Mac, Linux)  en utilisant les profils maven, ou encore permettre de réaliser des profils pour divers tests, que ce soit des tests d’intégration, des tests end-to-end, etc. 

Notre objectif est de découvrir comment la plupart des grandes entreprises ou gros projets open-source utilisent ce système de profil et dans quel but. Plus précisément, nous cherchons des corrélations entre les types de projets et l’utilisation des profils maven, afin de potentiellement déduire des bonnes pratiques (ou en tout cas des pratiques utilisées en production) sur les façons de se servir de ces profils.

# Problématiques
Notre question générale était initialement : Comment sont utilisés les profils Maven ? Mais notre objectif étant d’étudier les pratiques des grandes entreprises, nous avons déterminé des problématiques plus précises. 

Comme nous utilisons Maven nous-même, nos intuitions sur l’utilisation des profils sont déjà forgées. Nous sommes donc partis d’une approche en partant de nos intuitions, puis en construisant au fur et à mesure les réponses à nos questions. Nous avions ainsi prévu de réaliser deux phases d’analyse : la première consistait à étudier des projets open-source réalisés par de grandes entreprises, afin d’identifier des potentielles “bonnes pratiques” sur l’utilisation des profils Maven. La seconde étape prévue était de regarder les projets open-source de la communauté, afin de comparer les pratiques des entreprises et tenter d’obtenir une corrélation.

Dans la suite de ce chapitre, nous allons nous concentrer sur notre méthodologie d’étude des grandes entreprises, et comment nous avons identifié les métriques d’utilisation des profils Maven. Nous nous sommes donc posé les questions suivantes :


- Comment accéder aux projets qui manipulent différents profils Maven ? 
- Dans quel but les profils sont-ils majoritairement utilisés dans les grandes entreprises, qu’est-ce qui diffère entre les profils suivant les situations ?
- Peut-on en dégager une tendance globale ou cela dépend fortement du type de projet étudié ?
- Quelles solutions apportent les profils maven sur les enjeux de portabilité d’un projet ?

# Hypothèses & Expériences

## Intuitions initiales 

La documentation de Maven met en avant l’aspect de portabilité que des profils peuvent apporter à un projet, en permettant de s’adapter aux conditions d'exécutions qui peuvent différer selon les environnements. Notre intuition première était de voir s’il semblait pertinent d’affirmer que les profils sont majoritairement utilisés à ses fins-là dans les grandes entreprises. D’autres usages pourraient être trouvés, comme par exemple pour configurer des tests spécifiques. 

A partir des informations du dataset constitué, nous voulions extraire les premières indications permettant d’associer à un profil des critères de portabilité. Nous voulions nous appuyer sur des articles et documents qui traitent de ce sujet, où il serait alors possible d’en dégager un ensemble de métriques et de bonne pratiques relatives à la portabilité. 

Nous pensions qu’il serait intéressant de creuser dans cette direction, et nous avions ainsi sélectionné un ensemble de documents qui traitaient de profils et de portabilité :

- Apache Maven documentation - Introduction to profiles  https://maven.apache.org/guides/introduction/introduction-to-profiles.html
 
- Maven: The Complete Reference
https://books.sonatype.com/mvnref-book/pdf/mvnref-pdf.pdf 

- Unix to Linux - Chapter 1: Porting Project Considerations - Mendoza, Walker, Chakarat http://catalogue.pearsoned.co.uk/samplechapter/0131871099.pdf

L’aspect de portabilité dans les profils maven a constitué ainsi le “fil rouge” de nos recherches, et nous avons tenté de centrer nos recherches autour de ce sujet.

## Protocole de recherche

Afin de déterminer une réponse fiable à notre problématique, nous avons établi un protocole de recherche empirique basé sur des métriques atomiques. Ces métriques sont des calculs numériques à réaliser sur un jeu de données composé de projets Java utilisant Maven. Nos assertions sont les suivantes :


- Nombre de profils dans le pom
Nombre de profils ayant un nom contenant “dev”, “prod” ou “release”
- Nombre de profils ayant un nom contenant “test”
- Nombre de profils re-définissant des properties
- Nombre de profils contenant une condition d’activation

Ces métriques étant basées sur nos observations et nos intuitions, elles ne donneront pas forcément de résultats convenables. De plus, les assertions vérifiant des mot-clés dans les noms de profils peuvent aussi être soumises à des biais. Il faut donc toujours garder un esprit critique sur nos statistiques. Néanmoins, afin de valider ou réfuter nos prédicats, nous considérons qu’une métrique pourra être pertinente à partir d’environ 33% d’apparition dans notre dataset. En dessous de ce seuil, nous admettons que la pratique est négligeable ou trop spécifique par rapport à toutes les grandes entreprises. Cette donnée est néanmoins assez arbitraire, et il est donc complexe de trouver une méthode totalement pertinente.

Notre protocole se résume donc en 4 grandes étapes : 

1. Construire un jeu de données varié et représentatif des grandes entreprises;
2. Extraire les projets Maven depuis le dataset;
3. Parcourir tous les pom en calculant nos métriques sur chacun d’eux;
4. Interpréter les résultats grâce à des analyses statistiques (pourcentages, graphiques etc...).

La concrétisation de ce protocole doit donc commencer par l’élaboration de différents scripts, pour automatiser les étapes qui seraient manuellement trop coûteuses en temps.

# Description des scripts réalisés 

Nous avons développé deux scripts principaux pour répondre à notre problématique. Le premier est un script de fouille en Python qui utilise l’API de Github pour récupérer des projets selon certains critères. Le second est un script qui permet d’analyser les profils pour un projet donné. Nous avons été contraint d’utiliser deux scripts différents car l’API de Github ne permet pas de récupérer un ensemble de projets tout en vérifiant si ceux-ci contiennent un pom.xml ayant des profils. Cette contrainte a ralenti la constitution de notre dataset car cela multiplie par deux le nombre de requêtes par repository. Or, le rate limit de l’API est relativement strict pour effectuer une fouille de données (30 requêtes par minutes en passant un token nous identifiant). Cette contrainte nous oblige à espacer nos appels. Par ailleurs, cela nous a permis, de manière disjointe, la construction du dataset de projets de grandes entreprises et l’exploitation de nos métriques et assertions sur ce dernier. Ainsi, même s'il est nécessaire de parcourir tous les projets, nous n’avons plus à les rechercher lorsque nous modifions nos assertions.

Intéressons-nous un peu plus au premier script de fouille. Il s’agit d’un script Python. Le langage a été choisi pour sa simplicité et sa facilité à faire de requêtes HTTP. Dans sa première version, nous ne pouvions spécifier qu’un seul nom d’utilisateur Github (donc entreprise) et un seul langage cible. Avec ces paramètres, nous pouvons construire la requête. Ainsi, nous pouvions déjà récolter les projets de ces entreprises en analysant la réponse de l’API et la système de pagination de cette dernière. Nous avons également  pu constater certaines incohérences provenant de l’API notamment au niveau du tri lorsqu’il est associé à la pagination. Lors de l’extraction et de l’analyse de la réponse, nous construisons un fichier CSV. Ce format nous permet d’exploiter facilement les informations dans un tableur type Excel. La deuxième version de ce script survient suite à l’identification de nombreuses entreprises sur lesquelles effectuer la recherche. Nous avons donc mis en place une boucle suivant la lecture d’un fichier texte comprenant les noms d’utilisateurs Github de ces entreprises. Ainsi, nous pouvions itérer sur celles-ci en reprenant le processus de la première version. Nous avons ajouté la recherche de projets de trois différents langages (Java, Kotlin et Scala) de la même manière, en parcourant cette fois-ci un tableau, le nombre d’éléments étant beaucoup moins importants. 

Une fois le dataset constitué de divers projets de grandes entreprises, il était nécessaire de les filtrer et analyser ceux-ci. Ainsi, un second script a été écrit en NodeJS (ce langage a aussi été choisi pour la facilité qu’il offre à écrire des algorithmes). Il lit le fichier écrit par le premier script de fouille afin de récupérer le chemin du projet et utiliser l’API Github pour un second type de requête. Premièrement, nous cherchons la présence d’un fichier pom.xml dans le repository. Si celui-ci existe, nous le téléchargeons afin de pouvoir l’analyser. La seconde partie du script utilise le chemin vers le pom et une liste d’assertions à vérifier. Une assertion est représentée par un nom et une liste de condition faisant varier le résultat du test. Le fichier étant sous le format XML, il est facile de naviguer à travers les données pour filtrer ou compter des propriétés. Il a ainsi été simple pour nous d’ajouter, modifier ou supprimer des assertions dans le script. Tout comme le premier script, nous écrivons le résultat de nos calculs dans un fichier de résultats, sous forme de CSV, afin de les exploiter notamment manuellement avec des graphes.

Enfin, nous avons développé plusieurs autres scripts “utilitaires” afin de simplifier notre exploitation. Ces derniers nous ont été utiles principalement pour formater nos fichiers de résultats en CSV. En effet, pour préparer celui-ci à son exploitation dans un tableur il fallait par exemple remplir des colonnes avec le nom du projet ou encore transformer les compteurs en booléen afin de sortir des pourcentages ou des graphiques en camembert par exemples.

# Premiers résultats

Pour nos premiers résultats, nous n’avons pas obtenu un ensemble de projets satisfaisants. En effet, malgré le nombre important de projets récupérés (environ 1300), nous n’avions pas obtenu de projets provenant de nombreuses entreprises différentes. Les principaux projets appartenaient à la fondation Apache et à Google et peu de ces projets contenaient divers profils dans leur configuration Maven.  Ce faible nombre d’entreprises représentées dans notre dataset était en fait dû à une erreur de notre part concernant les critères de recherche pour constituer ce dernier. Nous étions partis du principe que, comme les grandes entreprises sont populaires, les projets qu’ils mettaient à disposition sur des plateformes comme GitHub allaient également être populaires. La popularité d’un projet se caractérise par le nombre de “Stars” que celui-ci possède. Plus ce nombre est élevé, plus il est connu du grand public. Nous avons alors cherché manuellement un certain nombre d’utilisateurs sur git qui appartenaient à de grosses entreprises (donc Apache, Google et consorts) et avons récolté depuis ces utilisateurs tous les projets ayant le nombre de “Stars” le plus élevé. Nous nous étions rendu compte que ce type de recherche nous limitait fortement dans la diversité des entreprises, et que le filtrage par "Stars" n’était pas un indicateur assez pertinent.
Nous avons donc revu nos objectifs, dans la mesure où la constitution d’un dataset approprié à nos recherches se trouvait être plus long et complexe que prévu.

D’un autre côté, nous avions débuté l’approfondissement de divers articles relatant de la portabilité pour en extraire des pratiques. Par exemple, le document (Unix to Linux - Chapter 1) décrit que les dépendances d’un projet vers des versions spécifiques de “softwares” sont nécessairement à connaître pour que le processus de portabilisation d’un projet (“porting process”) soit réalisable. Également, la notion de “scoping” dans un processus comme celui-ci est réalisé grâce à l'évaluation des différents composants de l’environnement. On retrouve cette notion avec Maven, où il est possible dans un profil de spécifier des “Modules” pour un projet. 

Nous avions débuté la prise en main de l’outil d’analyse des profils afin d’extraire quelques métriques. Nous avions ainsi observé si les profils contenaient des balises “properties". Cela signifiait que les propriétés d'un projet (comme des URL, des identifiants) pouvaient varier.

![properties-count](https://github.com/RIMEL-UCA/RIMEL-UCA.github.io/blob/master/chapters/2020/assets/MavenProfileUsageAndProjectType/PROP.png)

Nous avions également constaté que des profils étaient utilisés principalement pour mettre en place des livraisons (“release”), et nous avons tenté de quantifier le nombre de profils mettant en oeuvre cette tâche.

![release-count](https://github.com/RIMEL-UCA/RIMEL-UCA.github.io/blob/master/chapters/2020/assets/MavenProfileUsageAndProjectType/RELEASE.png)

Les recherches de métriques étaient variées mais il nous était encore difficile de caractériser avec précision des métriques qui étaient étroitement liées à des processus de portabilisation d’un projet.

Après avoir obtenu plus de maîtrise sur notre sujet et découvert ce qu’il était possible de rechercher ou non, il était nécessaire de réaliser une nouvelle itération de notre processus dans sa globalité afin d’obtenir des résultats concluants. Dans un premier temps, il nous était essentiel de trouver un dataset qui correspondait mieux à nos critères et qui nous permettrait d’obtenir des résultats plus pertinents. Ensuite, il était important de trouver des métriques permettant d’affirmer si les profils Maven étaient utilisés à des fins de portabilisation d’un projet.

# Revue des objectifs et nouvelles expériences

La préoccupation de notre revue concerne premièrement la constitution d’un nouveau dataset. Pour réaliser cela, deux recherches différentes ont été mise en place. D’un côté, nous avons “amélioré” la première expérience qui consistait en une recherche manuelle des utilisateurs correspondant à des profils d’entreprise. Cette fois-ci, nous avons tout simplement concentré nos recherches sur les sociétés qui sont les plus actives sur les dépôts git comme GitHub.

![ranking-companies](https://github.com/RIMEL-UCA/RIMEL-UCA.github.io/blob/master/chapters/2020/assets/MavenProfileUsageAndProjectType/companies.png)

Classement des entreprises les plus actives dans l’open-source (https://solutionshub.epam.com/osci)

A partir de ces résultats, il nous a alors été possible de récolter tous les projets appartenant à ces entreprises. Il a par la suite fallu filtrer les projets pour n’obtenir que ceux qui utilisaient potentiellement Maven.
L’autre recherche constituait en l’analyse d’archive de GitHub qui dataient de 2019. En effet, la plateforme Google permet via l’utilisation de son outil “BigQuery” de réaliser des requêtes sur un ensemble volumineux de données. Nous avons trouvé un article qui traitait justement les différents entreprises qui ont le plus contribué sur GitHub en 2017 - 2018 : https://www.freecodecamp.org/news/the-top-contributors-to-github-2017-be98ab854e87/ .

Nous avons alors pu récupérer la requête SQL que l’utilisateur a réalisé pour récupérer cela. Nous avons tenté de modifier cette requête pour obtenir les informations qui nous intéressaient, basé sur les archives de GitHub en 2019. Nous avons ainsi pu ajouter un filtre sur les projets qui étaient développés en Java. 

![big-query](https://github.com/RIMEL-UCA/RIMEL-UCA.github.io/blob/master/chapters/2020/assets/MavenProfileUsageAndProjectType/BIG-QUERY.png)

L’interface BigQuery se présente sous la forme suivante. A noter que plus d’1,5 To de données ont été traités pour obtenir les projets, qui en cas de résultats concluants pouvaient constituer notre dataset. Le principe de la requête est de rechercher les projets en filtrant par les contributeurs qui ont des adresses mail professionnelles, et qui ont tendance à souvent contribuer. L’approche de la recherche semble étonnant mais les résultats sont particulièrement pertinents.

D’un autre côté, il restait essentiel de donner plus de cohérence à nos métriques pour pouvoir démontrer un quelconque processus de portabilisation d’un projet avec l’usage des profils Maven. Il était également essentiel de garder en mémoire qu’il ne s’agissait-là que d’une intuition, et qu’il était totalement possible de tomber sur une démonstration qui ne pouvait aboutir. Il fallait néanmoins être capable de trouver des métriques qui permettraient de valider ou réfuter l’hypothèse de la portabilité.  

Nous avons alors tenté de récupérer au sein des profils maven des critères sur les modifications de versions d’éléments qui sont modifiés via les propriétés système, comme des versions de base de données, ou des versions de framework. Ces modifications de version sont des indicateurs de portabilité, comme expliqué précédemment après étude du document (Unix to Linux - Chapter 1). Enfin, le dernier critère qui permettrait d’affirmer si les profils Maven sont utilisés à des fins de portabilité est l’usage de la balise “activation”. Cette balise est essentielle à l’usage de profils, car elle permet l’activation automatique de profils, sous certaines conditions.  En effet, il est possible de spécifier l’activation  d’un profil si l’environnement technique dans lequel est exécuté le projet Maven correspond. On peut par exemple demander à activer un profil automatiquement si l’environnement d’exécution du projet tourne sur Windows ou sur Linux.
Il existe un ensemble de balises qui vont permettre de vérifier les propriétés du système et agir en conséquence. On peut vérifier si des variables d’environnement précises sont présentes, la configuration OS, la présence/absence d’un fichier, ou encore des paramètres de Maven. Si les vérifications sont validées, le profil sera activé de façon automatique. 

Pour plus de détails, consulter la documentation officielle sur les profils maven : https://maven.apache.org/guides/introduction/introduction-to-profiles.html#, partie “Details on profile activation”.

Si cette balise “activation” est présente en grand nombre dans les profils Maven, on peut clairement affirmer que ceux-ci permettent d’adapter un projet à l’environnement dans lequel il est exécuté, et renforce donc sa portabilité.

# Nouveaux résultats et interprétation 

Premièrement, intéressons nous aux modifications des scripts de fouille. L’ajout des entreprises les plus actives dans l’open-source a considérablement augmenté notre ensemble de projet. En effet, nous en avons récupéré 1699 dont la majorité sont tout de même écrits en Java. Outre ce nouveau volume de dataset qui nous satisfaisait, ce dernier était surtout beaucoup plus diversifié au niveau des entreprises, ce qui était prometteur pour nos expérimentations. 

À cela est venu s'ajouter les projets récupérés grâce à la requête de Google BigQuery. Comme décrit précédemment, ce script a permis de rechercher des projets dans un ensemble beaucoup plus large que celui offert par la Github API. C’est comme cela que nous avons pu récolter 755 projets supplémentaires provenant de 65 entreprises différentes. 

Une fois les projets récoltés à partir des deux méthodes, il fallait s’assurer que l’ensemble ne comprenait pas de doublons afin de ne pas fausser les résultats de nos analyses. Nous avons ainsi fusionné les deux sources de données et obtenu un taux de duplication très raisonnables (moins de 50%) nous permettant de finalement obtenir un total de 2454 projets. Par rapport à la première phase de l’étude, nous sommes tout à fait satisfait de cet ensemble que nous considérons valide par sa taille et la diversité des entreprises et des langages (de manière minimale).

À partir de celui-ci, nous avons recherché la présence d’un fichier pom.xml à la racine de chaque projet afin de filtrer notre dataset. Après cette étape, nous obtenons 598 projets sur lesquels nous pouvons tester nos différentes assertions. Le nombre semble relativement faible par rapport au nombre total de projets récoltés, mais il faut savoir qu’un nombre important d’entreprises mettent en place des projets via l’utilisation de l’outil Gradle, au lieu de Maven.

Suite aux observations citées précédemment, nous nous sommes intéressés aux balises “<Activation>” qui sont un des moyens, si ce n’est le moyen, recommandé par Maven pour faire de la portabilité. Par conséquent nous avons tout d’abord compté le nombre de projets au sein de notre dataset utilisant ce mécanisme. Les résultats sont probants. Sur 565 profils Maven analysés, plus de 350 (354 exactement) contiennent des balises d’activations. De plus, si nous comparons ces chiffres avec les précédentes métriques, nous pouvons constater le compte des balises d’activations est supérieur ou très supérieur aux autres comptes au niveau notamment des différentes propriétés. Nous pouvons donc ainsi penser que ces balises sont utilisés par les grandes entreprises.

![profiles-analyse](https://github.com/RIMEL-UCA/RIMEL-UCA.github.io/blob/master/chapters/2020/assets/MavenProfileUsageAndProjectType/METRIC_NUMBER_TOTAL.png)

Ces résultats nous ont motivés à approfondir l’utilisation de ce mécanisme. Ainsi en lisant la documentation de référence, nous avons ajouté des métriques sur les moyens possibles de mettre en place les activations.

![activations-analyse](https://github.com/RIMEL-UCA/RIMEL-UCA.github.io/blob/master/chapters/2020/assets/MavenProfileUsageAndProjectType/ACTIVATIONCAMEMBERT.png)

Sur 354 activations, nous observons donc que le mécanisme est, pour 54% des balises, mis en place via des propriétés. La moitié des activations de profils dépendent donc des propriétés systèmes. Pour 160 profils, une activation est définie par défaut. Nous pouvons donc constater avec ce graphique que les balises d’activations sont effectivement utilisées à des fins de portabilité. Cette expérience ne nous permet pas de valider notre ressenti car il faudrait analyser plus précisément d’une part les valeurs des conditions et des propriétés dans ces balises d’activations. D’autre part, il faudrait également étudier les liens entre les différentes balises dans différents profils d’un même fichier pom.xml.

# Conclusion et perspectives
Maven présente les profils comme étant le mécanisme principal pour rendre un projet portable. Nous avons donc avec cette étude analysé la composition des fichiers pom afin de constater ou non l’utilisation de cette fonctionnalité. Nos métriques vont donc dans le sens de la portabilité. Nous nous sommes basés sur un ensemble de projets de grandes entreprises qui, grâce à leurs budgets et de nombreux employés, ont de plus fortes chances de développer des gros projets avec beaucoup de caractéristiques de portabilité. Nous pensons avoir constitué un dataset conséquent et représentatif de ce domaine en utilisant plusieurs méthodes de récolte. Nous avons, avec une démarche incrémentale, affiné nos métriques et analyses, pour finalement se concentrer sur les balises “activations”. Ces dernières se sont avérées très utilisées avec notamment celles basées sur les propriétés systèmes qui semble être prépondérantes. 

Concernant notre ensemble de projets, nous pourrions également tenter de récolter encore plus de données. Cela serait possible en établissant des partenariats avec des entreprises afin d’accéder à des projets non open-sources mais tout aussi intéressants pour étudier la portabilité ammenée par les profils Maven.

Nos métriques actuelles sur les balises activation sont uniquement numéraires. Il serait appréciable d'extraire le contenu exact des activations afin d'approfondir notre investigation. Nous aimerions donc constituer un “bag of words”. En effet, nous savons que les activations par les propriétés systèmes sont beaucoup utilisés mais il serait d’autant plus intéressant de connaître quelles sont ces propriétés. Ainsi, nous pourrions faire de même pour les versions des JDK, des OS ou encore les noms des fichiers qui doivent être présents pour activer un profil.


# Références

 * Apache Maven documentation - Introduction to profiles  https://maven.apache.org/guides/introduction/introduction-to-profiles.html  

 * Maven: The Complete Reference - https://books.sonatype.com/mvnref-book/pdf/mvnref-pdf.pdf  

 * Unix to Linux - Chapter 1: Porting Project Considerations - Mendoza, Walker, Chakarat http://catalogue.pearsoned.co.uk/samplechapter/0131871099.pdf  

 * OSCI Ranking - https://solutionshub.epam.com/osci  

 * GitHub Top Contributors 2017 - https://www.freecodecamp.org/news/the-top-contributors-to-github-2017-be98ab854e87/  
