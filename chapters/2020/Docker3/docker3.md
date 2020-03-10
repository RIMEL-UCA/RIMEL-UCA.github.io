# Les configurations Docker permettent-ils de rendre le code plus simple ?

## Auteurs

Nous sommes 4 étudiants en dernière année à Polytech Nice-Sophia (SI5), en Architecture Logicielle :

* Lazrak Sami &lt;sami.lazrak@etu.unice.fr&gt;
* Larabi Walid &lt;walid.larabi@etu.unice.fr&gt;
* Aourinmouche Soufiane &lt;soufiane.aourinmouche@etu.unice.fr&gt;
* Massol Damien &lt;damien.massol@etu.unice.fr&gt;

## I. Contexte de la recherche

Durant cette dernière décennie, la conteneurisation est devenue une pratique courante au sein du développement logiciel car elle permet de simplifier plusieurs choses dans un environnement logiciel dont le **déploiement** et l'**isolation des composants**. Elle permet également de disposer d'**environnements homogènes** et de gérer le **dimensionnement de l'infrastructure**. 

**Docker** est devenu la référence principale au niveau des technologies de conteneurs, et est adopté massivement par d'autres outils. On pourrait citer par exemple **Kubernetes** qui propose de l'orchestration de conteneurs, ou encore des solutions Cloud telle que **Google Cloud Run** qui propose du scaling automatique de conteneur stateless. 

De nos jours, l'utilisation de la conteneurisation est très répandue et son impact sur la simplicité du déploiement et la gestion d'une infrastructure logicielle n'est plus à prouver. 

![Figure 1: Logo UCA](../assets/model/UCAlogoQlarge.png)

## II. Question générale

Dans ce contexte, nous pouvons nous demander si conteneuriser son logiciel simplifie également le développement du code d'un projet, sa qualité et la manière qu'ont les développeurs de l'appréhender.

La question sur l’utilisation des paramètres de haut niveau pour agir sur les logiciels conteneurisés est intéressante parce qu'elle peut avoir des impacts à différents niveaux : 

* Dans le contexte de CI/CD par exemple, la configuration varie considérablement d'un déploiement à un autre, alors que le code non. 

* Il peut également y avoir une même base de code et des comportements différents, entrainés par des features toggle, par exemple, des features accessibles en fonction de la licence de produit, etc...

On peut alors relier les configurations Docker de haut niveau à des thématiques importantes sur les bonnes pratiques du développement logiciel.

Durant notre parcours, nous avons développé des applications qui stockent la configuration sous forme de constantes dans le code or il s’agit d’une violation de la méthodologie “Twelve Factor App” (Reference 2), qui nécessite une séparation stricte de la configuration du code; pour une meilleure maintenabilité, portabilité et résilience. 

Cette mauvaise pratique nous a coûté beaucoup de temps et d’efforts et nous a poussé à choisir ce sujet, pour nous informer et voir comment ces pratiques sont utilisées dans le monde industriel / projets open-source, la complexité ajoutée et l’intérêt réel que cela apporte.  



**Question générale : Comment les configurations haut niveau de Docker influent-elles sur la simplicité du code ?**

Commençons par définir ce qu'est pour nous un code simple : 

Un code simple est un code **lisible**, c'est-à-dire **structuré**; mais c'est aussi un code **maintenable**, donc **flexible** et **extensible**.

Afin de faciliter notre étude, nous avons restreint notre analyse en étudiant l'influence d'un seul type de configuration haut-niveau de Docker : _les variables d'environnements_ 

En effet, parmi toutes les autres façons de configurer des containers Docker (Volumes, Ports, Restart Policies, File Systems...), les variables d'environnements ont plus de chances d'être utilisées dans le code, et donc d'influencer directement ce que l'on appelle la simplicité.  

## III. Rassemblement d'informations

### 1 - Les projets à analyser

Afin de mener notre étude à bien, nous nous sommes mis à la recherche de **jeu de données**, c'est-à-dire de projets pour appliquer notre démarche.

Notre démarche consistait à explorer les projets open-source hébergés sur GitLab/GitHub (parmi les dépôts de code source les plus utilisés aujourd’hui).

Tout d'abord, il fallait que l'on ait des projets qui utilisent **Docker**, et que dans les fichiers de configurations Docker (_docker-compose.yml_ ou _Dockerfile_) il y ait **un nombre suffisant de variable d'environnements** initialisées.

Ensuite, d'autres critères de sélections sont entrés en jeu comme :
* L'historique qui doit être important 

  > Pour pouvoir observer un projet à différent moments de son développement

* la popularité (stars) 

  > Pour observer des projets aboutis, avec un grand nombre de contributeurs

* les messages de commit qui doivent être clairs 

  > Pour faciliter la recherche par mots clés

* langage dont on connaît les conventions (JS, Python et Java) 

  > Pour travailler sur une base de code dont on peut voir les défauts/qualités



Ces critères nous ont permis de comparer des projets de différentes complexités, ayant un certain besoin de développer sur plusieurs phases ou pas, ou d’avoir un système qui change de comportement. 

Les projets que nous avons retenus sont :
* [ElasticSearch](https://github.com/elastic/elasticsearch) : **42** variables d'environnement
* [Thingsboard](https://github.com/thingsboard/thingsboard.git) : **64** variables d'environnement
* [Magma](https://github.com/facebookincubator/magma) : **104** variables d'environnement
* [Apache Skywalking](https://github.com/apache/skywalking) : **8** variables d'environnement
* [Openmrs-sdk](https://github.com/openmrs/openmrs-sdk) : **16** variables d'environnement

### 2 - Les outils utilisés

* **Analyse de code**

Dans notre démarche, l'évaluation de la complexité du code est une partie importante, pour nous permettre d'évaluer la simplicité d'un dépôt de code.

Il nous fallait donc un outil pour évaluer la complexité cyclomatique performant, car les projets sélectionnés contiennent des milliers de commits et de fichiers.

Nous sommes partis sur un outil que toutes les personnes de notre groupe connaissaient, et avaient déjà utilisé : **SonarQube**, qui permet plus généralement de mesurer la qualité du code en continu.

Néanmoins nous avons rencontré des problèmes avec Sonar, en effet, pour effectuer une analyse nous avons besoin de builder au préalable le projet concerné. Cependant, les projets que nous avons selectionnés étaient lourds et long à builder, nous n'avons donc pas réussi à réaliser cette étape préliminaire pour simplement évaluer la complexité de notre étude.

Nous avons décidé de nous lancer à la recherche d'un outil qui nous permettrait d'avoir ces métriques, sans nous obliger à builder le projet en amont (un analyseur de code statique). 

A l'issue de cette recherche, nous avons décidé d'utiliser **Lizard**, qui nous permet de récupérer des métriques intéressantes pour notre étude : à l'échelle du code, du fichier et de la méthode. 

Au niveau de la performance, Lizard peut analyser un ensemble d'environ 1000 fichiers en environ 2 minutes, et ressortir un rapport riche sur les métriques de cet ensemble.

* **Recherche dans le code**

Lib Git Python
Walid s'exprimera ...

* **GitJSON**

**GitJSON** est un script permettant de récupérer tous les commits d'un repository sous le format JSON ou chaque commit à le format suivant : 

```json 
{   
    "commit": "0dd313dd61e1d1dbeb89e3f326cb81b944ecabe4",  
    "subject": "new feature",   
    "body": "add feature",
    "verification_flag": "N",   
    "author": {     
        "name": "John Doe",     
        "email": "johndoe@unice.io",     
        "date": "Mon, 20 Jan 2020 15:24:15 +0200"   
    },   
    "commiter": {     
        "name": "Toto",     
        "email": "toto@unice.io",     
        "date": "Mon, 20 Jan 2020 15:24:15 +0200"   
    } 
}
```

Nous avons choisi d'utiliser le format JSON pour simplifier le traitement des commits par nos scripts, le format JSON à l'avantage d'être structuré et simple, permettant ainsi de récupérer directement nos commits sous forme d'objet et d'accéder à leurs données facilement.

## IV. Hypothèse et Démarche

Avec la réduction de scope décrite précédemment, l'hypothèse que nous allons essayer de vérifier est la suivante :

**_Hypothèse_ : L'ajout de variables d'environnements rend le code moins complexe** 

Expliquer pourquoi on choisit cette hypothèse ...

Pour prouver cette hypothèse, nous avons établi le protocole suivant :
- Enumérer toutes les variables d'environnements d'un projet
- Pour chaque variable d'environnement :
-> **Détecter les commits** ayant utilisé la variable d'environnement, et leur **commit précédent**.
-> **Détecter les fichiers** (dans le code) dans lesquels les variables d'environnements ont été ajoutées
-> **Calculer l'évolution de la complexité cyclomatique** de ces fichiers entre le commit précédent 
      et le commit où la variable d'environnement a été utilisée.

Description du protocole ... Etablir le lien avec les outils utilisés (et comment). Justifier en quoi ce protocole permettra de vérifier 
ou pas l'hypothèse.

## V. Mauvaises pistes explorées

Avant de fixer définitivement le protocole utilisé durant cette étude, nous avons exploré d'autres possibilités qui ont échouées 

## V. Présentations des résultats et de l'analyse

1. Analyse des résultats & construction d’une conclusion : Une fois votre expérience terminée, vous récupérez vos mesures et vous les analysez pour voir si votre hypothèse tient la route. 
2. Conclusion de l'analyse


## VI. Description de la partie automatisée

1. **Automatisation de la détection des variables d'environnement**

Nous avons choisi dans un premier temps de tester d'appliquer notre démarche sur un seul projet (ThingsBoard). 

La première chose à faire a été de récupérer les variables d'environnement que le projet contient. 

Vu que cette opération devra être effectuée sur plusieurs projets (voir pourra nous aider à voir le nombre de variables d'environnements dans un projet), nous avons décidé de l'automatiser.

Nous avons automatisée cette partie, qui nous permet de détecter les variables d'environnements d'un projet, en supposant qu'elles soient localisées dans un des fichiers suivants :
* docker-compose (ou tout variante, ex : docker-compose-stack.yml)
* un fichier .env (qui sont utilisés pour être injectés dans des docker-compose ou dans des dockerFiles)
* un Dockerfile

A partir de ce script, on peut extraire la liste de variables d'environnements d'un projet (donc leur nombre).

L'élaboration de ce script a été incrémentale, au fur et à mesure que l'on appliquait notre démarche à un nouveau projet, ce script a été étendu et amélioré.

2. **Automatisation de la récupération de tout l'historique (commits) d'un projet**

3. **Script permettant de detecter le/les commits qui ont manipulé une variable d'environnement donnée**

4. **Script permettant de récupérer le commit ayant manipulé une variable d'environnement, et le précédent**

5. **Script permettant de calculer la complexité de certains fichiers données entre deux commits**

## VII. Menaces à la validité

1. Peu de projets analysés

  Durant notre étude, nous avons étudié un nombre faible de projets, qui est insuffisant pour pouvoir valider notre hypothèse. Ces projets étaient également différents, certains s'interfaçaient avec de nombreux composants externes (MYSQL, Kafka, etc...) tandis que d'autres non. 

2. Peu de données

  Au cours de nos recherches, nous avons pu chercher ou étaient utilisés les variables d'environnement, il en ressort que peu d'entre-elles sont utilisées dans le code. Nous n'avons par exemple pas trouvé beaucoup d'utilisations concrètes ou la variable modifiait à proprement parler le comportement du code, en effet, la plupart des variables trouvées étaient des variables utilisées pour la configuration du projet (PORT, HOST, PASSWORD, USER ...)

  Voici par exemple la liste des variables présentes sur le projet Open-Mrs :

  [ 'DEBUG', 'MYSQL_DATABASE', 'MYSQL_ROOT_PASSWORD', 'MYSQL_USER',  'MYSQL_PASSWORD', 'DB_DATABASE', 'DB_HOST', 'DB_USERNAME',  'DB_PASSWORD', 'DB_CREATE_TABLES', 'DB_AUTO_UPDATE', 'MODULE_WEB_ADMIN', 'MYSQL_DEV_PORT', 'TOMCAT_DEV_PORT', 'TOMCAT_DEV_DEBUG_PORT',  'TOMCAT_PORT' ]

## VIII. Recul sur le projet

Tout d'abord, ce projet (par extension, cette matière) a été très enrichissant pour nous car il nous a permis de nous initier aux problématiques de recherche. 

En prenant un peu de recul sur la manière dont nous avons appréhendé le projet et la problématique, nous aurions pu diriger notre analyse non pas sur la qualité du code business, mais sur la qualité du code d'infrastructure.

En effet, nous mettons en avant la différence entre ces deux types de code car nous nous sommes aperçus que les configurations Docker ne peuvent avoir qu'un impact limité (et indirect) sur la qualité du code développé. 

En revanche, il aurait été intéressant d'évaluer **la qualité du code qui décrit l'infrastructure conteneurisée**, c'est-à-dire, essayer de mesurer la simplicité, la qualité des Dockerfiles, docker-compose, 
en établissant et utilisant des métriques spécifiques pour mesurer la qualité d'un code d'infrastructure. 

Globalement, l'idée serait de traduire chaque docker-compose par exemple en un graphe décrivant toutes les configurations Docker présentes et comment est-ce que ces derniers lient les conteneurs; à partir de ce graphe, nous pourrions réaliser des études sur sa forme et sa compléxité, en utilisant pourquoi pas du machine learning pour entraîner un modèle qui determinera la qualité d'un code d'infrastructure. 

## IX. References

1. https://dzone.com/articles/top-10-benefits-of-using-docker
1. https://12factor.net/fr/
