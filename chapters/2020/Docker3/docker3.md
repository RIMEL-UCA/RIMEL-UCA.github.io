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

Pour trouver les variables d'environnement dans le code, on filtre l'ensemble des commits pour trouver ceux dans lesquels on manipule les variables d'environnement. Ce filtre, on le fait en faisant une recherche avec la commande ```git log -S VARIABLE_ENV```. Une fois que nous avons les commits filtrés, on se positionne dessus et on filtre les fichiers en fonction de s'ils contiennent la variable d'environnement ou pas.

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

Pour prouver cette hypothèse, nous avons établi le protocole suivant :

Nous avons choisi dans un premier temps de tester d'appliquer notre démarche sur un seul projet (ThingsBoard). Par la suite, nous avons améliorer nos scripts de manières incrémentales afin qu'ils puissent s'appliquer aux différents projets. Enfin, nous avons automatisé notre protocole, afin de l'appliquer rapidement sur différents projets. 

**Objectif Step 1** :  Obtenir pour chaque VE le commit ou elle a été utilisée ou ajoutée et le commit précédent le commit en question

**Objectif Step 2** : Obtenir les fichiers ou les variables d'environnement ont été modifiées entre le commit et son précédent.

**Objectif Step 3** : Analyser (calculs de complexité) les fichiers au niveau de chaque commit pour pouvoir comparer si l'ajout des VE a un impact sur la complexité et le nombre de lignes.

**Objectif Step 4** : Récupérer les résultats des analyses et construire des statistiques (moyenne, delta avant/après) puis créer des visualisations. 

Le schéma ci-dessous décrit plus en détails les différentes étapes (automatisée) de notre démarche : 

 

![pipeline](https://image.noelshack.com/fichiers/2020/11/2/1583871891-floz-page-1-1.png)



Lien si l'image charge mal : [Protocol](https://image.noelshack.com/fichiers/2020/11/2/1583871891-floz-page-1-1.png)

Nous avons utilisé cette procédure qui consiste à étudier deux commits qui se suivent, afin de mieux cibler les changements de variables d'environnement lors de la comparaison. La granularité utilisée est au niveau des fichiers concernant les variables d'environnement et non pas l'ensemble des fichiers du commits. Cependant, notre script est incomplet, en effet des améliorations peuvent être réalisées, notamment sur la récupération de l'ensemble des variables d'environnements du projet. 

## VI. Présentations des résultats et de l'analyse

### Analyse des résultats

**A. OpenMrs**

Voici les résultats obtenus sur le projet OpenMrs. 

**NLOC**

L'image ci-dessous montre : 

- Colonne 1 : Les variables d'environnements trouvées

- Colonne 2 : Le nombre de ligne dans le commit précédent l'ajout de la variable

- Colonne 3 : Le nombre de ligne dans le commit d'ajout de la variable

- Colonne 4 : La différence de complexité entre les deux commits 

  

![Image](https://scontent-cdg2-1.xx.fbcdn.net/v/t1.15752-9/89108041_493365074900659_1770274503096532992_n.png?_nc_cat=102&_nc_sid=b96e70&_nc_ohc=OtCCxQiZOi0AX-HTZt-&_nc_ht=scontent-cdg2-1.xx&oh=1443e4254a574a4808997a56c6961ce5&oe=5E96F743)

![Image2](https://scontent-cdt1-1.xx.fbcdn.net/v/t1.15752-9/89023721_295291068112872_7188106946472312832_n.png?_nc_cat=103&_nc_sid=b96e70&_nc_ohc=pwCITRSZ4pEAX9ofCqJ&_nc_ht=scontent-cdt1-1.xx&oh=59c2690fcb4b84a793460fc16bb5200a&oe=5E8DB8BB)

**Interprétation** : 

​	Nous observons que plus on ajoute de variables d'environnement plus le nombre de ligne augmente. 



**Complexité cyclomatique** 

L'image ci-dessus montre : 

   - Colonne 1 : Les variables d'environnements trouvées
   - Colonne 2 : La complexité cyclomatique (CCN) dans le commit précédent l'ajout de la variable
   - Colonne 3 : La complexité cyclomatique (CCN) dans le commit d'ajout de la variable
   - Colonne 4 : La différence de complexité entre les deux commits 

   ![Image](https://scontent-cdt1-1.xx.fbcdn.net/v/t1.15752-9/88979522_2621295381441641_3173252310801317888_n.png?_nc_cat=106&_nc_sid=b96e70&_nc_ohc=l4MliccmlQwAX_83DjZ&_nc_ht=scontent-cdt1-1.xx&oh=d2effbfa389e358632d07f4d912f64ca&oe=5E95441D)

   ![Image2](https://scontent-cdt1-1.xx.fbcdn.net/v/t1.15752-9/89359283_1014915985575090_7539768633761726464_n.png?_nc_cat=103&_nc_sid=b96e70&_nc_ohc=RrFXG0Aj2q8AX-nyxUw&_nc_ht=scontent-cdt1-1.xx&oh=a3039b2dd37547c640cf6f9f0cdab429&oe=5E8E1735)



**Interprétation** : 

Nous observons que la complexité cyclomatique diminue sur plusieurs variables (MYSQL_USER, MYSQL_PASSWORD). 



**B. Apache Skywalking**


   **NLOC**

   ![Image](https://scontent-cdg2-1.xx.fbcdn.net/v/t1.15752-9/89178863_624830931670773_5962046804559134720_n.png?_nc_cat=110&_nc_sid=b96e70&_nc_ohc=0cOptgeu1yEAX_gWjus&_nc_ht=scontent-cdg2-1.xx&oh=660f87f1f5f8ade34858411a3fecaf97&oe=5EA5AE80)

   ![Image2](https://scontent-cdt1-1.xx.fbcdn.net/v/t1.15752-9/89024668_2476373432676814_8035139346954715136_n.png?_nc_cat=101&_nc_sid=b96e70&_nc_ohc=3v8DN2J4GoMAX9uHGai&_nc_ht=scontent-cdt1-1.xx&oh=966f8278ae501ed1e391078d59bd0472&oe=5E928B8A)

   **Interprétation** : 

   Nous observons, comme sur le projet précédent que plus on ajoute de variables d'environnement plus le nombre de ligne augmente.

   

   **Complexité cyclomatique** 

![Image](https://scontent-cdt1-1.xx.fbcdn.net/v/t1.15752-9/89384401_2557079704620209_5055113946898366464_n.png?_nc_cat=101&_nc_sid=b96e70&_nc_ohc=4yhvJKtkomYAX9Dn6Dl&_nc_ht=scontent-cdt1-1.xx&oh=6155a4e282687756f818e831a90fba1e&oe=5E8DFFFC)
    
![Image2](https://scontent-cdg2-1.xx.fbcdn.net/v/t1.15752-9/88357402_540906743205371_2543530115736797184_n.png?_nc_cat=102&_nc_sid=b96e70&_nc_ohc=KqH2ZFGUW78AX-cPi6f&_nc_ht=scontent-cdg2-1.xx&oh=ac873ee116ba692ed39644e54c336f86&oe=5EA50623)

   **Interprétation** : 

   Nous observons que la complexité cyclomatique a diminué lors de l'ajout de la variable TAG. La variable SW_STORAGE, n'a pas eu d'impact sur la complexité. Enfin, la variable COLLECTORS_SERVER a légèrement augmenté la complexité. 

### Conclusion de l'analyse

L'impact des variables d'environnement sur les fichiers est minime, en effet, à travers ces graphes, nous observons que la différence de complexité nloc et ccn est à hauteur de -1 à 1. Il aurait fallu mesurer l'impact au niveau de la fonction. 

Contrairement à ce que nous pensions, l'ajout de variable d'environnement augmente nloc au lieu de le diminuer. Cette augmentation pourrait être expliquée par l'ajout du code qui prend le dessus sur la diminution de la complexité induit par l'ajout de la variable d'environnement.

## VII. Menaces à la validité

1. Peu de projets analysés

  Durant notre étude, nous avons étudié un nombre faible de projets, qui est insuffisant pour pouvoir valider notre hypothèse. Ces projets étaient également différents, certains s'interfaçaient avec de nombreux composants externes (MYSQL, Kafka, etc...) tandis que d'autres non. 

2. Peu de données

  Au cours de nos recherches, nous avons pu chercher ou étaient utilisés les variables d'environnement, il en ressort que peu d'entre-elles sont utilisées dans le code. Nous n'avons par exemple pas trouvé beaucoup d'utilisations concrètes ou la variable modifiait à proprement parler le comportement du code, en effet, la plupart des variables trouvées étaient des variables utilisées pour la configuration du projet (PORT, HOST, PASSWORD, USER ...)

  Voici par exemple la liste des variables présentes sur le projet Open-Mrs :

  [ 'DEBUG', 'MYSQL_DATABASE', 'MYSQL_ROOT_PASSWORD', 'MYSQL_USER',  'MYSQL_PASSWORD', 'DB_DATABASE', 'DB_HOST', 'DB_USERNAME',  'DB_PASSWORD', 'DB_CREATE_TABLES', 'DB_AUTO_UPDATE', 'MODULE_WEB_ADMIN', 'MYSQL_DEV_PORT', 'TOMCAT_DEV_PORT', 'TOMCAT_DEV_DEBUG_PORT',  'TOMCAT_PORT' ]

3. L'ajout de code biaise le résultat

Dans le cadre de l'ajout d'une variable d'environnement, si du code (non lié à la variable) est ajouté parallèlement à l'ajout de la variable cela biaise le résultat. Malheureusement il s'agit de quelque chose que nous ne pouvons pas bien vérifier/cibler automatiquement. 

## VIII. Recul sur le projet

Tout d'abord, ce projet (par extension, cette matière) a été très enrichissant pour nous car il nous a permis de nous initier aux problématiques de recherche. Nous avons néanmoins pris beaucoup de temps pour trouver une démarche, en effet, en début de projet nous nous sommes lancés sur de nombreuses mauvaises pistes : 

* Comparaison de complexité par release
* Mauvais choix d'outils (Sonar)
* Réalisation de script existant déjà sur git (recherche par mots clés)

En prenant un peu de recul sur la manière dont nous avons appréhendé le projet et la problématique, nous aurions pu diriger notre analyse non pas sur la qualité du code business, mais sur la qualité du code d'infrastructure.

En effet, nous mettons en avant la différence entre ces deux types de code car nous nous sommes aperçus que les configurations Docker ne peuvent avoir qu'un impact limité (et indirect) sur la qualité du code développé. 

En revanche, il aurait été intéressant d'évaluer **la qualité du code qui décrit l'infrastructure conteneurisée**, c'est-à-dire, essayer de mesurer la simplicité, la qualité des Dockerfiles, docker-compose, 
en établissant et utilisant des métriques spécifiques pour mesurer la qualité d'un code d'infrastructure. 

Globalement, l'idée serait de traduire chaque docker-compose par exemple en un graphe décrivant toutes les configurations Docker présentes et comment est-ce que ces derniers lient les conteneurs; à partir de ce graphe, nous pourrions réaliser des études sur sa forme et sa complexité, en utilisant pourquoi pas du Machine Learning pour entraîner un modèle qui déterminera la qualité d'un code d'infrastructure. 



## IX. References

1. https://dzone.com/articles/top-10-benefits-of-using-docker
1. https://12factor.net/fr/
