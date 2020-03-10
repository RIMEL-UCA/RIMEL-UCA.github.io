# Les configurations Docker permettent-ils de rendre le code plus simple ?

## Authors

Nous sommes 4 étudiants en dernière année de Polytech Nice-Sophia (SI5), en Architecture Logicielle :

* Lazrak Sami &lt;sami.lazrak@etu.unice.fr&gt;
* Larabi Walid &lt;walid.larabi@etu.unice.fr&gt;
* Aourinmouche Soufiane &lt;soufiane.aourinmouche@etu.unice.fr&gt;
* Massol Damien &lt;damien.massol@etu.unice.fr&gt;

## I. Contexte de la recherche

Durant cette dernière décennie, la conteneurisation est devenue une pratique courante au sein du développement des logiciels, car il permet de simplifier une partie importante
du logiciel : le déploiement, l'isolement des composants, environnements homogènes, gestion du dimensionnement de l'infrastructure. 

Docker est devenue la référence principale au niveau des technologies de conteneurs, et elle est adoptée par d'autre outils open-sources comme Kubernetes qui fournissent des 
services de gestion des conteneurs. 

L'impact sur la simplicité du déploiement automatique et la gestion d'une infrastructure logicielle est indéniable. 

![Figure 1: Logo UCA](../assets/model/UCAlogoQlarge.png)

## II. Question générale

Dans ce contexte, nous pouvons nous demander si conteneuriser son logiciel simplifie également le développement du code d'un projet, sa qualité, et
la manière qu'ont les développeurs de l'appréhender.

En effet, l'histoire d'ING laisse supposer que l'adoption de Docker leur a permis de grandement fluidifier leur cycle de développement. 

La question sur l’utilisation des paramètres de haut niveau pour agir sur les logiciels conteneurisés est intéressante parce qu’elle tourne 
autour de thématiques importantes sur les bonnes pratiques du développement logiciel.
 
Dans le contexte de CI/CD par exemple, la configuration varie considérablement d'un déploiement à un autre, alors que le code non. 

Un autre exemple serait d’avoir une même base de code et des comportements différents (avec les features toggle, gestion de licences etc.)

Durant notre parcours, nous avons développé des applications qui stockent la configuration sous forme de constantes dans le code or il s’agit 
d’une violation de la méthodologie “Twelve Factor App” (Reference 2), qui nécessite une séparation stricte de la configuration du code; pour une meilleure maintenabilité, portabilité et résilience. 

Cette mauvaise pratique nous a coûté beaucoup de temps et d’efforts et nous a poussé à choisir ce sujet, pour nous informer et voir comment ces pratiques sont utilisées 
dans le monde industriel / projets open-source, la complexité ajoutée et l’intérêt réel que cela apporte.  

**Question générale : Comment les configurations haut niveau de Docker influent la simplicité du code ?**

Commençons par définir ce qu'est pour nous la simplicité : 

Un code simple est un code lisible, c'est-à-dire structuré; mais c'est aussi un code maintenable, donc flexible et extensible.

Afin de faciliter notre étude, nous avons restreint notre analyse en étudiant l'influence d'un seul type de configuration haut-niveau de Docker : les variables d'environnements; 
qui parmi toutes les autres façons de configurer des containers Docker (Volumes, Ports, Restart Policies, File Systems...), ont le plus de chance d'être utilisée dans le code, 
et donc d'influencer directement nos critères de simplicité.  

## III. Rassemblement d'informations
 
### 1 - Les projets à analyser

Afin de mener notre étude à bien, nous nous sommes mis à la recherche de jeu de données, c'est-à-dire de projets pour appliquer notre démarche.

Notre démarche consiste à explorer les projets open-source hébergés sur GitLab/Github (parmis les dépôts de code source les plus utilisés aujourd’hui).

Tout d'abord, il fallait qu'on ait des projets qui utilisent **Docker**, et que dans les fichiers de configurations Docker (Docker-compose ou Dockerfile) soient présents, 
et qu'il y ait un nombre suffisant de variable d'environnements initialisées dans ces fichiers.

Ensuite, d'autres critères de sélections sont entrés en jeu comme :
* l'historique qui doit être important
* la popularité (stars)
* les messages de commit qui doivent être clairs
* langage dont on connaît les conventions (JS, Python et Java) 

Ces critères nous permettront de comparer des projets de différentes complexités, ayant un certain besoin de développer sur plusieurs phases ou pas, ou d’avoir un système qui change de comportement. 

Les projets que l'on a retenu sont :
* ElasticSearch : <https://github.com/elastic/elasticsearch>
* Thingsboard : <https://github.com/thingsboard/thingsboard.git>
* Magma : <https://github.com/facebookincubator/magma>
* Apache Skywalking : <https://github.com/apache/skywalking>
* Openmrs-sdk : <https://github.com/openmrs/openmrs-sdk>

### III.2 - Les outils utilisés

* **Analyse de code**

Dans notre démarche, l'évaluation de la compléxité du code est une partie importante, pour nous permettre d'évaluer la simplicité d'un dépôt de code.

Il nous fallait donc un outil pour évaluer la compléxité cyclomatique performant, car les projets selectionnées contiennent des milliers de commits et de fichiers.

Nous sommes partis sur un outil que toutes les personnes de notre groupe connaissaient, et avaient déjà utilisé : SonarQube, qui permet plus généralement de mesurer la qualité du code en continu.

Le problème que l'on a eu avec Sonar est que pour effectuer ses mesures, on a besoin de builder au préalable les projets sur lesquels nous souhaitons faire notre analyse.

Et vu que les projets que l'on a selectionnés sont plutôt lourds et long à builder, nous n'avons pas réussi à réaliser cette étape préliminaire pour simplement évaluer la compléxité de notre projet.

Nous avons décidé de nous lancer à la recherche d'un outil qui nous permettait d'avoir ces métriques, sans nous obliger à avoir buildé le projet avant (un analyseur de code statique). 

A l'issue de cette recherche, nous avons décidé d'utiliser Lizard, qui nous permet de récupérer des métriques intéressantes pour notre étude (surtout la compléxité cyclomatique), à l'échelle du code, 
du fichier et de la méthode. 

Au niveau de la performance, Lizard peut analyser un ensemble d'environ 1000 fichiers en environ 2 minutes, et ressortir un rapport riche sur les métriques de cet ensemble.

* **Recherche dans le code**

Lib Git Python
Walid s'exprimera ...

* **Recherche de commits**

API Github

* **GitJSON**

Damien, Soufiane s'exprimera ...

## IV. Hypothèse et Démarche

Avec la réduction de scope décrite précédemment, l'hypothèse que nous allons essayer de vérifier est la suivante :

**_Hypothèse_ : L'ajout de variables d'environnements rend le code moins complexe** 

Expliquer pourquoi on choisit cette hypothèse ...

Pour prouver cette hypothèse, nous avons établi le protocole suivant :

Description du protocole ... Etablir le lien avec les outils utilisés (et comment). Justifier en quoi ce protocole permettra de vérifier 
ou pas l'hypothèse.


## V. Présentations des résultats et de l'analyse

1. Analyse des résultats & construction d’une conclusion : Une fois votre expérience terminée, vous récupérez vos mesures et vous les analysez pour voir si votre hypothèse tient la route. 
2. Conclusion de l'analyse

## VI. Description de la partie automatisée

1. **Automatisation de la détection des variables d'environnement**

Nous avons choisi dans un premier temps de tester d'appliquer notre démarche sur un seul projet (ThingsBoard). La première chose à faire
a été de récupérer les variables d'environnement que le projet contient. 

Vu que cette opération devra être effectuée sur plusieurs projets (voir pourra nous aider à voir le nombre de variables d'environnements
dans un projet), nous avons décidé de l'automatiser.
 
Nous avons automatisée cette partie, qui nous permet de détecter les variables d'environnements d'un projet, en supposant qu'elles soient localisées dans un des fichiers suivants :
* docker-compose (ou tout variante, ex : docker-compose-stack.yml)
* un fichier .env (qui sont utilisés pour être injectés dans des docker-compose ou dans des dockerFiles)
* un DockerFile

A partir de ce script, on peut extraire la liste de variables d'environnements d'un projet (donc leur nombre).

2. 

## VII. Menaces à la validité

1. Peu de projets analysés
Durant notre étude, nous avons étudié un nombre faible de projets pour pouvoir appuyer 
2. Peu de données
Peu d'utilisation concrètes de variables d'environnement dans le code trouvées dans les projets
3. Plus de visualisation

## VIII. Recul sur le projet

Tout d'abord, ce projet (par extension, cette matière) a été très enrichissant pour nous car il nous a permis de nous initier
aux problématiques de recherche, et on a pu avoir un avant-goût de ce qui peut être réalisé durant
un doctorat par exemple.

En prenant un peu de recul sur la manière dont nous avons apréhender le projet et la problématique, nous aurions pu diriger notre analyse 
non pas sur la qualité du code business, mais sur la qualité du code d'infrastructure.

En effet, nous mettons en avant la différence entre ces deux types de code car nous nous sommes aperçus que les configurations Docker ne peuvent avoir qu'un impact limité (et indirect)
sur la qualité du code developpé. 

En revanche, il aurait été intéressant d'évaluer la qualité du code qui décrit l'infrastructure conteneurisée, c'est-à-dire, essayer de mesurer la simplicité, la qualité des Dockerfiles, docker-compose, 
en établissant et utilisant des métriques spécifiques pour mesurer la qualité d'un code d'infrastructure. 

Globalement, l'idée serait de traduire chaque docker-compose par exemple en un graphe décrivant toutes les configurations Docker présentes et comment est-ce que ces derniers lient les conteneurs;
et à partir de ce graphe, réaliser des études sur sa forme et sa compléxité, en utilisant pourquoi pas du machine learning pour entraîner un modèle qui determinera la qualité d'un code d'infrastructure. 

## IX. References

1. https://dzone.com/articles/top-10-benefits-of-using-docker
1. https://12factor.net/fr/
