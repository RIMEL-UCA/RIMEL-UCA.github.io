---
layout: default
title : Qualité logicielle dans les notebooks Jupyter
date:   2022-11
---

**_3 février 2023_**

## Auteurs

Nous sommes quatre étudiants ingénieurs en dernière année à Polytech Nice Sophia, spécialisés en Architecture Logicielle :

- Laurie Fernandez ([@Laurie-Fernandez](https://github.com/Laurie-Fernandez)),
- Emma Glesser ([@Emma-Glesser](https://github.com/Emma-Glesser)),
- Arthur Soens ([@Arthur-Soens](https://github.com/Arthur-Soens)),
- Vincent Turel ([@Vincent-Turel](https://github.com/Vincent-Turel/)).


## I. Contexte de recherche

Le sujet d'étude de la qualité logicielle dans les notebooks Jupyter nous a intéressé de part des expériences personnelles passées avec ce type de projet. 

Plusieurs points nous ont poussé à nous poser des questions sur la qualité du code des notebooks Jupyter. Dans de précédents projets collaboratifs, les notebooks Jupyter ont dû être utilisés.
Lors de son stage de 4ème année, un des membres de l'équipe a dut convertir un notebook jupyter d’IA en code Python dans le but d’une démonstration destinée à un client. Il a donc eu à récupérer tous les bouts de code du notebook utilisé et les assembler en un script Python tout en le rendant lisible, en optimisant les imports et en commentant du code. Lors de cette expérience, il a été confronté à un code qui n’avait pas été écrit par un développeur mais un data scientist. Les noms des variables et des fonctions n’étaient pas explicites, de plus il manquait des commentaires essentiels à la compréhension de chaque partie du programme. N’ayant pas participé à la création du notebook, il a donc eu beaucoup de difficultés à le comprendre, le reprendre et le convertir en quelque chose d’acceptable pour un développeur. 

Ce sujet est intéressant à aborder car il nous permettrait de savoir sur quels critères de qualité logicielle mesurer la qualité du code d'un notebook Jupyter. Ce sujet n'étant que peu voire pas abordé la question reste assez nébuleuse. On pourrait ainsi savoir quels sont les forces et faiblesses des codes des notebooks.

Pour que notre étude ne devienne pas trop complexe, nous ne considèrerons que des notebooks reproductibles dont le code build.


## II. Observations & Question générale

Nous allons nous demander dans cette étude comment mesure-t-on la qualité d’un notebook ? Nous en avons déduit deux sous-questions :
Afin de compléter notre étude et pouvoir mieux répondre à cette question nous allons chercher à répondre aux deux sous-questions suivantes :
* Quelles métriques de qualité de code sont pertinentes dans le cas d'un notebook Jupyter ?
* Comment reconnaître une métrique pertinente ? 


## III. Collecte d'informations
### Articles
Pour notre recherche, nous allons nous baser sur les quatre articles suivants : 
1. [Eliciting Best Practices for Collaboration with Computational Notebooks](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/28337085-da8f-41ef-a9af-7070497bd728/Quaranta2022.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230109%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230109T131206Z&X-Amz-Expires=86400&X-Amz-Signature=4bf01bf31aee79119238f238c1b4efac8cda2b7c660ea73c0bf924f43c494301&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Quaranta2022.pdf%22&x-id=GetObject) 
   
    Auteurs : 
    * Luigi QUARANTA, *University of Bari, Italy*
    * Fabio CALEFATO, *University of Bari, Italy*
    * Filippo LANUBILE, *University of Bari, Italy*
    
    Date de publication : Avril 2022
    
    Ce qui nous est montré par cet article : Dans cet article, une revue de littérature multivocale a été effectuée afin de construire un catalogue de 17 meilleures pratiques pour la collaboration avec les notebooks. Ensuite, sont évalués qualitativement et quantitativement la mesure dans laquelle les experts en sciences des données les connaissent et les respectent. Dans l'ensemble, cet article permet de constater que les scientifiques spécialistes des données connaissent les meilleures pratiques identifiées et ont tendance à les adopter dans leur routine de travail quotidienne. 
   Néanmoins, certains d'entre eux ont des opinions contradictoires sur des recommandations spécifiques, en raison de facteurs contextuels variables. En outre, nous avons constaté que les meilleures pratiques les plus négligées ont trait à la modularisation du code, aux tests de code et au respect des normes de codage.
   Cela met en évidence ce qui semblerait être une tension entre la vitesse et la qualité.
   Pour résoudre cette tension, les plateformes de notebooks devraient mettre en œuvre des fonctionnalités natives qui facilitent l'adoption des pratiques de collaboration susmentionnées. En disposant de cadres de test et de linting intégrés et spécifiques aux blocs-notes, ainsi que de fonctions de refactoring et de modularisation du code des blocs-notes, nous pouvons nous attendre à ce que les blocs-notes de données soient plus faciles à utiliser, nous pouvons espérer que les scientifiques des données écrivent des notebooks de haute qualité sans compromettre
   sur la rapidité d'exécution. 

2. [What’s Wrong with Computational Notebooks? Pain Points, Needs, and Design Opportunities](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/8ae72282-8712-4a68-9dc6-6b00ebd7ecc2/Chattopadhyay2020.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230109%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230109T152624Z&X-Amz-Expires=86400&X-Amz-Signature=28a23fcdfdaca9ae48c01c82bbadc3b5f8c5b2dcb7185c2129b0b2e99c4f65b3&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Chattopadhyay2020.pdf%22&x-id=GetObject)
   
    Auteurs : 
    * Souti CHATTOPADHYAY, *Oregon State University*
    * Ishita PRASAD, *Microsoft*
    * Austin Z. HENLEY, *University of Tennessee-Knoxville* 
    * Anita SARMA, *Oregon State University*
    * Titus BARIK, *Microsoft*
    
    Date de publication : 25 au 30 avril 2020
    
    Ce qui nous est montré par cet article : Dans cet article, une étude à méthode mixte a été menée avec des scientifiques spécialisés dans les données en utilisant des observations sur le terrain, des entretiens et une enquête. Lors des études sur le terrain et des entretiens, les spécialistes des données ont signalé diverses difficultés lorsqu'ils travaillaient avec des ordinateurs portables et synthétisé ces difficultés dans une classification des points posant problème dans le code.
   Nous avons validé les activités difficiles qui contribuent à ces points posant problème dans le code par le biais d'une enquête et nous avons constaté que le soutien de toutes les activités étaient au moins modérément importantes pour les scientifiques spécialistes des données, et que quatre activités - la refonte du code, le déploiement en production, la gestion et l'utilisation de l'historique, et l'exécution de tâches de longue durée - étaient à la fois difficiles et importantes, ce qui en fait des activités à fort impact. Nos résultats suggèrent plusieurs possibilités de conception pour les chercheurs et les développeurs de notebooks. La résolution de ces problèmes peut améliorer considérablement l'utilité, la productivité et l'expérience utilisateur des scientifiques qui travaillent avec des notebooks.


3. [Ten simple rules for writing and sharing computational analyses in Jupyter Notebook](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/4709bfe2-0ac6-4dac-aaaa-b64063ca688c/Rule2019.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230109%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230109T153113Z&X-Amz-Expires=86400&X-Amz-Signature=9d4bc9f5c3d2b85884e0673e1512dad09aa007390f71429f27db5b47294bf0ca&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Rule2019.pdf%22&x-id=GetObject). 
   
    Auteurs : 
    * Adam RULE, *Design Lab, UC San Diego*
    * Amanda BIRMINGHAM, *Center for Computaional Biology and Bioinformatics, UC San Diego*
    * Cristal ZUNIGA, *Department of Pediatrics, UC San Diego*
    * Ilkay ALTINTAS, *Data Science Hub, San Diego Supercompter Center, UC San Diego*
    * Shih-Cheng HUANG, *Data Science Hub, San Diego Supercompter Center, UC San Diego*
    * Rob KNIGHT,  *Department of Pediatrics and Departments of Bioengineering, UC San Diego*
    * Niema MOSHIRI, *Bioinformatics and Systems Biology Graduate Program, UC San Diego*
    * Mai H. NGUYEN, *Data Science Hub, San Diego Supercompter Center, UC San Diego*
    * Sara Brin ROSENTHAL, *Center for Computaional Biology and Bioinformatics, UC San Diego*
    * Fernando PÉREZ, *UC Berkeley and Lawrence Berkeley National Laboratory*
    * Peter W. Rose, *UC San Diego*
    
    Date de publication : 25 juillet 2019
    
    Ce qui nous est montré par cet article : Des analyses robustes et reproductibles sont au cœur de la science, et plusieurs articles ont déjà fourni d'excellents conseils généraux sur la manière de réaliser et de documenter la science informatique. Cependant, l'avènement des notebooks informatiques présente de nouvelles opportunités et de nouveaux défis, facilitant la documentation précise de flux de travail complexes tout en la compliquant par l'interactivité. Nous présentons 10 règles simples pour écrire et partager des analyses dans des carnets Jupyter, en nous concentrant sur l'annotation de l'analyse, l'organisation du code, et la facilité d'accès et de réutilisation. Informés par notre expérience, nous espérons qu'ils contribueront à l'écosystème des individus, des laboratoires, des éditeurs et des organisations qui utilisent les notebooks pour performer et partager leurs recherches informatiques.

4. [Managing Messes in Computational Notebooks](https://lms.univ-cotedazur.fr/2022/pluginfile.php/399461/mod_folder/content/0/Head2019.pdf?forcedownload=1) 
   
    Auteurs : 
    * Andrew HEAD, *UC Berkeley*
    * Fred HOHMAN, *Georgia Institute of Technology*
    * Titus BARIK, *Microsoft*
    * Steven M. DRUCKER, *Microsoft Research*
    * Robert De LINE, *Microsoft Research*
    
    Date de publication : 4 au 9 mai 2019
    
    Ce qui nous est montré par cet article : L'étude qualitative menée de scientifiques a confirmé que le nettoyage des notebooks consiste principalement à supprimer le code d'analyse indésirable et les résultats indésirables. La tâche de nettoyage peut également comporter des étapes secondaires telles que l'amélioration de la qualité du code, la rédaction de la documentation, l'adaptation de la présentation des résultats pour un nouveau public cible, ou la création de scripts. Les participants considèrent que la tâche principale de nettoyage est un travail de bureau et qu'elle est sujette à des erreurs.
    Ils ont donc réagi positivement aux outils de collecte de code qui produisent automatiquement le code minimal nécessaire pour reproduire un ensemble choisi de résultats d'analyse, en utilisant une nouvelle application de découpage de programme. Les analystes ont principalement utilisé la collecte de code en dernière action pour partager leur travail, mais y ont aussi trouvé des utilisations inattendues comme la génération de matériel dede référence, la création de branches plus légères dans leur code, et créer des résumés pour des publics multiples.
    
### Outils
Les outils que nous envisageons d'utiliser pour effectuer notre étude sont :
* SonarQube : logiciel libre de qualimétrie en continu de code. Il aide à la détection, la classification et la résolution de défaut dans le code source, permet d'identifier les duplications de code, de mesurer le niveau de documentation et connaître la couverture de test déployée.
* Bandit : outil d'analyse des vulnérabilités de sécurité Python qui analyse les packages Python à la recherche de failles de sécurité. C'est un outil qui permet de créer un code conforme aux normes organisationnelles et qui génère un rapport de vulnérabilité de sécurité avec des informations détaillées sur le problème de sécurité.
* Code Climate : outil d'analyse proposant deux produits : le premier identifiant les failles logiques et les mauvais modèles de conception dans le code se concentrant sur l'amélioration de la qualité fonctionnelle du code et le second axé sur la qualité du code en termes de formatage, d'importations inutilisées, de variables et de couverture des tests unitaires garantissant la qualité du code avant la fusion.
* RATS : “Rough-Auditing-Tool-for-Security”, outil d’analyse de plusieurs langages dont Python pouvant signaler les erreurs de code courantes liées à la sécurité, au buffer overflow et runtime TOCTOU.
* PyLint : linter vérificateur de code permettant de signaler les erreurs Python ainsi que toutes les parties de code qui ne respectent pas un ensemble de conventions prédéfinies. Il permet donc d’éviter les erreurs et d’avoir un code homogène. Les règles qu’impose Pylint par défaut suivent le guide de style Python PEP 8.

### Jeu de données
Les codes que nous analyserons pour notre études seront des projets de différentes tailles et auteurs proposées sur GitHub classés par nombre de visites avec pour choix un sujet commun : la réalité augmentée. Un script de récupération automatisé de projets de type notebook Jupyter (avec extension .ipynb) sera utilisé pour avoir une diversité de projets. Nous récupérons un fichier par page de projets publics GitHub. Ils seront ensuite transformés en code Python de qualité mesurable grâce aux outils cités dans la partie précédente par ce même script. Ce choix a été fait de par le fait que la majorité des notebooks Jupyter sont codés dans ce même langage et que cela n'entrainera donc pas de transformation du code. A ces projets récupérés, nous ajouterons des projets de réalité augmentée effectués en cours à Polytech Nice-Sophia lors de nos études.

## IV. Hypothèses & Expériences
### Hypothèses de travail 

Nous avons émis plusieurs hypothèses au début de notre étude :
* Le code analysé est la somme de toutes les cellules du notebook
* La couverture de tests ne fait pas partie des métriques que nous pensons étudier compte tenu qu’il n’y a pas de tests unitaires du code dans les notebooks
* On part du principe que les repositories notebooks Microsoft et Jupyter sont de bonne qualité car il s’agit de références et cours. De plus, l’image de marque portée par Microsoft nous laisse penser que les projets qu’ils proposent sont revus avant d’être publiés. Nous nous servirons donc de ces projets pour effectuer une moyenne sur les métriques trouvées afin de mettre en place une échelle de mesure de la bonne qualité du code de nos notebooks et classerons les projets analysés suivant ces mesures référentes.

### Limites de notre étude 

Au vu du temps imparti pour mener cette étude, nous sommes bien sûr obliger de faire des expériences avec certaines limitations au niveau de l'analyse des résultats. 

Nous ne connaissons pas précisément les critères utilisés par les outils que nous utilisons. Nous ne maitrisons par exemple pas la pondération des différentes erreurs dans Pylint et l'analyse faites par SonarQube s'oriente plus sur l'analyse de la POO. Nous ferons tout ce qu'il est possible pour essayer d'obtenir des résultats objectifs mais l'idéal serait bien sûr de définir beaucoup plus précisément quels sont nps critères que qualité dans le cas spécifique des notebooks Jupyter. 

De plus, nous avons décidé de récupérer un notebook par page de résultat GitHub ce qui est tout de même plutôt bien mais cela reste peut-être une quantité un peu faible pour représenter correctement la qualité moyenne des notebooks. Nous aurions pu prendre plusieurs notebooks par page ou encore aller chercher des notebooks sur d'autres plateformes et notamment Kaggle pour que note jeu de donnée sont complètement représentatif de la moyenne des notebooks qui sont écrits.

### Expériences

Préalablement à la mise en place de nos expériences, nous avons eu à récupérer un ensemble de notebooks Jupyter. Le code des fichiers .ipynb récupérés est par la suite transformé en un unique code, somme de toutes les cellules du notebook, via passage dans notre script. L'extension de ce fichier est celle du langage de programmation utilisé dans le notebook. Le script filtre ensuite les fichiers obtenus afin de ne garder que ceux dont l'extension est celle des codes Python. 

#### Expérience 1 : Evaluation de la qualité du code Python des notebooks par SonarQube

La démarche que nous avons prévu pour l’analyse de ce sujet se découpe en quatre étapes : 
1. Lancement de notre script passant nos fichiers Python récupérés auparavant sur SonarQube
2. Récupération de toutes les analyses sur les métriques choisies par SonarQube
3. Exploitation des résultats, et voir si ceux-ci pourraient mener à des hypothèses supplémentaires ou conclusions intéressantes dont les métriques pertinentes dans l'analyse des notebooks
4. Classification des notebooks de notre jeu de données d'après les notes référentes récupérées d'après notre hypothèse de départ de qualité des notebooks

SonarQube basera ses analyses du code sur sept critères :
* Si l'analyse a pu être menée à terme
* Le nombre de bugs 
* Le nombre de vulnérabilités de code 
* Le nombre de hotspots 
* Le nombre de code smells
* Le pourcentage de code couvert par des tests
* Le pourcentage de code dupliqué

Il donnera a chacun de ces critères numériques une note allant de A à F permettant de juger de sa qualité.

#### Expérience 2 : Evaluation de la qualité du code Python des notebooks par CodeClimate

Cette première démarche terminée, nous avons décidé d'utiliser CodeClimate pour analyser nos notebooks. 
CodeClimate permet d'analyser des projets Python et de leur attribuer une note de qualité décrémentale en fonction du nombre d'erreurs, d'exceptions et de non respect de critères de qualité du code tels que les emplacements de déclaration d'imports ou des redéfinitions de variables.
Malheureusement, l'application de CodeClimate à nos notebooks de référence cause des erreurs, nous sommes donc à la recherche d'autres solutions d'analyse des projets Noteboook et de la qualité du code Python. 

#### Expérience 3 : Evaluation de la qualité du code Python des notebooks par PyLint

Après des recherches plus approfondies, CodeClimate propose l'équivalent d'un PyLint Python, c'est-à-dire un outil qui analyse votre code sans l'exécuter. Il vérifie les erreurs, applique une norme de codage, recherche les odeurs de code et peut faire des suggestions sur la façon dont le code pourrait être remanié. Pylint peut déduire les valeurs réelles de votre code en utilisant sa représentation interne du code. Une première étape serait l'utilisation du plugin PyLint simple sur nos notebooks de référence pour analyse de leur qualité, une note entre 0 et 10 leur serait alors attribuée.
Après une première utilisation de PyLint nous nous sommes aperçus que ses configurations de base ne sont pas forcément pertinentes dans le cas des notebooks. Nous avons donc écrit un script de configuration de PyLint afin d'exclure de ses vérification la liste d'exceptions suivante :
* *bad-inline-exception* 
* *locally-disabled*
* *file-ignored*
* *suppressed-image*
* *useless-suppression*
* *deprecated-pragma*
* *use-symbolic-message-instead*
* *too-many-lines*
* *trailing-whitespace*
* *line-too-long*
* *import-error*
* *too-few-public-methods*

Ces erreurs sont récurrentes mais en vérité peu pertinentes dans le cas des notebooks.

Les autres outils envisagés n'ont pas été utilisés car ils reprenaient des métriques déjà mesurées au cours de nos autres expériences ou se concentraient au final trop sur l'aspect sécuritaire du code qui est sûrement celui-ci qui a le moins d'intérêt pour nos scientifiques spécialistes des données.

## V. Analyse des résultats obtenus et Conclusion   
### Analyse de résultats obtenus
#### Expérience 1 : Evaluation de la qualité du code Python des notebooks par SonarQube
Comme expliqué précédemment, SonarQube nous permet de donner une note de qualité aux projets évalués. Ci-dessous, des résultats observés sur les projets proposés par Microsoft :

![Figure 1: Bons projets Sonar](images/sonar-good-quality.png)

Comme nous le pressentions, les projets Microsoft reçoivent de bonnes notes lors de leur analyse par SonarQube ce qui viendrait appuyer notre hypothèse de départ selon laquelle les projets Microsoft sont de bonne qualité.
Tous les projets ne sont pas classés comme étant de bonne qualité par SonarQube ce qui démontre une réelle application des critères de classement de cet outil, on peut ainsi voir ci-dessous que parmi les projets que nous avons sélectionné, les scores les plus bas obtenus sont des C.

![Figure 2: Mauvais projets Sonar](images/sonar-bad-quality.png)

Nous pouvons toutefois noter plusieurs choses : 
* Aucune vulnérabilité du code n'est jamais recensée lors de notre analyse
* Le taux de couverture de tests est systématiquement de 0%
* Le taux de duplication du code est également toujours de 0%

##### Vulnérabilités du code
Comme dans tout code informatique nous nous attendons à trouver des vulnérabilités dans le code, d'autant plus que certains des projets étudiés sont des labs écrits par des étudiants, il est donc étonnant de noter que l'ensemble de nos projets obtiennent la note maximale pour ce critère car ne possèdent aucune vulnérabilité.

##### Taux de couverture de tests
Comme formulé en hypothèse, le taux de couverture de tests retrouvé est nul. Nous pouvons donc à la fois valider nothre hypothèse de départ et écarter ce critère des métriques de qualimétrie de nos notebooks. Ajoutons à cela que les notebooks sont des types de projets généralement utilisés par des scientifiques spécialistes de données et que les tests de code ne sont pas la priorité ni le but de ces projets.

##### Taux de duplication du code
Voici le résultat à une métrique le plus étonnant, le taux de duplication du code. Il est connu que les notebooks contiennent des lignes dupliquées, le fait de n'en trouver aucune lors de l'analyse par SonarQube parait donc suspect car l'on devrait au moins tomber sur des déclarations multiples de mêmes variables et ainsi des duplications de code. On peut donc penser que cette

#### Expérience 2 : Evaluation de la qualité du code Python des notebooks par CodeClimate
Pour cette seconde expérience nous avons voulu mesurer la qualité du code de nos notebooks à l'aide de l'outil CodeClimate afin d'avoir des résultats mais aussi des métriques complémentaire à ce que nous avions pu trouver avec l'outil SonarQube. 
Malheureusement, de l'intégration à notre script d'évaluation de la qualité de l'outil CodeClimate n'a résulté que l'erreur suivante ne nous permettant donc par d'analyser nos projets.

![Figure 3: Erreur CodeClimate](images/codeclimate-error.png)

#### Expérience 3 : Evaluation de la qualité du code Python des notebooks par PyLint
En étudiant de plus près CodeClimate, nous avons pu nous rendre compte que pour une analyse de code Python, l'outil s'appuyait en fait sur le linter PyLint pour évaluer la qualité de code. Nous avons donc tenté d'utiliser PyLint sur nos projets mais les résultats obtenus étaient peu concluants. En effet, de nombreux codes certes de faible qualité mais compilant malgré tout obtenaient la note de 0 sur 10. Nous nous sommes alors posés la question des critères d'évaluations appliqués par PyLint et de leur pertinence dans le cas de notebooks Jupyter. 
Ainsi, nous avons pu écarter la liste de critères de notation suivants : 

![Figure 4: Configuration Pylint](images/pylint-config.png)

Les 7 premiers critères ont été placés directement par Pylint et il semble cohérent de les laisser, d'autant plus quand on regarde en détail leur signification. Si on regarde ensuite les critères que nous avons retirés nous-mêmes, on trouve plusieurs critères que nous ne jugeons pas révélateurs de la qualité d'un fichier python en général tels que comme *trailing-whitespace* ou encore *too-many-lines*. Nous avons ensuite retiré certains critères qui ne correspondent pas au cas des notebooks Jupyter comme *too-few-public-methods* et *too-many-lines*. Enfin, nous avons retiré import-error pour des raisons pratiques car nous ne pouvons pas installer tout les packages python avant chaque analyse dans le cadre de plusieurs centaines de fichiers. Il semble de toute manière peu probable que des mauvais imports soit fait puisque cela empêcherait le code de exécuter.	

Cela à ainsi à titre d'exemple fait évoluer la note du projet *NLP_C1_W1_lecture_nb_01* de 3.60 à 8.27 et cela se confirme globalement sur tous nos notebooks même si on retrouve toujours de très mauvaises notes. Cela nous amène à penser que évaluation s'affine et deviens un peu plus juste.

![Figure 5: Note projet avant configuration Pylint](images/pylint-before-config.png)

A la note suivante après application des configurations à désactiver.

![Figure 6: Note projet après configuration Pylint](images/pylint-after-config.png)

Une fois nos critères de notations précisés, il nous est donc resté à analyser les erreurs les plus courantes dans les notebooks analysés afin de déterminer de potentielles métriques de qualité de notre code. Ainsi, après analyse de 649 notebooks, les vingt erreurs les plus rencontrées lors de notre analyse étaient les suivantes :

![Figure 7: 20 erreurs les plus retrouvées par Pylint](images/pylint-top-20-errors.png)

On remarque que les deux erreurs les plus récurrentes concernent un problème de nommage de variable. C'est une erreur qui sera probablement commise par des personne peu adeptes des principes de qualité. Ce problème de nommage n'est pas une erreur importante car il ne va pas empêcher la compréhension ni l'exécution d'un notebook. En revanche, si le problème est étendu à tout le fichier, on pourrait perdre le lecteur dans la compréhension du notebook. Cette erreur est donc mineure et n'impactera pas énormement la note finale.  

Cependant, les erreurs suivantes concernent des variables inutilisées, des erreurs de syntaxe ou encore un surplus d'arguments pour une méthode. Ces erreurs sont plus lourdes et impacteront grandement la note car elles peuvent rendre un notebook inutilisable ou non-reproductible. On se rend assez vite compte que plus une erreur est impactante, moins elle est commise. C'est un résultat cohérent dans la mesure où une erreur qui rendrait un notebook non-reproductible est plus connue qu'une simple erreur d'inattention et donc plus corrigée.  

On a maintenant les preuves que PyLint est un indicateur de qualité pertinent dans l'évaluation d'un notebook. Certaines erreurs comme le nommage des variables ou la documentation des fonctions manquante pourraient former une première métrique de lisibilité. D'autres erreurs  comme le nombre trop élevé d'arguments dans une fonction ou une mauvaise syntaxe sont des erreurs plus lourdes et pourraient constituer des métriques d'erreurs d'exécution. On pourrait alors évaluer la qualité d'un notebook sur la base de sa lisibilité et de son exécution.


Nous avons ensuite voulu visualiser la répartition des notes des différents notebook dans le but de vérifier si les nombreuses erreurs repérées influaient sur la note des notebooks. Nous obtenons la figure suivante représentant le nombre de notebook par note (arrondie à 1 chiffre après la virgule) : 

![Figure 8: Répartition des notes de projets Pylint](images/pylint-nb-per-rate.png)

La répartition des notes sur notre schéma, même si elle est moins disparate que celle observée avec SonarQube, semble classer les précédents bons projets comme de bons projets. Nous avons toutefois une bien plus grande répartition des projets moyens qu'auparavant. On peut également remarquer une légère courbe de Gauss décalée vers la droite, on peut en déduire que les projets semblent tendre vers une qualité acceptable. 

Nous avons également voulu rassembler les projets par tranche de note afin de mieux visualiser la répartition des notes obtenues. Nous avons établi 4 sections :  
- [0 - 3] rassemblant les notebooks qu'on peut classer comme mauvais
- [3 - 5] rassemblant les notebooks qu'on peut classer comme moyen 
- [5 - 8] rassemblant les notebooks qu'on peut classer comme moyen mais meilleurs que les précédents
- [8 - 10] rassemblant les notebooks qu'on peut classer comme bons

![Figure 9: Répartition des notebook par tranche de notes](images/pylint-nb-ranges.png)

On retrouve la courbe de Gauss décalée vers la droite du précédent schéma qui semblent indiquer que la majorité des notebooks analysés se trouvent dans la tranche haute des notebooks moyens. 

Ces deux graphiques utilisant la note générée par PyLint montrent que la majorité des notebooks sont de qualité moyenne. Cela semble indiquer que PyLint est un bon outil pour mesurer la qualité d'un notebook.

Pour terminer, nous avons voulu vérifier notre hypothèse de bonne qualité des notebooks Microsoft avec Pylint. Voici les résultats obtenus pour les différents fichiers :

![Figure 10: Note projet Microsoft Pylint 1](images/microsoft-pylint-1.png)
![Figure 11: Note projet Microsoft Pylint 2](images/microsoft-pylint-2.png)
![Figure 12: Note projet Microsoft Pylint 3](images/microsoft-pylint-3.png)

Nous avons ainsi pu observer que nous obtenions pour ces notebooks deux notes plutôt moyennes et une bonne note. Ces résultats remettent en question notre hypothèse de départ de bonne qualité des notebooks Microsoft car il semblerait qu'ils soient au contraire de qualité plutôt moyenne. 

### Conclusion
???? A COMPLETER ????

## VI. Outils

Au niveau des outils de développement, nous avons fait un script python nommé *main.py*qui nous permet de gérer massivement toutes les étapes de notre analyse. Le main nous permet de commenter/décommenter les méthodes que nous souhaitons utiliser en modifiant les paramètres des méthodes. Il faut absolument exécuter ce script depuis le dossier dans lequel il est place sans quoi cela ne fonctionnera pas.

Pour ce qui est des [outils d'évaluation des métriques](#outils), nous avons dans un premier temps utilisé l'outil SonarQube. Nous avons deux container pour deux instances. Ils est possible de les lancer en faisant la commande *docker-compose up*. Nous avons mis en place des volumes qui permettent de conserver les analyses dans le temps. Les fichiers sont analysés en masse grâce au script python.

Au niveau Code climate, nous avons utiliser leurs outils de ligne de commande présent sur GitHub pour analyse nos fichiers python avec la commande *codeclimate analyze path_to_file*. Cela est également fait de manière automatique dans notre script.

Dans le cas de PyLint, nous utilisons simplement la commande *pylint path_to_file* mais cela est à nouveau fait de manière automatisée dans notre script.

Pour finir, nous avons un second script *parse_pylint_results.py* qui nous a permis de générer les graphiques présentés ci-dessus.

## VI. Références

[Articles étudiés](#articles) pour une meilleure compréhension des notebooks et du contexte dans lequel ils s'incrivent pour une utilisation par les scientifiques spécilistes des données :

* [Eliciting Best Practices for Collaboration with Computational Notebooks](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/28337085-da8f-41ef-a9af-7070497bd728/Quaranta2022.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230109%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230109T131206Z&X-Amz-Expires=86400&X-Amz-Signature=4bf01bf31aee79119238f238c1b4efac8cda2b7c660ea73c0bf924f43c494301&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Quaranta2022.pdf%22&x-id=GetObject) 

* [What’s Wrong with Computational Notebooks? Pain Points, Needs, and Design Opportunities](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/8ae72282-8712-4a68-9dc6-6b00ebd7ecc2/Chattopadhyay2020.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230109%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230109T152624Z&X-Amz-Expires=86400&X-Amz-Signature=28a23fcdfdaca9ae48c01c82bbadc3b5f8c5b2dcb7185c2129b0b2e99c4f65b3&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Chattopadhyay2020.pdf%22&x-id=GetObject)

* [Ten simple rules for writing and sharing computational analyses in Jupyter Notebook](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/4709bfe2-0ac6-4dac-aaaa-b64063ca688c/Rule2019.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230109%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230109T153113Z&X-Amz-Expires=86400&X-Amz-Signature=9d4bc9f5c3d2b85884e0673e1512dad09aa007390f71429f27db5b47294bf0ca&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Rule2019.pdf%22&x-id=GetObject). 

* [Managing Messes in Computational Notebooks](https://lms.univ-cotedazur.fr/2022/pluginfile.php/399461/mod_folder/content/0/Head2019.pdf?forcedownload=1)

  

![Figure 10: Logo UCA](images/logo_uca.png)
![Figure 11: Logo Polytech](images/logoPolytechUCA.png)
