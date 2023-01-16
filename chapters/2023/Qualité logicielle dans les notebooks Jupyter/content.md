---
layout: default
title : Qualité logicielle dans les notebooks Jupyter
date:   2022-11
---

---

   **Date de rendu finale : fin février**
   - Respecter la structure pour que les chapitres soient bien indépendants
   - Remarques :
        - La structure proposée est là pour vous aider, mais peut-être adaptée à votre projet
        - Les titres peuvent être modifiés pour être en adéquation avec votre étude.
        - Utiliser des références pour justifier votre argumentaire, vos choix, etc.
        - Vous avez le choix d'utiliser le français ou l'anglais.

    Dans l'article de Blog [Debret 2020], l'auteure donne les éléments principaux de la démarche d'une manière simple et très facile à lire, dans la partie [Quelles sont les étapes d’une bonne démarche scientifique ?](https://www.scribbr.fr/article-scientifique/demarche-scientifique/#:~:text=La%20d%C3%A9marche%20scientifique%20permet%20d,de%20nouvelles%20hypoth%C3%A8ses%20%C3%A0%20tester.)

---

**_3 février 2023_**

## Auteurs

Nous sommes quatre étudiants ingénieurs en dernière année à Polytech Nice Sophia, spécialisés en Architecture Logicielle :

- Laurie Fernandez ([@Laurie-Fernandez](https://github.com/Laurie-Fernandez)),
- Emma Glesser ([@Emma-Glesser](https://github.com/Emma-Glesser)),
- Arthur Soens ([@Arthur-Soens](https://github.com/Arthur-Soens)),
- Vincent Turel ([@Vincent-Turel](https://github.com/Vincent-Turel/)).


## I. Contexte de recherche

Préciser ici votre contexte et Pourquoi il est intéressant. **
Le sujet d'étude de la qualité logicielle dans les notebooks Jupyter nous a intéressé de part des expériences personnelles passées avec ce type de projet. 

Plusieurs points nous ont poussé à nous poser des questions sur la qualité du code des notebooks Jupyter. Dans de précédents projets collaboratifs, les notebooks Jupyter ont dû être utilisés.
Lors de son stage de 4ème année, un des membres de l'équipe a dut convertir un notebook jupyter d’IA en code Python dans le but d’une démonstration destinée à un client. Il a donc eu à récupérer tous les bouts de code du notebook utilisé et les assembler en un script Python tout en le rendant lisible, en optimisant les imports et en commentant du code. Lors de cette expérience, il a été confronté à un code qui n’avait pas été écrit par un développeur mais un data scientist. Les noms des variables et des fonctions n’étaient pas explicites, de plus il manquait des commentaires essentiels à la compréhension de chaque partie du programme. N’ayant pas participé à la création du notebook, il a donc eu beaucoup de difficultés à le comprendre, le reprendre et le convertir en quelque chose d’acceptable pour un développeur. 

Ce sujet est intéressant à aborder car il nous permettrait de savoir sur quels critères de qualité logicielle mesurer la qualité du code d'un notebook Jupyter. Ce sujet n'étant que peu voire pas abordé la question reste assez nébuleuse. On pourrait ainsi savoir quels sont les forces et faiblesses des codes des notebooks.

Pour que notre étude ne devienne pas trop complexe, nous ne considèrerons que des notebooks reproductibles dont le code build.


## II. Observations & Question générale

1. Commencez par formuler une question sur quelque chose que vous observez ou constatez ou encore une idée émergente.
Nous allons nous demander dans cette étude comment mesure-t-on la qualité d’un notebook et quelles métriques de qualité de code sont pertinentes dans le cas d’un notebook Jupyter ?

2. Préciser pourquoi cette question est intéressante de votre point de vue.

Attention pour répondre à cette question, vous devrez être capable d'émettre des hypothèses vérifiables, de quantifier vos réponses, ...

     :bulb: Cette première étape nécessite beaucoup de réflexion pour se définir la bonne question afin de poser les bonnes bases pour la suite.

## III. Collecte d'informations

Préciser vos zones de recherches en fonction de votre projet, les informations dont vous disposez, ... :

1. les articles ou documents utiles à votre projet
2. les outils
3. les jeux de données/codes que vous allez utiliser, pourquoi ceux-ci, ...

     :bulb: Cette étape est fortement liée à la suivante. Vous ne pouvez émettre d'hypothèses à vérifier que si vous avez les informations, inversement, vous cherchez à recueillir des informations en fonction de vos hypothèses.

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
Pour résoudre cette tension, les plateformes de carnets de calcul devraient mettre en œuvre des fonctionnalités natives qui facilitent l'adoption des pratiques de collaboration susmentionnées. En disposant de cadres de test et de linting intégrés et spécifiques aux blocs-notes, ainsi que de fonctions de refactoring et de modularisation du code des blocs-notes, nous pouvons nous attendre à ce que les blocs-notes de données soient plus faciles à utiliser, nous pouvons espérer que les scientifiques des données écrivent des notebooks de haute qualité sans compromettre
sur la rapidité d'exécution. 

2. [What’s Wrong with Computational Notebooks? Pain Points, Needs, and Design Opportunities](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/8ae72282-8712-4a68-9dc6-6b00ebd7ecc2/Chattopadhyay2020.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230109%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230109T152624Z&X-Amz-Expires=86400&X-Amz-Signature=28a23fcdfdaca9ae48c01c82bbadc3b5f8c5b2dcb7185c2129b0b2e99c4f65b3&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Chattopadhyay2020.pdf%22&x-id=GetObject)
    
    Auteurs : 
    * Souti CHATTOPADHYAY, *Oregon State University*
    * Ishita PRASAD, *Microsoft*
    * Austin Z. HENLEY, *University of Tennessee-Knoxville* 
    * Anita SARMA, *Oregon State University*
    * Titus BARIK, *Microsoft*
    
    Date de publication : 25 au 30 avril 2020
    
    Ce qui nous est montré par cet article :  Dans cet article, une étude à méthode mixte a été menée avec des scientifiques spécialisés dans les données en utilisant des observations sur le terrain, des entretiens et une enquête. Lors des études sur le terrain et des entretiens, les spécialistes des données ont signalé diverses difficultés lorsqu'ils travaillaient avec des ordinateurs portables et synthétisé ces difficultés dans une taxonomie des points de douleur.
Nous avons validé les activités difficiles qui contribuent à ces points de douleur par le biais d'une enquête et nous avons constaté que le soutien de toutes les
activités étaient au moins modérément importantes pour les scientifiques des données, et que quatre activités - la refonte du code, le déploiement en production, la gestion et l'utilisation de l'historique, et l'exécution de tâches de longue durée - étaient à la fois difficiles et importantes, ce qui en fait des activités à fort impact. Nos résultats suggèrent plusieurs possibilités de conception pour les chercheurs et les développeurs de notebooks. La résolution de ces problèmes peut améliorer considérablement l'utilité, la productivité et l'expérience utilisateur des scientifiques qui travaillent avec des notebooks.


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

### Jeux de données
Les codes que nous analyserons pour notre études seront des projets de différentes tailles et auteurs proposées sur Github avec pour choix un sujet commun : la réalité augmentée. Un script de récupération automatisé de projets de type notebook Jupyter (avec extension .ipynb) sera utilisé pour avoir une diversité de projets. Ils seront ensuite transformés en code Python de qualité mesurable grâce aux outils cités dans la partie précédente par ce même script. Ce choix a été fait de par le fait que la majorité des notebooks Jupyter sont codés dans ce même langage et que cela n'entrainera donc pas de transformation du code. A ces projets récupérés, nous ajouterons plusieurs projets de réalitée augmentée effectués en cours à Polytech Nice-Sophia lors de nos études.

## IV. Hypothèses & Expériences

1. Il s'agit ici d'**énoncer sous forme d'hypothèses** ce que vous allez chercher à démontrer. Vous devez définir vos hypothèses de façon à pouvoir les _mesurer/vérifier facilement._ Bien sûr, votre hypothèse devrait être construite de manière à _vous aider à répondre à votre question initiale_. Explicitez ces différents points.

Nous avons émis plusieurs hypothèses au début de notre étude :
* Le code analysé est la somme de toutes les cellules du notebook
* La couverture de tests ne fait pas partie des métriques que nous pensons étudier compte tenu qu’il n’y a pas de tests unitaires du code dans les notebooks
* On part du principe que les repositories notebooks Microsoft et Jupyter sont de bonne qualité car il s’agit de références et cours. De plus, l’image de marque portée par Microsoft nous laisse penser que les projets qu’ils proposent sont revus avant d’être publiés. Nous nous servirons donc de ces projets pour effectuer une moyenne sur les métriques trouvées afin de mettre en place une échelle de mesure de la bonne qualité du code de nos notebooks et classerons les projets analysés suivant ces mesures référentes.
* Nous pensons nous concentrer sur les notebooks ayant pour thème la réalité augmentée. Nous formons l’hypothèse que les notebooks de Microsoft étant plus spécialisés sur le sujet de réalité augmentée, ils seront de meilleure qualité que le projet de création de notebook proposé par Jupyter lui-même.

2. Vous **explicitez les expérimentations que vous allez mener** pour vérifier si vos hypothèses sont vraies ou fausses. Il y a forcément des choix, des limites, explicitez-les.
     :bulb: Structurez cette partie à votre convenance : Hypothèse 1 => Expériences, Hypothèse 2 => Expériences ou l'ensemble des hypothèses et les expériences....

La démarche que nous avons prévu pour l’analyse de ce sujet se découpe en six étapes : 
1. Récupérer plusieurs notebooks Jupyter en langage Python conséquents ou non dont nos propres projets parmi ceux présents sur Github
2. Analyser ces projets à travers Sonar et d’autres outils de qualimétrie cités auparavant 
3. Faire les modifications nécessaires si l’étape précédente ne fonctionne pas (à voir si cela possible)
4. Récupérer toutes les analyses sur les métriques choisies par Sonar
5. Exploiter les résultats, et voir si ceux-ci pourraient mener à des conclusions intéressantes dont les métriques pertinentes dans l'analyse des notebooks
6. Classer les notebooks récupérés d'après les notes référentes récupérées d'après notre hypothèse de départ de qualité des notebooks

Cette première démarche terminée, nous avons décidé d'utiliser CodeClimate pour analyser nos notebooks. Après des recherches plus approfondies, CodeClimate propose l'équivalent d'un PyLint Python, c'est-à-dire un outil qui analyse votre code sans l'exécuter. Il vérifie les erreurs, applique une norme de codage, recherche les odeurs de code et peut faire des suggestions sur la façon dont le code pourrait être remanié. Pylint peut déduire les valeurs réelles de votre code en utilisant sa représentation interne du code.
Malheureusement, l'application de CodeClimate à nos notebooks de référence cause des erreurs, nous sommes donc à la recherche d'autres solutions d'analyse des projets Noteboook et de la qualité du code Python. Une première étape serait l'utilisation du plugin PyLint simple sur nos notebooks de référence pour analyse de leur qualité.

## V. Analyse des résultats obtenus et Conclusion

1. Présentation des résultats
2. Interprétation/Analyse des résultats en fonction de vos hypothèses
3. Construction d’une conclusion

     :bulb:  Vos résultats et donc votre analyse sont nécessairement limités. Préciser bien ces limites : par exemple, jeux de données insuffisants, analyse réduite à quelques critères, dépendance aux projets analysés, ...

## VI. Outils \(facultatif\)

Précisez votre utilisation des outils ou les développements \(e.g. scripts\) réalisés pour atteindre vos objectifs. Ce chapitre doit viser à \(1\) pouvoir reproduire vos expérimentations, \(2\) partager/expliquer à d'autres l'usage des outils.

En plus des [outils](#outils) cités précédemment pour l'analyse de la qualité du code des notebooks, nous avons utilisé des scripts shell pour la récupération automatique de notebooks Jupyter sur Github et les invite de commande Windows et WSL.

![Figure 1: Logo UCA](images/logo_uca.png){:height="25px"}
![Figure 2: Logo Polytech](images/logoPolytechUCA.png){:height="25px"}


## VI. Références

[Debret 2020] Debret, J. (2020) La démarche scientifique : tout ce que vous devez savoir ! Available at: https://www.scribbr.fr/article-scientifique/demarche-scientifique/ (Accessed: 18 November 2022).


