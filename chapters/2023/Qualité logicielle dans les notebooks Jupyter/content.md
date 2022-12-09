---
layout: default
title : Qualié logicielle dans les notebooks Jupyter
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

**_février 2023_**

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
*** A COMPLETER AVEC LES EXPERIENCES PERSONNELLES ET PLUS PARTICULIEREMENT CELLE D'ARTHUR QUI SEMBLE TRES PERTINENTE DANS NOTRE CAS)

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
1. [Eliciting Best Practices for Collaboration with Computational Notebooks](https://s3.us-west-amazonaws.com/secure.notion-static.com/28337085-da8f-41ef-a9af-7070497bd728/Quaranta2022.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221209%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221209T175202Z&X-Amz-Expires=86400&X-Amz-Signature=13c5f03162c44048743a585207c6882688f2f946134e2d86dbef7c2da8268d1b&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Quaranta2022.pdf%22&x-id=GetObject). 
    
    Auteurs : 
    * Luigi QUARANTA, *University of Bari, Italy*
    * Fabio CALEFATO, *University of Bari, Italy*
    * Filippo LANUBILE, *University of Bari, Italy*
    
    Date de publication : Avril 2022
    
    Hypothèse : 

2. [What’s Wrong with Computational Notebooks? Pain Points, Needs, and Design Opportunities](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/8ae72282-8712-4a68-9dc6-6b00ebd7ecc2/Chattopadhyay2020.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221209%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221209T175805Z&X-Amz-Expires=86400&X-Amz-Signature=220ed3e43a86e520074fb22271a707ba5d6806312c80b0c99a5ebbf300bb9d6c&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Chattopadhyay2020.pdf%22&x-id=GetObject). 
    
    Auteurs : 
    * Souti CHATTOPADHYAY, *Oregon State University*
    * Ishita PRASAD, *Microsoft*
    * Austin Z. HENLEY, *University of Tennessee-Knoxville* 
    * Anita SARMA, *Oregon State University*
    * Titus BARIK, *Microsoft*
    
    Date de publication : 25 au 30 avril 2020
    
    Hypothèse : 

3. [Ten simple rules for writing and sharing computational analyses in Jupyter Notebook](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/4709bfe2-0ac6-4dac-aaaa-b64063ca688c/Rule2019.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221209%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221209T175915Z&X-Amz-Expires=86400&X-Amz-Signature=f9aa043bb454b0c802d2ab9420a9cd8bfb76ef9f080a8a3c459385c06a3fe143&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Rule2019.pdf%22&x-id=GetObject). 
    
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
    
    Hypothèse : 

4. [Managing Messes in Computational Notebooks](https://lms.univ-cotedazur.fr/2022/pluginfile.php/399461/mod_folder/content/0/Head2019.pdf?forcedownload=1). 
    
    Auteurs : 
    * Andrew HEAD, *UC Berkeley*
    * Fred HOHMAN, *Georgia Institute of Technology*
    * Titus BARIK, *Microsoft*
    * Steven M. DRUCKER, *Microsoft Research*
    * Robert De LINE, *Microsoft Research*
    
    Date de publication : 4 au 9 mai 2019
    
    Hypothèse : 
    
### Outils
Les outils que nous envisageons d'utiliser pour effectuer notre étude sont :
* SonarQube : logiciel libre de qualimétrie en continu de code. Il aide à la détection, la classification et la résolution de défaut dans le code source, permet d'identifier les duplications de code, de mesurer le niveau de documentation et connaître la couverture de test déployée.
* Bandit : outil d'analyse des vulnérabilités de sécurité Python qui analyse les packages Python à la recherche de failles de sécurité. C'est un outil qui permet de créer un code conforme aux normes organisationnelles et qui génère un rapport de vulnérabilité de sécurité avec des informations détaillées sur le problème de sécurité.
* Code Climate : outil d'analyse proposant deux produits : le premier identifiant les failles logiques et les mauvais modèles de conception dans le code se concentrant sur l'amélioration de la qualité fonctionnelle du code et le second axé sur la qualité du code en termes de formatage, d'importations inutilisées, de variables et de couverture des tests unitaires garantissant la qualité du code avant la fusion.
* RATS : “Rough-Auditing-Tool-for-Security”, outil d’analyse de plusieurs langages dont Python pouvant signaler les erreurs de code courantes liées à la sécurité, au buffer overflow et runtime TOCTOU.

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


## V. Analyse des résultats obtenus et Conclusion

1. Présentation des résultats
2. Interprétation/Analyse des résultats en fonction de vos hypothèses
3. Construction d’une conclusion

     :bulb:  Vos résultats et donc votre analyse sont nécessairement limités. Préciser bien ces limites : par exemple, jeux de données insuffisants, analyse réduite à quelques critères, dépendance aux projets analysés, ...

## VI. Outils \(facultatif\)

Précisez votre utilisation des outils ou les développements \(e.g. scripts\) réalisés pour atteindre vos objectifs. Ce chapitre doit viser à \(1\) pouvoir reproduire vos expérimentations, \(2\) partager/expliquer à d'autres l'usage des outils.

En plus des [outils](#outils) cités précédemment pour l'analyse de la qualité du code des notebooks, nous avons utilisé des scripts shell pour la récupération automatique de notebooks Jupyter sur Github.

![Figure 1: Logo UCA, exemple, vous pouvez l'enlever](images/logo_uca.png){:height="25px"}


## VI. Références

[Debret 2020] Debret, J. (2020) La démarche scientifique : tout ce que vous devez savoir ! Available at: https://www.scribbr.fr/article-scientifique/demarche-scientifique/ (Accessed: 18 November 2022).


