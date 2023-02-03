---
layout: default
title : Paternité de la variabilité dans un langage orienté objet (Java)
date:   2022-12
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

**_Février 2023_**

## Auteurs

Nous sommes 4 étudiants en dernière année du cursus ingénieur informatique de Polytech' Nice-Sophia spécialisés en Architecture Logiciel :

- Alexandre Arcil ([@Alexandre-Arcil](https://github.com/AlexandreArcil)),
- Mohamed Belhassen ([@Mohamed-Belhassen](https://github.com/mohamedlouay)),
- Thomas Di Grande ([@Thomas-Di-Grande](https://github.com/DigrandeArduino)),
- Dan Nakache ([@Dan-Nakache](https://github.com/danlux18)).

## I. Contexte de recherche/Projet

Préciser ici votre contexte et Pourquoi il est intéressant. **

Dans un monde où l'informatique est de plus en plus présent, de nombreux logiciels voient le jour.
Les entreprises réalisent des projets toujours plus grands et la complexité de ces projets augmente en conséquence.
Les équipes grandissent et le nombre de développeurs devient parfois si important qu’il est difficile d'intégrer de nouvelles recrues. 
De plus, certains développeurs peuvent quitter le développement d’un projet pendant sa réalisation. 
Il devient donc d’autant plus important de pouvoir engager de nouveaux salariés tout en les intégrants rapidement dans le projet.
Les nouveaux développeurs intégrant une équipe en cours de développement sont amenés à modifier des logiciels déjà existants.
Le problème principal est la complexité des logiciels qui augmente rapidement et le code qui est en constante évolution.

Mais alors se pose un premier problème de taille.
Comment faire en sorte de correctement intégrer ces nouveaux ingénieurs dans le projet existant ?
La solution la plus simple est de les confier à un développeur expert du projet pour leur transmettre les bases.
Mais si cela n’est pas possible ? 
Il faudrait ainsi répertorier les développeurs et les parties du code sur lesquelles ils ont travaillé.

C'est dans ce contexte qu'a été créé un outil d'analyse de la variabilité pour du code orienté objet en Java.
La variabilité est l’ensemble des mécanismes permettant de configurer du logiciel pour l’adapter à un contexte précis.
La variabilité peut être implémentée dans le code par différents mécanismes génériques, comme l’héritage, la surcharge 
et certains patrons de conception dans des systèmes orientés objets.
L'outil créé va permettre de visualiser les endroits (classes, méthodes, attributs...) où on peut trouver cette variabilité.
Cette analyse du code peut servir de base pour avoir une vision d'ensemble des parties complexes du code. 
L'étape suivante est de comprendre ces points de variation pour participer au développement du logiciel.
Si la documentation est absente et que le développeur ne sait pas à qui s'adresser pour comprendre, il peut rester bloqué de son côté.
Il serait donc intéressant de savoir qui est l'auteur de cette variabilité afin de lui poser directement des questions.
Nous ne traiterons que le développement de projet orienté objets ici pour rester dans le cadre de notre sujet.

## II. Observations/Question générale

1. Commencez par formuler une question sur quelque chose que vous observez ou constatez ou encore une idée émergente. 
    
2. Préciser pourquoi cette question est intéressante de votre point de vue.

Attention pour répondre à cette question, vous devrez être capable d'émettre des hypothèses vérifiables, de quantifier vos réponses, ...

     :bulb: Cette première étape nécessite beaucoup de réflexion pour se définir la bonne question afin de poser les bonnes bases pour la suite.

### Reformulation du sujet 

On peut reformuler le sujet en question de la façon suivante :
> Comment déterminer la paternité de la variabilité du code d’un projet orienté objet ?

### Intérêt

Selon le type d’application développée et le fonctionnement en entreprise, le développement peut être fait par une ou plusieurs personnes.  
Dans une entreprise, il peut y avoir des départs, des changements de postes, des nouveaux arrivants ce qui impliquent que les personnes qui s’occupent d’une application peuvent varier.  
C’est dans cette optique que l’analyse de la paternité est un outil qui va permettre d’améliorer la transmission de connaissances et la découverte du fonctionnement d’une application sur les parties complexes qui peuvent nécessiter une grande maitrise de ce qui a déjà été développé.  
En effet, savoir qui est ou sont les développeurs principaux d’une partie de l’application permet d’améliorer la montée en compétence de ce qui n’ont pas ces connaissances.  
La mise en place de cet outil serait donc une grande amélioration dans le monde du développement.

### Question générale
Notre question générale sera donc :
> Comment la variabilité est distribuée entre plusieurs auteurs dans du code orienté objet ?


La première étape est donc de bien déterminer quel type de variabilité nous allons considérer.
Grâce à un outil de Git, il est possible d’obtenir à un instant “t” du projet, tous les auteurs d’un fichier précis.

À partir des résultats obtenus, nous avons donc reformulé la question :
> Comment analyser ces résultats pour identifier les différents auteurs de la variabilité et ressortir des statistiques sur la paternité du projet à un instant “t” ?

Comme dit précédemment, l’identification de la paternité va permettre de faciliter la transmission de connaissance 
sur les points complexes du code entre les développeurs experts et les nouveaux arrivants.
La variabilité pouvant être décomposée sous forme de "patterns" (patron de conception), le nouveau développeur pourrait cibler sa recherche 
sur un pattern spécifique afin de trouver les auteurs auprès de qui poser des questions pour comprendre le fonctionnement du pattern à travers le code.

### Limites
Dans notre approche, on peut identifier les limites suivantes :
- L'analyse du code se fait à un instant t, on prend le dernier commit de la branche principale du repository. 
  Une extension intéressante serait d'analyser les modifications à travers les différentes versions du projet.
- Plus les projets sont vieux, plus la variabilité peut être diluée à travers les différents contributeurs. 
  Les résultats obtenus pourraient être différents selon l'ancienneté du projet.
- Plusieurs contributeurs peuvent être une seule et même personne sur un projet avec des adresses mail différentes. 
  L'identification de l'auteur peut être faussée.
- Les patterns de variabilité étudiés sont "VP" et "VARIANT". 
  VP correspond à un "VARIATION POINT", c'est-à-dire une classe qui peut être implémenter par d'autres et possède des "VARIANTS" qui l'implémente.
  Ces deux patterns permettent d'identifier facilement quelles sont leurs lignes de code associées (le fichier entier correspond au code du pattern).  

## III. Collecte d'information

Préciser vos zones de recherches en fonction de votre projet, les informations dont vous disposez, ... :

     :bulb: Cette étape est fortement liée à la suivante. Vous ne pouvez émettre d'hypothèses à vérifier que si vous avez les informations, inversement, vous cherchez à recueillir des informations en fonction de vos hypothèses. 

**1. Les articles ou documents utiles à votre projet📝**

Dans le cadre de notre recherche, nous avons basé notre travail sur les ressources suivantes :
1. [On the notion of variability in software product lines](https://doi.org/10.1109/WICSA.2001.948406)  
 Les auteurs abordent la notion de variabilité dans le développement de logiciels. Ils fournissent un cadre de terminologie et de concepts, identifient des modèles courants de variabilité et proposent une méthodologie pour gérer la variabilité dans les systèmes de logiciels industriels.


2. [Visualization of Object-Oriented Variability Implementations as Cities](https://hal.archives-ouvertes.fr/hal-03312487)  
   VariCity est un outil de visualisation qui représente un code orienté objet comme une ville pour aider les nouveaux venus à comprendre les parties les plus importantes du projet, en particulier en ce qui concerne la mise en œuvre de la variabilité. Il détecte la variabilité à travers les symétries du code et la représente à travers la taille des bâtiments, la couleur et des formes de bâtiments spécifiques. Les rues de la ville sont agencées en fonction des relations entre les classes. 


3. [On the usefulness of ownership metrics in open-source software projects](https://www.sciencedirect.com/science/article/abs/pii/S0950584915000294)  
  Le papier examine la relation entre les métriques de propriété de code et la qualité du logiciel dans les projets de logiciels libres. Les résultats montrent une relation entre les deux, mais d'autres métriques ont un plus grand impact sur la qualité du logiciel


**2. Les jeux de données 💾**

Nous procéderons à l'analyse de projets GitHub. 
La liste contenant la totalité des projets utilisés se trouve dans un excel disponible [ici.](chapters/2023/Paternité de la variabilité sur de l'orienté objet/assets/data/GitHub_projects_list.xlsx)
Cette liste contient 

Les projets GitHub que nous allons analyser comportent les critères suivants :
- langage de développement : JAVA (orienté objet),
- nombre de contributeurs : entre 10 et 40,
- taille de la base de code : inférieure à 500 KB.

Ces critères sont définis pour limiter le choix des projets à analyser. 
Pour faire ressortir la notion de paternité, il faut avoir plus d'un seul contributeur. 
Cependant, avec un nombre trop conséquent de contributeur, l'analyse risque d'être trop morcelée donc une limite expérimentale est fixée à 40 contributeurs.

La taille de la base de code choisie est directement liée au temps d'analyse du projet pour calculer la variabilité de celui-ci. 
Un trop gros projet mettrait beaucoup de temps à être analysé donc nous ciblons des projets de taille moyenne ou petite.

Un projet peut donc être défini par son nombre de lignes de code, son nombre de contributeurs et sa quantité de variabilité.

**3. Les outils🔨🪓**

- [git blame](https://git-scm.com/docs/git-blame) : Outil d'identification des derniers auteurs qui ont modifié les lignes de code d'un fichier. 
- [Symfinder](https://deathstar3.github.io/symfinder-demo/) : Outil d'analyse de la variabilité d’un projet orienté objet à un instant donné.
- [Docker/Docker-Compose](https://docs.docker.com/get-started/overview/) : Outil de lancement de l'analyse de la variabilité
- [Scripts Python](chapters/2023/Paternité de la variabilité sur de l'orienté objet/assets/code)

Pour plus de détails sur comment utiliser ces outils, voir partie [VI. Outils](#vi-outils).

## IV. Hypothèses et Expériences

1. Il s'agit ici d'**énoncer sous forme d'hypothèses** ce que vous allez chercher à démontrer. Vous devez définir vos hypothèses de façon à pouvoir les _mesurer/vérifier facilement._ Bien sûr, votre hypothèse devrait être construite de manière à _vous aider à répondre à votre question initiale_. Explicitez ces différents points.
2. Vous **explicitez les expérimentations que vous allez mener** pour vérifier si vos hypothèses sont vraies ou fausses. Il y a forcément des choix, des limites, explicitez-les.

     :bulb: Structurez cette partie à votre convenance : Hypothèse 1 => Expériences, Hypothèse 2 => Expériences ou l'ensemble des hypothèses et les expériences....

Pour notre analyse, on va seulement considérer les contributeurs qui ont participé à l'écriture de code contenant de la variabilité.

### 1. Répartition de la variabilité selon le nombre de contributeurs

#### Hypothèse

*La variabilité est distribuée dans le projet selon le nombre de contributeurs. Plus le nombre de contributeurs est élevé, plus la variabilité est répartie entre eux.*

#### Sous-question

> Comment évolue la répartition de la variabilité avec le nombre de contributeurs qui augmentent ?

#### Métriques
- Pourcentage de variabilité moyenne par contributeur (%/personne)
- Nombre moyen de contributeurs ayant participé à l'écriture de la variabilité (N personnes)

#### Experience
##### Source

Les projets que nous avons choisis pour cette expérience sont les suivants :

|                      Projet                       | Lien                                                                 | Nombre de contributeurs | Nombre de développeurs |
|:-------------------------------------------------:|:---------------------------------------------------------------------|:-----------------------:|:----------------------:|
|             JakeWharton/DiskLruCache              | https://github.com/JakeWharton/DiskLruCache                          |           10            |           ?            |
|                JakeWharton/RxRelay                | https://github.com/JakeWharton/RxRelay                               |           15            |           ?            |
|               Flipboard/bottomsheet               | https://github.com/Flipboard/bottomsheet                             |           20            |           ?            |
|      VerbalExpressions/JavaVerbalExpressions      | https://github.com/VerbalExpressions/JavaVerbalExpressions           |           25            |           ?            |
| EnterpriseQualityCoding/FizzBuzzEnterpriseEdition | https://github.com/EnterpriseQualityCoding/FizzBuzzEnterpriseEdition |           31            |           ?            |
|                EngineHub/WorldEdit                | https://github.com/EngineHub/WorldEdit                               |           96            |           52           |

#### Description

Le but de cette expérience est de déterminer si le nombre de contributeurs influe sur la répartition de la variabilité.
Pour cela, on a sélectionné des projets contenant de la variabilité avec un nombre de contributeurs croissant (de 10 à 52).
À partir de l'analyse de la variabilité de chaque projet, un filtre est appliqué pour isolé tous les "VARIANTS" 
et itérer sur chacun afin d'appliquer ``git blame`` sur le fichier contenant le "VARIANT" 
et identifier les différents auteurs ainsi que leur pourcentage de participation à l'écriture de ce fichier.

#### Démarche
Une fois que la variabilité est analysée (grâce à Symfinder), la démarche à suivre est composée des étapes suivantes : 
1. Execution du script python ``paternity_variability_detail.py`` avec comme arguments  

### 2. Paternité commune sur la variabilité de type "VP" (Variant Point) et "VARIANT"

**Hypothèse**

*Un contributeur qui modifie un VP va aussi modifier ses VARIANTS.*

**Sous-question**

> Y a-t-il une relation de paternité entre la variabilité présente dans un VP et son implémentation dans ses VARIANTS ?

**Métriques**
- Nombre (et liste) de contributeurs ayant participé à l'écriture d'un VP
- Nombre (et liste) de contributeurs ayant participé à l'écriture des VARIANTS associés au VP
- Pourcentage de corrélation entre les deux
- Le nombre moyen de contributeurs supplémentaires sur les VARIANTS (pas dans les VPs)

**Experience**  
Les projets que nous avons choisis pour cette expérience sont les suivants : 

|                      Projet                       | Lien                                                                 | Nombre de contributeurs | Nombre de développeurs |
|:-------------------------------------------------:|:---------------------------------------------------------------------|:-----------------------:|:----------------------:|
| EnterpriseQualityCoding/FizzBuzzEnterpriseEdition | https://github.com/EnterpriseQualityCoding/FizzBuzzEnterpriseEdition |           31            |           ?            |
|                EngineHub/WorldEdit                | https://github.com/EngineHub/WorldEdit                               |           96            |           52           |

## V. Résultat d'analyse et Conclusion

### Présentation des résultats

#### Experience 1

#### Experience 2

### Analyse et interprétation des résultats en fonction des hypothèses

### Limites rencontrées

### Recul et pertinence des remarques

### Conclusion


     :bulb:  Vos résultats et donc votre analyse sont nécessairement limités. Préciser bien ces limites : par exemple, jeux de données insuffisants, analyse réduite à quelques critères, dépendance aux projets analysés, ...

## VI. Outils

Précisez votre utilisation des outils ou les développements (e.g. scripts) réalisés pour atteindre vos objectifs. Ce chapitre doit viser à (1) pouvoir reproduire vos expérimentations, (2) partager/expliquer à d'autres l'usage des outils.

### Scripts Python
**scraper.py**  
Ce script permet d’analyser la variabilité des projets java. Il utilise l'API Github pour obtenir une liste de dépôts, puis pour chaque dépôt, il utilise l’outil  "SymFinder"  pour effectuer une analyse de la variabilité et enregistre le résultat sous forme de fichier JSON. Ensuite, le script "paternity_variability.py" est exécuté pour trouver la paternité de la variabilité. 

**paternity_variability.py**  
Ce script calcule la paternité de la variabilité dans un projet Git donné. Il utilise la sortie de "SymFinder" (stockée dans un fichier JSON appelé db.json) pour trouver les classes de variabilité dans le projet. Pour chaque classe de variabilité trouvée, il utilise la commande Git "blame" pour trouver les auteurs des lignes de code modifiées pour cette classe et calcule la fraction de lignes modifiées par chaque auteur. Les résultats sont ensuite stockés dans un fichier de sortie au format JSON qui va être consommer par le script ‘’ Visualization’’.

**visualization.py**  
Ce script définit une classe PlotPie qui permet de tracer des graphiques en secteurs (pie charts) à partir de données générer précédemment. Le script prend en entrée le chemin vers le fichier JSON, lit les données à partir du fichier, les trie et les utilise pour tracer un graphique en secteurs pour chaque type de variabilité. Les graphiques sont enregistrés dans un sous-dossier "Visualization" avec le même nom du projet. 

![Figure 1: Workflow](assets/images/workflow.svg)

![Figure 1: Logo UCA](assets/images/logo_uca.png)


## VI. Références


 [1] J. van Gurp, J. Bosch, and M. Svahnberg. 2001. On the notion of variability in software product lines. In Proceedings Working IEEE/IFIP Conference on Software Architecture, IEEE Comput. Soc, Amsterdam, Netherlands, 45–54. DOI:https://doi.org/10.1109/WICSA.2001.948406
 
 [2] Johann Mortara, Philippe Collet, Anne-Marie Dery-Pinna. Visualization of Object-Oriented Variability Implementations as Cities. 9th IEEE Working Conference on Software Visualization (VISSOFT 2021), Sep 2021, Luxembourg https://hal.archives-ouvertes.fr/hal-03312487
 
[3] Foucault, M., Teyton, C., Lo, D., Blanc, X., & Falleri, J. R. (2015). On the usefulness of ownership metrics in open-source software projects. Information and Software Technology, 64, 102-112. https://www.sciencedirect.com/science/article/abs/pii/S0950584915



