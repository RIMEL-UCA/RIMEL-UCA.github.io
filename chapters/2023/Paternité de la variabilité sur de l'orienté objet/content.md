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
Cet outil va permettre de visualiser les endroits (classes, méthodes, attributs...) où il y a de la variabilité.
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

Comment la variabilité est distribuée entre plusieurs auteurs dans du code orienté objet ?

Selon le type d’application développée et le fonctionnement en entreprise, le développement peut être fait par une ou plusieurs personnes.  
Dans une entreprise, il peut y avoir des départs, des changements de postes, des nouveaux arrivants ce qui impliquent que les personnes qui s’occupent d’une application peuvent varier.  
C’est dans cette optique que l’analyse de la paternité est un outil qui va permettre d’améliorer la transmission de connaissances et la découverte du fonctionnement d’une application sur les parties complexes qui peuvent nécessiter une grande maitrise de ce qui a déjà été développé.  
En effet, savoir qui est ou sont les développeurs principaux d’une partie de l’application permet d’améliorer la montée en compétence de ce qui n’ont pas ces connaissances.  
La mise en place de cet outil serait donc une grande amélioration dans le monde du développement.

Notre question générale sera donc :
Comment déterminer la paternité de la variabilité du code d’un projet orienté objet ?

La première étape est donc de bien déterminer quel type de variabilité nous allons considérer.
Grâce à un outil de Git, il est possible d’obtenir à un instant “t” du projet, tous les auteurs d’un fichier précis.

À partir des résultats obtenus, nous avons donc reformuler la question :
Comment analyser ces résultats pour identifier les différents auteurs de la variabilité et ressortir des statistiques sur la paternité du projet à un instant “t” ?

Comme dit précédemment, l’identification de la paternité va permettre de faciliter la transmission de connaissance 
sur les points complexes du code entre les développeurs experts et les nouveaux arrivants.
La variabilité pouvant être décomposée sous forme de patterns, le nouveau développeur pourrait cibler sa recherche 
sur un pattern spécifique afin de trouver les auteurs auprès de qui poser des questions pour comprendre le fonctionnement du pattern à travers le code.


## III. Collecte d'information

Préciser vos zones de recherches en fonction de votre projet, les informations dont vous disposez, ... :

     :bulb: Cette étape est fortement liée à la suivante. Vous ne pouvez émettre d'hypothèses à vérifier que si vous avez les informations, inversement, vous cherchez à recueillir des informations en fonction de vos hypothèses. 

1. Les articles ou documents utiles à votre projet📝

Dans le cadre de notre recherche, nous prévoyons de nous baser sur les ressources suivantes :
- [On the notion of variability in software product lines](https://doi.org/10.1109/WICSA.2001.948406)
- [Visualization of Object-Oriented Variability Implementations as Cities](https://hal.archives-ouvertes.fr/hal-03312487)
- [An analysis of the variability in forty preprocessor-based software product lines](https://doi.org/10.1145/1806799.1806819)
- [On the usefulness of ownership metrics in open-source software projects](https://www.sciencedirect.com/science/article/abs/pii/S0950584915000294)



2. Les jeux de données 💾

Nous procéderons à l'analyse des projets GitHub présents dans le répertoire Assets/data,Nous allons examiner ces projets en raison de leur nombre conséquent de contributeurs (entre 10 et 40) ainsi que de leur taille inférieure à 500KB, ce qui nous permettra de faciliter notre procédure d'analyse à l'aide de notre outil "ScraperPV"."

3. Les outils🔨🪓

- git blame
- Symfinder (Pour connaître la variabilité d’un projet orienté objet à un instant donné, on a utilisé Symfinder. Cela va servir de point de départ pour remonter le nom de la ou les personnes responsables de cette variation. )
- Docker/Docker-Compose
- Scripts Python

## IV. Hypothèses et Experience

1. Il s'agit ici d'**énoncer sous forme d'hypothèses** ce que vous allez chercher à démontrer. Vous devez définir vos hypothèses de façon à pouvoir les _mesurer/vérifier facilement._ Bien sûr, votre hypothèse devrait être construite de manière à _vous aider à répondre à votre question initiale_. Explicitez ces différents points.
2. Vous **explicitez les expérimentations que vous allez mener** pour vérifier si vos hypothèses sont vraies ou fausses. Il y a forcément des choix, des limites, explicitez-les.

     :bulb: Structurez cette partie à votre convenance : Hypothèse 1 => Expériences, Hypothèse 2 => Expériences ou l'ensemble des hypothèses et les expériences....

### Hypothèses

#### Hypothèse 1

*La variabilité est bien distribuée à travers tous les contributeurs quand le projet est de grande taille (beaucoup de line de code, beaucoup de développeurs).*

Sous-question :

*Comment la paternité de la variabilité est répartie entre les contributeurs dans un gros projet ?*

Résultats attendus :

Une paternité très fragmentée avec un faible pourcentage pour chaque auteur GitHub

#### Hypothèse 2

*Pour de petit projet, il y a peu de développeur, voire potentiellement un seul pour qui la paternité de la variabilité est la plus importante, ou autrement dit, un développeur apparais comme principale dans le projet.*

Sous-question :

La paternité de la variabilité est-elle répartie de la même façon dans de petit projet, ou dans des projets avec peu de développeur ?

Résultats attendus :

En l'occurrence ici on s'attend à confirmer l'hypothèse notamment due au fait que dans des petits projets, il y a souvent un ou deux développeurs experts, pères de la majeure partie de la variabilité.

#### Hypothèse 3

*La paternité de variabilité est le même pour tous les types de variabilité (tjs les mêmes contributeurs pour les différentes variabilités).*

Sous-question :

Est-ce que la répartition de la paternité suit un schéma par rapport aux patterns de variabilité ?

Résultats attendus :

On s'attend à ce qu'un développeur qui produit de la variance d'un type produise de la variance sur les autres types du même niveau.

### Experiences

#### Hypothèse 1

Projets ciblés :
- amitshekhariitbhu/from-java-to-kotlin (https://github.com/amitshekhariitbhu/from-java-to-kotlin)
- EnterpriseQualityCoding/FizzBuzzEnterpriseEdition
- frohoff/ysoserial
- gcacace/android-signaturepad
- spotify/dockerfile-maven

#### Hypothèse 2

Petits projets ou projets avec peu de développeur.

#### Hypothèse 3

Projets avec peu de contributeurs, mais beaucoup de code.

## V. Résultat d'analyse et Conclusion

1. Présentation des résultats
2. Interprétation/Analyse des résultats en fonction de vos hypothèses
3. Construction d’une conclusion 

     :bulb:  Vos résultats et donc votre analyse sont nécessairement limités. Préciser bien ces limites : par exemple, jeux de données insuffisants, analyse réduite à quelques critères, dépendance aux projets analysés, ...

## VI. Outils \(facultatif\)

Précisez votre utilisation des outils ou les développements \(e.g. scripts\) réalisés pour atteindre vos objectifs. Ce chapitre doit viser à \(1\) pouvoir reproduire vos expérimentations, \(2\) partager/expliquer à d'autres l'usage des outils.

**Scraper.py**
Ce script permet d’analyser la variabilité des projets java. Il utilise l'API Github pour obtenir une liste de dépôts, puis pour chaque dépôt, il utilise l’outil  "SymFinder"  pour effectuer une analyse de la variabilité et enregistre le résultat sous forme de fichier JSON. Ensuite, le script "paternity_variability.py" est exécuté pour trouver la paternité de la variabilité. 

**paternity_variability.py**
Ce script calcule la paternité de la variabilité dans un projet Git donné. Il utilise la sortie de "SymFinder" (stockée dans un fichier JSON appelé db.json) pour trouver les classes de variabilité dans le projet. Pour chaque classe de variabilité trouvée, il utilise la commande Git "blame" pour trouver les auteurs des lignes de code modifiées pour cette classe et calcule la fraction de lignes modifiées par chaque auteur. Les résultats sont ensuite stockés dans un fichier de sortie au format JSON qui va être consommer par le script ‘’ Visualization’’.

**Visualization.py**
Ce script définit une classe PlotPie qui permet de tracer des graphiques en secteurs (pie charts) à partir de données générer précédemment. Le script prend en entrée le chemin vers le fichier JSON, lit les données à partir du fichier, les trie et les utilise pour tracer un graphique en secteurs pour chaque type de variabilité. Les graphiques sont enregistrés dans un sous-dossier "Visualization" avec le même nom du projet. 


![Figure 1: Logo UCA, exemple, vous pouvez l'enlever](assets/images/logo_uca.png){:height="25px"}


## VI. Références

[Debret 2020] Debret, J. (2020) La démarche scientifique : tout ce que vous devez savoir ! Available at: https://www.scribbr.fr/article-scientifique/demarche-scientifique/ (Accessed: 18 November 2022).


