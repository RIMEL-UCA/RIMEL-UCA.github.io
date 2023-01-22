---
layout: default
title: Analyse des dépendances entre les étapes de build
date: 2022-11
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

## Authors

We are four students in last year of M2 Nice-Sophia specialized in Software Architecture :

- KHERROUBI Abdelkader ([@abdelkader1996](https://github.com/abdelkader1996)),
- SI DEHBI Ahmed El Hanafi ([@AhmedElHanafi](https://github.com/AhmedElHanafi)),
- HERAUD Antoine  ([@herauda](https://github.com/herauda)),
- NAJI Abdellah ([@abdellah07](https://github.com/abdellah07)). 

## I. Contexte de recherche /Projet

Dans cette étude, on examinent le DockerFile pour analyser les dépendances entre les différentes étapes de construction. Le DockerFile est un fichier qui contient des instructions qui sont utilisées par Docker pour créer et exécuter une image docker. Les images docker sont des ensembles d'instruction utilisés pour construire et déployer des applications logicielles sur le cloud.
Lors de la construction d'une image Docker, il est important de comprendre les dépendances entre les étapes de construction, 
la définition des dépendances est la partie la plus importante du fichier Dockerfile lors de la construction d'une image Docker. La gestion des dépendances améliore les performances et la stabilité de la construction du projet. 

## II. Questions General /Observations

1. Commencez par formuler une question sur quelque chose que vous observez ou constatez ou encore une idée émergente.

Peut-on analyser l'existence et la validité des dépendances contenues dans les étapes de construction d'image Docker a l'interieure d'un workflow de Github action ? 

2. Préciser pourquoi cette question est intéressante de votre point de vue.

Attention pour répondre à cette question, vous devrez être capable d'émettre des hypothèses vérifiables, de quantifier vos réponses, ...

     :bulb: Cette première étape nécessite beaucoup de réflexion pour se définir la bonne question afin de poser les bonnes bases pour la suite.

- Comment identifier les différentes dépendances dans un Dockerfile?
- Comment valider la stabilité d’un build Docker à l’aide de l’analyse de ses dépendances contenues dans les steps du workflow?
- Comment relever différentes mauvaises pratiques empêchant une meilleure analyse? 

Limites : Quelles sont les mauvaise pratiques pour l'ecriture d'un DockerFile ? 

## III. Collecte d'informations 

Notre projet se concentre sur la détection de  l'existence des fichiers important pour l'étape de build de l'image Docker, ainsi que l'application des bonnes pratiques de l'écriture du Dockerfile.

Les projets utilisés comme références sont les projets récupérés à partir du Docker Hub sous condition d'avoir un fichier de GitHub workflow. 

1. les articles ou documents utiles à votre projet

2. Outils  : 
- Notre outil Python: <a href="https://github.com/AhmedElHanafi/Dockerfile-Analyser">Dockerfile-Analyser</a>
   Un outil qu'on a développé dans l'équipe dans le but d'analyser les dépendances utilisées dans le Dockerfile, de faire une recherche dans le dossier du projet pour déterminer si les dépendances sont existantes ou manquantes. 

   Par la suite, on a une levée de drapeaux sur l'état de notre Dockerfile pour dire s'il est stable ou pas (prendre en compte les limitations mentionner dans cet article).

- <a href="https://github.com/">Github</a> 
- <a href="https://hub.docker.com/">Docker Hub</a>

3. Code utilisé :
   | Nom du Repository                                            |   Taille   | Popularité |  Régularité  | Contributeurs |
   |:-------------------------------------------------------------|:----------:|:----------:|:------------:|:-------------:|
   | [Docker Postgris](https://github.com/kartoza/docker-postgis) |  1027 ko   | 560 stars  | 316 commits  |       38      |
   | [Moby Project](https://github.com/moby/moby)                 | 77 496 ko  | 65k stars  | 45446 commits|     2195      |
   | [Camuflon - Api](https://github.com/camuflon/camuflon-api)   |   432 ko   |   0 stars  | 34 commits   |        1      |
   | [Desafio Backend](https://github.com/uandisson/desafio_backend)| 6 774 ko   | 1 stars | 46 commits   |        1      |
   | [OpenZipkin](https://github.com/openzipkin/zipkin-go)        |   572 ko   | 565 stars  | 287 commits  |      19       |

   :bulb: Cette étape est fortement liée à la suivante. Vous ne pouvez émettre d'hypothèses à vérifier que si vous avez les informations, inversement, vous cherchez à recueillir des informations en fonction de vos hypothèses.

## IV. Hypothèse et expériences

1. Il s'agit ici d'**énoncer sous forme d'hypothèses** ce que vous allez chercher à démontrer. Vous devez définir vos hypothèses de façon à pouvoir les _mesurer/vérifier facilement._ Bien sûr, votre hypothèse devrait être construite de manière à _vous aider à répondre à votre question initiale_. Explicitez ces différents points.
2. Vous **explicitez les expérimentations que vous allez mener** pour vérifier si vos hypothèses sont vraies ou fausses. Il y a forcément des choix, des limites, explicitez-les.

   :bulb: Structurez cette partie à votre convenance : Hypothèse 1 => Expériences, Hypothèse 2 => Expériences ou l'ensemble des hypothèses et les expériences....

## V. Analyse des résultats et conclusion

1. Présentation des résultats
2. Interprétation/Analyse des résultats en fonction de vos hypothèses
3. Construction d’une conclusion

   :bulb: Vos résultats et donc votre analyse sont nécessairement limités. Préciser bien ces limites : par exemple, jeux de données insuffisants, analyse réduite à quelques critères, dépendance aux projets analysés, ...

## VI. Outils \(facultatif\)

Précisez votre utilisation des outils ou les développements \(e.g. scripts\) réalisés pour atteindre vos objectifs. Ce chapitre doit viser à \(1\) pouvoir reproduire vos expérimentations, \(2\) partager/expliquer à d'autres l'usage des outils.

![Figure 1: Logo UCA, exemple, vous pouvez l'enlever](images/logo_uca.png){:height="25px"}

## VI. Références

[Debret 2020] Debret, J. (2020) La démarche scientifique : tout ce que vous devez savoir ! Available at: https://www.scribbr.fr/article-scientifique/demarche-scientifique/ (Accessed: 18 November 2022).
