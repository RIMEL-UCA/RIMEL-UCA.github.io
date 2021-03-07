---
layout: default
title : Etude de la relation entre tests fonctionnels et tests unitaires dans un projet BDD
date:   2021-01-14 21:08:00 +0100
---

## Auteurs

Nous sommes 4 étudiants en dernière année à Polytech' Nice-Sophia spécialisés en Architecture Logicielle :

* Alexandre Longordo &lt;alexandre.longordo@etu.univ-cotedazur.fr&gt;
* Alexis Lefebvre &lt;alexis.lefebvre@etu.univ-cotedazur.fr&gt;
* Lydia Baraukova &lt;lydia.baraukova@etu.univ-cotedazur.fr&gt;
* Younes Abdennadher &lt;younes.abdennadher@etu.univ-cotedazur.fr&gt;

## I. Contexte de recherche

Le BDD (Behaviour-Driven Development) est une technique de développement encourageant les équipes de développement à collaborer avec les équipes fonctionnelles en 
poussant à utiliser un langage naturel structuré pour formaliser la manière dont l’application doit se comporter d’un point de vue fonctionnel. 

Ce type de développement permet de formuler clairement et de manière formelle les fonctionnalités (features) au travers de scénarios exécutables et proches des critères 
d’acceptation des User Stories. Les tests d’acceptation ainsi créés permettent donc d’exécuter automatiquement des vérifications liant au plus près le besoin métier et le code testé. 

L'utilisation du BDD dans les projets logiciels est en forte augmentation depuis quelques années, notamment grâce à des frameworks comme Cucumber. 
Cette forte utilisation s'accompagne cependant d'un manque de recul sur certaines pratiques conjointes, comme les tests unitaires. De plus, des règles de bonne utilisation 
sont encore en cours de définition par les praticiens. 

Le but général de notre recherche est donc d’explorer les comportements conjoints du BDD et des tests unitaires sur les projets qui respectent ces techniques de développement et de tests. 

En analysant les exécutions des Tests Fonctionnels (TF) BDD et Tests Unitaires (TU) sur le code de plusieurs projets Open-Source utilisant la technique de BDD, 
on pourrait étudier la relation qu’il y a entre les deux types de tests et ainsi avancer dans la compréhension du comportement d’un projet BDD. 

![Figure 1: Logo UCA, exemple, vous pouvez l'enlever](../assets/model/UCAlogoQlarge.png)

## II. Question générale

La question qui découle de ce que l’on vient d’introduire est donc "Quelles sont les relations entre Tests Fonctionnels BDD et Tests unitaires ?". 

Si cette question est intéressante, c’est parce qu’elle nous permet d’étudier la couverture de chaque type de tests (unitaires et fonctionnels) dans un contexte de développement 
piloté par les fonctionnalités métier. Cette étude va nous permettre de comprendre les relations existantes entre les deux types de tests. 

En allant plus loin dans l’étude (ce que nous ne ferons pas ici), celle-ci pourrait nous donner des indications sur l’intérêt d’allier tests fonctionnels et unitaires 
et sur la manière dont on peut les écrire pour optimiser la couverture des deux types de tests. Cela pourrait même nous permettre de dire si la programmation BDD permet réellement 
de mieux couvrir les fonctionnalités métier et permet de produire du code de meilleure qualité.

Pour préciser notre question générale, nous l’avons décomposée en trois sous-questions : 

* Question 1 : Les tests unitaires et fonctionnels testent-ils les mêmes endroits dans le code ou sont-ils complémentaires ? 

* Question 2 : Y a-t-il un lien entre le nombre de tests unitaires qui appellent une méthode et le nombre de tests fonctionnels appelant cette même méthode ? 

* Question 3 : Pour une fonctionnalité donnée y a-t-il une corrélation entre la complexité des scénarios fonctionnels et le nombre de tests unitaires liés à cette fonctionnalité ? 
(Nous définissons plus tard la complexité des scénarios)

## III. Collecte d'informations

### Choix des projets

Les projets que nous avons analysés se devaient d’être open source et accessibles. Nous avons donc utilisé le set de données suivant : 

* Les projets de Conception Logicielle 2019-2020 (21 projets) 

* Les projets de Conception Logicielle 2020-2021 (23 projets) 

Tous ces projets ont des caractéristiques communes : ils sont écrits en Java, ils ont à la fois des tests unitaires et des tests fonctionnels, et leurs tests sont faits avec Cucumber et Junit. 

Nous nous sommes limités à un seul langage et les mêmes outils de tests car nous ne voulons faire des statistiques que sur un seul environnement. 
En effet, les tests et la manière d’écrire les tests peuvent varier d’un langage à un autre ou d’un framework à un autre. 
Se cantonner à un seul et même environnement nous assure que l’on compare les mêmes types de tests et donc que les résultats que l’on obtient sont significatifs 
pour l’environnement choisi. Si les environnements analysés sont trop différents, les statistiques obtenues manqueront peut-être de critères de comparaison.  

Nous avons donc choisi Cucumber avec le langage Java car ce sont des technologies que nous connaissons bien et que nous pourrons donc analyser plus facilement 
(identifier plus facilement les fonctionnalités décrites par les tests fonctionnels par exemple).

### La récupération des données dans les projets

Pour étudier la couverture des projets par des tests nous avons choisi d’utiliser la technologie JaCoCo.
JaCoCo est un outil permettant d’étudier la couverture de tests des projets Java. Il produit des rapports XML et CSV et des visualisations HTML.
A partir de ces fichiers, il est possible d’extraire des informations concernant la couverture des lignes de code. 

## IV. Hypothèses et expériences

Afin de répondre aux questions énoncées, nous avons formulé trois hypothèses correspondant aux questions posées en partie II. 

* Hypothèse A : Les tests unitaires et fonctionnels testent en majorité les mêmes lignes de code. 

* Hypothèse B : Il n’y a aucune corrélation entre le nombre de tests unitaires appelant une méthode et le nombre de tests fonctionnels appelant cette même méthode. 

* Hypothèse C : Plus une fonctionnalité est complexe, plus il y a de tests unitaires liés à cette fonctionnalité. 

Explicitons les démarches que nous avons adopté pour tester la validité de ces hypothèses.

### A. Les tests unitaires et fonctionnels testent en majorité les mêmes lignes de code 

Pour chaque projet : 

##### Étape 1

On lance les tests 1 par 1 en produisant un rapport JaCoCo à chaque fois. Après la production d’un rapport JaCoCo, on extrait les données utiles avec notre script. 

JaCoCo permet d’extraire la couverture de lignes par des tests. C’est l’information qui nous intéresse. Et en lançant les tests un par un, on est sûrs que les lignes indiquées par JaCoCo sont couvertes par le test en cours. 

#### B.

#### C.

## V. Analyse des résultats et conclusion



## VI. Outils

Précisez votre utilisation des outils ou les développements \(e.g. scripts\) réalisés pour atteindre vos objectifs. Ce chapitre doit viser à \(1\) pouvoir reproduire vos expérimentations, \(2\) partager/expliquer à d'autres l'usage des outils.

## VI. Réferences

1. [Article modèle pour la démarche expérimentale](https://hal.inria.fr/hal-01344842/file/Test_Case_Selection_in_Industry-An_Analysis_of_Issues_related_to_Static_Approaches.pdf)
2. [Documentation JaCoCo](https://www.jacoco.org/jacoco/trunk/doc/)
3. [Notre repository GitHub pour ce projet](https://github.com/AlexandreLon/Rimel-TeamF)
