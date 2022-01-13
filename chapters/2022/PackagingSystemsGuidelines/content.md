---
layout: default
title : Quelles sont les bonnes pratiques de mise en place d’un système de packaging ?
date:   2022-01-08 20:00:00 +0100
---

**_janvier 2022_**


## Auteurs

Nous sommes cinq étudiants en dernière année à Polytech Nice Sophia Antipolis, dans la spécialité Architecture Logicielle :

* Bruel Martin &lt;martin.bruel@etu.univ-cotedazur.fr&gt;
* Esteve Thibaut &lt;thibaut.esteve@etu.univ-cotedazur.fr&gt;
* Lebrisse David &lt;david.lebrisse@etu.univ-cotedazur.fr&gt;
* Meulle Nathan &lt;nathan.meulle@etu.univ-cotedazur.fr&gt;
* Ushaka Kevin &lt;kevin.ushaka-kubwawe@etu.univ-cotedazur.fr&gt;


## I. Contexte de recherche / Projet

Le sujet de notre recherche porte sur les systèmes de packaging en Javascript. 

Un système de packaging permet de faciliter le développement et la gestion de sites/d'applications web. 
Il récupère les modules avec des dépendances et génère des ressources statiques représentant ces modules.

Les systèmes de packaging permettent également d'optimiser la taille de code en minifiant le code, réutilisant certains blocs.

Ce contexte de recherche est particulièrement intéressant pour nous puisque nous avons tous été amenés à travailler sur des projets incluant du Javascript. De plus, ce langage reste aujourd'hui l'un des plus utilisés. 

Il existe une multitude de librairies développées en Javascript : de nombreux projets utilisent des dépendances vers ces librairies Javascript. Il faut donc gérer et organiser ces librairies ce qui devient de plus en plus compliqué. Cela explique l'intérêt croissant des systèmes de packaging.


## II. Observations/Question générale

Nous avons remarqué, après une rapide recherche sur Github, que la plupart des grands projets Javascript utilisent des systèmes de packaging (afin de gérer les dépendances, répondre à des besoins d'optimisations, etc.). Ces gestionnaires de packages permettent, une fois bien configurés, de réduire les temps de développement et de faciliter le passage en production de l'application réduisant ainsi les coûts pour les entreprises.

Nous nous sommes ainsi demandés **Quelles sont les bonnes pratiques de mise en place d’un système de packaging**.


Nous avons ensuite affiné cette question générale en sous-questions :

- Quelles sont les étapes récurrentes dans la configuration d’un système ?
(metrics : identification manuelle puis taux d'occurrence parmi différents projets. Example d'étapes type : linter, minimiseur, offuscation...)

- Qu’est-ce qui fait une bonne pratique ? Est-ce qu’il y a des contraintes à utiliser les pratiques de webpack ou celles présentées par les developpeurs (dans les forums, blogs...) ? 

- Quels sont les paramètres d'un projets qui influent sur ces différentes étapes ?
On se penchera particulièrement sur la taille d'un projet à l'égard de l'utilisation d'un minimizer.
(métrics : lignes de code, présence d'un minimizer)
On étudiera également l'ancienneté d'un projet vis à vis de l'utilisation d'un minimizer.
(métrics : date de création, présence d'un minimizer)


Pour ce faire, nous avons prévu d'adopter la démarche suivante : 
- Monter en compétence sur les différents systèmes de packaging (comment ils sont utilisés, pourquoi c’est utilisé)
- Trouver les bonnes pratiques (documentation, littérature, forum,...)
- Chercher des projets de différentes tailles utilisant Gulp/Webpack
- Trouver les pratiques mises en place dans ces projets
- Evaluer les pratiques dans les projets


## III. information gathering

A venir
 
## IV. Hypothesis & Experiences

A venir

## V. Result Analysis and Conclusion

A venir

## VI. Tools \(facultatif\)

A venir

## VI. References

1. Webpack Documentation - https://webpack.js.org/
2. The JavaScript Packaging Problem - http://jamie-wong.com/2014/11/29/the-js-packaging-problem/
