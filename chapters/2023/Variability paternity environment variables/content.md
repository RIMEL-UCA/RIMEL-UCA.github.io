---
layout: default
title: Variability paternity environment variables
date: 2022-11
---

**_février 2023_**

## Auteurs

Nous sommes quatre étudiants ingénieurs en dernière année à Polytech Nice Sophia, spécialisés en Architecture Logiciel :

- Guillaume Piccina ([Github](https://github.com/guillaume-piccina))
- William D'Andrea ([Github](https://github.com/william-dandrea))
- Nicolas Fernandez ([Github](https://github.com/Nicolas-Fern))
- Yann Brault ([Github](https://github.com/Yann-Brault))


## I. Contexte de la recherche / projet

Notre contexte de recherche porte sur la variabilité et la paternité dans le code, dans ce sujet on se pose la question suivante :

> Peut-on déterminer la paternité de l’implémentation (et l’évolution) d’une fonctionnalité variable à partir du code en appliquant les méthodes de détermination de paternité aux endroits où la variabilité est implémentée ?
> Et en particulier, peut-on se concentrer sur la chaîne complète (depuis des outils de construction qui exploitent les variables d’environnement jusqu’au code) ?

Ce sujet est intéréssant à aborder puisque dans le métier de développeur nous sommes tout le temps amené à travailler sur des projets avec une base de code écrite par d'autres développeurs. Dans ce contexte il est important de pouvoir comprendre le code et également la manière dont il a été implémenté. C'est ainsi que le concept de paternité rentre en jeu : il peut être intéressant d'avoir des outils capables de nous donner rapidement quels sont les personnes qui ont travaillés sur certaines parties du code et donc qui sont les plus aptes à nous renseigner sur leur implémentation. Concernant la variabilité celle-ci peut être définie comme l’ensemble des mécanismes permettant de configurer un logiciel pour l’adapter à un contexte précis. La variabilité joue donc un rôle important dans le code d'un logicel, quel que soit le projet on retrouve de la variabilité et notamment sous forme de variable d'environnement (ce sur quoi nous nous sommes concentrés). Notre sujet relit ainsi ces deux concepts de variabilité et de paternité dans le but de répondre à la problématique ci-dessus.

## II. Observations / Question générale
  
Dans un premier temps nous avons établi une première question générale à savoir :

     Peut-on identifier, à gros grain, la paternité des variables d'environnement dans un code ?
     
L'objectif de cette question est de trouver le premier créateur d’une variable d’environnement, et ensuite suivre qui l’a édité au fur et à mesure du temps (jusqu’à sa potentielle suppression)

Pour répondre à cette question et réaliser l'objectif nous avons établi deux autres sous-questions pour mieux définir ce que l'on cherche. La première étant :

     Qui a créé pour la première fois une variable d’environnement ?
     
On fait l'hypothèse que lorsqu'un utilisateur ajoute une variable d’environnement au projet, il l’ajoute en premier lieu dans un fichier de configuration comme par exemple (le fichier .properties pour les projets Gradle). Nous avons réduit le scope de la question au projet Maven et Gradle.

La deuxième sous-question est :

     Qui a utilisé cette variable d’environnement dans le code ?
     
Pour cette question on fait l'hypothèse que ...

## III. Collecte d'informations

Préciser vos zones de recherches en fonction de votre projet, les informations dont vous disposez, ... :

1. les articles ou documents utiles à votre projet
2. les outils
3. les jeux de données/codes que vous allez utiliser, pourquoi ceux-ci, ...

   :bulb: Cette étape est fortement liée à la suivante. Vous ne pouvez émettre d'hypothèses à vérifier que si vous avez les informations, inversement, vous cherchez à recueillir des informations en fonction de vos hypothèses.
   
Pour ce sujet nous nous sommes appuyés

## IV. Hypothèses et expériences

1. Il s'agit ici d'**énoncer sous forme d'hypothèses** ce que vous allez chercher à démontrer. Vous devez définir vos hypothèses de façon à pouvoir les _mesurer/vérifier facilement._ Bien sûr, votre hypothèse devrait être construite de manière à _vous aider à répondre à votre question initiale_. Explicitez ces différents points.
2. Vous **explicitez les expérimentations que vous allez mener** pour vérifier si vos hypothèses sont vraies ou fausses. Il y a forcément des choix, des limites, explicitez-les.

   :bulb: Structurez cette partie à votre convenance : Hypothèse 1 => Expériences, Hypothèse 2 => Expériences ou l'ensemble des hypothèses et les expériences....

## V. Analyse des résultats et conclusion

1. Présentation des résultats
2. Interprétation/Analyse des résultats en fonction de vos hypothèses
3. Construction d’une conclusion

   :bulb: Vos résultats et donc votre analyse sont nécessairement limités. Préciser bien ces limites : par exemple, jeux de données insuffisants, analyse réduite à quelques critères, dépendance aux projets analysés, ...

## VI. Outils

Précisez votre utilisation des outils ou les développements \(e.g. scripts\) réalisés pour atteindre vos objectifs. Ce chapitre doit viser à \(1\) pouvoir reproduire vos expérimentations, \(2\) partager/expliquer à d'autres l'usage des outils.

![Figure 1: Logo UCA, exemple, vous pouvez l'enlever](assets/images/logo_uca.png)

## VI. Références

[Debret 2020] Debret, J. (2020) La démarche scientifique : tout ce que vous devez savoir ! Available at: https://www.scribbr.fr/article-scientifique/demarche-scientifique/ (Accessed: 18 November 2022).
