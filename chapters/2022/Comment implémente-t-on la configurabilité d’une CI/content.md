---
layout: default
title : Modèle de chapitre pour 2021
date:   2021-01-03 22:00:00 +0100
---

---

> **Date de rendu finale : Mars 2021 au plus tard**
> - Respecter la structure pour que les chapitres soient bien indépendants
> - Remarques :
>>    - Les titres peuvent changer pour être en adéquation avec votre étude.
>>    - De même il est possible de modifier la structure, celle qui est proposée ici est là pour vous aider.
>>    - Utiliser des références pour justifier votre argumentaire, vos choix etc.

---

**_janvier 2021_**

## Authors

We are five students in last year of Polytech' Nice-Sophia specialized in Software Architecture :

* Valentin Roccelli <valentin.roccelli@etu.unice.fr>
* Rachid El Adlani <rachid.el-adlani@etu.unice.fr>
* Abdelouhab Belkhiri <abdelouhab.belkhiri@etu.unice.fr>
* Armand Fargeon <armand.fargeon@etu.unice.fr>
* Mohamed Fertala <mohamed.fertala@etu.unice.fr>

## I. Research context /Project

Nous avons selectionné le sujet numéro 4 : Extraire des informations sur les systèmes de build (variabilité des builds)

Notre question est la suivante : Comment implémente-t-on la configurabilité d’une CI ?

Nous trouvons cette question intéressante car, au vu du grand nombre de combinaisons possibles (avec les types de CPU, les différentes plateformes (x64 et x32) et surtout tous les OS possibles (notamment toutes les versions d’Unix), il faut pouvoir s’y adapter afin de pouvoir livrer un produit disponible pour le plus d’utilisateurs possibles. Cela entraîne beaucoup de contraintes techniques et une charge de travail importante. Il est donc important de pouvoir l’automatiser, mais comment faire ? C’est pourquoi nous trouvons cette question intéressante, notamment dans une ère où la plupart des applications desktop sont en fait des applications webs avec un mini-navigateur intégré → Il faut que ce navigateur puisse fonctionner sur toutes les plateformes afin de profiter de l’avantage de la portabilité du web à son maximum. Ce qui nous amène à d’autres questions telles que comment ces tests sont réalisés, est-ce-que toutes les applications web de ce type (Discord, Slack …) nécessitent des tests multi-plateformes où reposent-elles en grande partie sur les tests effectués par le navigateur utilisé (Firefox, Chromium …) ? Quels tests sont dépendants de la plateforme et comment étendre facilement les tests à plusieurs plateformes (sans trop de réécriture) ?

![Figure 1: Logo UCA, exemple, vous pouvez l'enlever](../assets/model/UCAlogoQlarge.png){:height="25px" }


## II. Observations/General question

1. Commencez par formuler une question sur quelque chose que vous observez ou constatez ou encore une idée émergente. Attention pour répondre à cette question vous devrez être capable de quantifier vos réponses.
2. Préciser bien pourquoi cette question est intéressante de votre point de vue et éventuellement en quoi la question est plus générale que le contexte de votre projet \(ex: Choisir une libraire de code est un problème récurrent qui se pose très différemment cependant en fonction des objectifs\)

Cette première étape nécessite beaucoup de réflexion pour se définir la bonne question afin de poser les bonnes bases pour la suit.

## III. information gathering

Préciser vos zones de recherches en fonction de votre projet,

1. les articles ou documents utiles à votre projet
2. les outils
 
## IV. Hypothesis & Experiences

1. Il s'agit ici d'énoncer sous forme d' hypothèses ce que vous allez chercher à démontrer. Vous devez définir vos hypothèses de façon à pouvoir les _mesurer facilement._ Bien sûr, votre hypothèse devrait être construite de manière à v_ous aider à répondre à votre question initiale_.Explicitez ces différents points.
2. Test de l’hypothèse par l’expérimentation. 1. Vos tests d’expérimentations permettent de vérifier si vos hypothèses sont vraies ou fausses. 2. Il est possible que vous deviez répéter vos expérimentations pour vous assurer que les premiers résultats ne sont pas seulement un accident.
3. Explicitez bien les outils utilisés et comment.
4. Justifiez vos choix

## V. Result Analysis and Conclusion

1. Analyse des résultats & construction d’une conclusion : Une fois votre expérience terminée, vous récupérez vos mesures et vous les analysez pour voir si votre hypothèse tient la route. 

## VI. Tools \(facultatif\)

Précisez votre utilisation des outils ou les développements \(e.g. scripts\) réalisés pour atteindre vos objectifs. Ce chapitre doit viser à \(1\) pouvoir reproduire vos expérimentations, \(2\) partager/expliquer à d'autres l'usage des outils.

## VI. References

1. ref1
1. ref2
