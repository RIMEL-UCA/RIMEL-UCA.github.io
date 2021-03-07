---
layout: default
title : Comparaison architecturale physique et logicielle
date:   2021-01-17 14:40:00 +0100
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

We are four students in last year of Polytech' Nice-Sophia specialized in Software Architecture :

* Loïc Bertin &lt;loic.bertin@etu.univ-cotedazur.fr&gt;
* Virgile Fantauzzi &lt;virgile.fantauzzi@etu.univ-cotedazur.fr&gt;
* Guillaume Ladorme &lt;guillaume.ladorme@etu.univ-cotedazur.fr&gt;
* Stéphane Viale &lt;stephane.viale2@etu.univ-cotedazur.fr&gt;

## I. Research context /Project

Notre contexte d'étude se base sur la comparaison des applications de détection et de tracage du COVID-19 et en particulier sur les applications françaises et canadiennes. Nous allons donc nous baser sur l'application TousAntiCovid (ex StopCovid) et l'application CovidShield. 
- Projet canadien : https://github.com/CovidShield/
- Projet français : https://gitlab.inria.fr/stopcovid19

Avec toutes les plaintes autour de la sécurisation de nos données et de la conservation de la vie privée, l'étude de ces deux projets est très intéréssante afin de comprendre si l'inquiétude générale est justifiée ou non. 
De plus, en tant qu'élève architecte logicielle, la rétro ingénierie de projet tels que ceux ci nous permet de comprendre et d'analyser les choix qui ont été fait, à nuancer évidemment avec la rapidité des décisions et les contraintes temporelles dûes à la crise.

## II. Observations/General question

Notre problématique est issue d'une idée qui nous est venu en tant que citoyen français qui commençons à découvrir petit à petit la complexité de notre système. En effet, lors du développement de l'application, les développeurs ont du prendre en compte toutes les nuances du système de santé français afin d'informer les bonnes institutions et de s'interfacer correctement avec les organismes déjà en place. C'est cette complexité qui nous a intrigué et qui nous a donné envie de répondre à cette problématique générale : 
**En quoi l'architecture des projets reflète l'organisation administrative des pays et leur gestion de la crise du covid-19**

Cette problématique va nous permettre de nous intérésser à la fois à l'architecture globale des projets mais aussi de venir investiguer dans le code l'implémentation concrète des mesures gouvernementales. Il y aura donc 2 axes de reflexion à suivre, portés sur différentes échelles de vision de l'architecture, une vision gros grain et un zoom dans le code.
La question étant beaucoup trop générale et impossible à traiter par une équipe de 4 personnes avec le temps accordé, nous nous sommes intéréssés à deux sous questions qui seront détaillées plus bas dans ce rapport. Ces deux sous questions nous paraissent très intéréssante car elles reprennent l'idée générale de la problématique globale mais axée sur de la compréhension des projets et leur comparaison.

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

### En quoi les dépendances externes reflètent l’organisation administrative du pays autour de la crise du COVID-19 ?

#### France

#### Canada

![Figure 1: method claim kay canada](../assets/Physical&LogicalComparisonOfArchitecture/canadaCodeClaimKey.png)

### Comment est implémenté la gestion de la distanciation sociale et des cas contacts dans les applications ?
### Ces implémentations ont-elles évoluées au fil des décisions gouvernementales ?

#### France

#### Canada

## VI. Tools \(facultatif\)

Précisez votre utilisation des outils ou les développements \(e.g. scripts\) réalisés pour atteindre vos objectifs. Ce chapitre doit viser à \(1\) pouvoir reproduire vos expérimentations, \(2\) partager/expliquer à d'autres l'usage des outils.

## VI. References

1. ref1
1. ref2

