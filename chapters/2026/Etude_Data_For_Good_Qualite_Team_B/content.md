---
layout: default
title : Modèle de chapitre pour 2025-26
date:   2026-01
---

**_février 2026_**

## Authors

Nous sommes 4 étudiants en 5ème année à Polytech Nice Sophia de la mineure SSE en Informatique.

* Théo LASSAUNIERE &lt;theo.lassauniere@etu.unice.fr&gt;
* Mathis JULLIEN &lt;mathis.jullien@etu.unice.fr&gt;
* Thibault RIPOCHE &lt;thibault.ripoche@etu.unice.fr&gt;
* Julie SEYIER &lt;julie.seyier@etu.unice.fr&gt;

## I. Contexte de recherche

### Contexte de l'étude

Nous avons réalisé une étude de cas des projets de l'association [Data For Good France](https://github.com/dataforgoodfr/).
Data For Good France est une association existant depuis 2015 dont l'objectif est de créer des projets numériques pour l’écologie, la justice sociale et la démocratie.
L'association possède plus de 160 dépôts de code open source sur GitHub qui sont créés et maintenus par des bénévoles avec des profils divers.

### Pourquoi cette étude est intéressante

Nous trouvons cette étude intéressante car elle explore l’impact que peut avoir un processus de développement réalisé par des contributeurs avec différents niveaux d'expériences sur la qualité des projets d’une organisation open source.
On espère à travers cette étude mieux comprendre comment des développeurs se coordonnent pour maintenir des projets avec un code durable et facile à prendre en main par différents acteurs dans le cadre d’organisation associative.
Nous espérons qu'à travers cette étude nous identifierons des pratiques de code et d'organisation intéressante et utiles dans notre carrière professionelle.

## II. Question générale et sous-questions

### Question générale

**Depuis 2023, quels facteurs liés aux contributeurs influencent la qualité des dépôts de codes de Data For Good France ?**

Dans une étude de cas sur les dépôts de code de l’association Data For Good France, cette problématique nous paraît pertinente car elle permet de mettre en exergue les particularités de ces dépôts qui sont gérés par des bénévoles et basés sur des templates mais dont l'implémentation diffèrent grandement dû aux différents contributeurs impliqués.

### Sous-questions

1. Quel est le niveau de qualité des dépôts de code étudiés ?

Cette sous-question porte l'étude au niveau du code des dépôts en s'intéressant particuliérement à la définition d'une métrique permettant d'évaluer le niveau de qualité.

2. A partir de quel nombre de contributeurs la qualité d’un dépôt de code s’écarte de manière significative de la médiane de l’échantillon ?

Cette sous-question va s'intéresser plus en détails aux contributeurs en les corrélants avec les niveaux de qualité évalué précédement.

3. Comment l’activité des contributeurs (nombre des commits, nature des contributions) influence-t-elle la qualité des dépôts ?

Cette sous-question se pose au niveau des dépôts de code et va évaluer plusieurs facteurs liés aux contributeurs en les comparant avec la qualité des dépôts.

### Nos hypothèses

Notre hypothèse globale est qu'un grand nombre de contributeurs a un impact positif sur la qualité d’un dépôt de code.
Dans notre intuition, plus il y a de contributeurs, plus la nécessité d'une bonne organisation et d'un code de qualité se fait sentir pour permettre de travailler collaborativement.

Pour la _première_ sous-question, notre hypothèse est que la qualité moyenne des dépôts de code sera de moyenne à bonne. 

Pour la _deuxième_ sous-question, notre hypothèse est que plus un projet a de contributeurs, plus les valeurs de qualité seront centrées autour de la médiane, celle-ci évoluant positivement.

Pour la _troisième_ sous-question, notre hypothèse est que plus un projet a de contributeurs, plus on pourra retrouver des commits fix/refactor pour améliorer la qualité d’un dépôt de code.

Consigne : Il s'agit ici d'**énoncer sous forme d'hypothèses** ce que vous allez chercher à démontrer. Vous devez définir vos hypothèses de façon à pouvoir les _mesurer/vérifier facilement._ Bien sûr, votre hypothèse devrait être construite de manière à _vous aider à répondre à votre question initiale_. Explicitez ces différents points.

### Limites éventuelles

Les limites éventuelles de notre étude que nous avons identifié sont : 
- Que le score SonarQube ne reflète pas tous les aspects de la qualité d’un code, juste s’il correspond à un set de règles statiques, il n'est donc pas une garantie totalement fiable d'évaluer la qualité de code de manière objective
- Que les messages des commits ne respectent pas toujours les conventions
- Que le regex pour classifier les messages de commits doit être très correctement défini pour bien capter les tendances
- Que le grand nombre de dépôts de code à l’abandon ou qui ne sont pas des projets parmi ceux de Data For Good France résulte en un petit échantillon de dépôts étudiable
- Nous ne prenons pas en compte les tests dans notre étude car ils sont évalués par l'équipe A.

## III. Collecte d'informations

Préciser vos zones de recherches en fonction de votre projet, les informations dont vous disposez, ... 

Voici quelques pistes : 

1. les articles ou documents utiles à votre projet 
2. les outils que vous souhaitez utiliser
3. les jeux de données/codes que vous allez utiliser, pourquoi ceux-ci, ...

     :bulb: Cette étape est fortement liée à la suivante. Vous ne pouvez émettre d'hypothèses à vérifier que si vous avez les informations. inversement, vous cherchez à recueillir des informations en fonction de vos hypothèses. 
 
## IV. Expérimentation

Nous avons choisi d'utiliser Sonarqube pour évaluer certaines métriques de la qualité de code car il n'est pas présent par défaut sur les dépôts de code de Data For Good France ce qui réduit les biais potentiels et que c'est un standard de l'industrie pour évaluer la santé d'un dépôt.

Vous **explicitez les expérimentations que vous allez mener** pour vérifier si vos hypothèses sont vraies ou fausses. Il y a forcément des choix, des limites, explicitez-les.

     :bulb: Structurez cette partie à votre convenance : 
     Par exemples : 
        Pour Hypothèse 1 => 
            Nous ferons les Expériences suivantes pour la démontrer
        Pour Hypothèse 2 => Expériences 
        
        ou Vous présentez l'ensemble des hypothèses puis vous expliquer comment les expériences prévues permettront de démontrer vos hypothèses.

## V. Analyse & Réponse aux hypothèses

Les graphiques ci-dessous ont été réalisé par nos soins.

### Présentation des résultats

![Figure 1: Nuage de points du ratio du nombre de commits de fix par rapport au nombre de contributeurs](assets/results/ratio_fix_vs_contributeurs.png)

![Figure 2: Nuage de points du ratio du nombre de commits de refactor sur le nombre de commits de feat par rapport au nombre de contributeurs](assets/results/ratio_refactor_feat_vs_contributeurs.png)

### Interprétation et analyse des résultats en fonction des hypothèses

### Conclusion 

Vos résultats et donc votre analyse sont nécessairement limités. Préciser bien ces limites : par exemple, jeux de données insuffisants, analyse réduite à quelques critères, dépendance aux projets analysés, ...

### Ouverture

## VI. Outils utilisés & Reproductibilitée

Précisez votre utilisation des outils ou les développements \(e.g. scripts\) réalisés pour atteindre vos objectifs. Ce chapitre doit viser à \(1\) pouvoir reproduire vos expérimentations, \(2\) partager/expliquer à d'autres l'usage des outils.

## VII. Codes et résultats brutes

Nos résultats bruts sont disponibles ici : [Team B - Results](https://github.com/RIMEL-UCA/RIMEL-UCA.github.io/tree/master/chapters/2026/Etude_Data_For_Good_Qualite_Team_B/results).

Nos codes sont disponibles ici : [Team B - Codes](https://github.com/RIMEL-UCA/RIMEL-UCA.github.io/tree/master/chapters/2026/Etude_Data_For_Good_Qualite_Team_B/codes).

## VIII. Références

[Debret 2020] Debret, J. (2020) La démarche scientifique : tout ce que vous devez savoir ! Available at: https://www.scribbr.fr/article-scientifique/demarche-scientifique/ (Accessed: 18 November 2022).


