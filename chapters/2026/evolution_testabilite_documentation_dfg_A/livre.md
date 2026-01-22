# Chapitre de livre

> Dans le rendu sur moodle, un étudiant du groupe, met uniquement le lien vers le chapitre de son équipe sous https://rimel-uca.github.io/
> 

> Voici quelques-uns des critères utilisés pour évaluer les chapitres
> 

> Critères d'évaluation :
> 
> 1. Format : structure, forme adaptée au contenu d'un livre en ligne…
> 2. Contexte dont la motivation
> 3. Observations/Question générale : formulation, intérêt, limites éventuelles.
> 4. Quelle est votre base d'information
> 5. Quelles sous-questions et hypothèses et expériences, vous allez mener
>     1. Quelles expérimentations, démarches choisies pour vérifier ou non vos hypothèses, justifications.
>     2. Quels outils sont utilisés (référence si besoin aux scripts)
>     3. Justification des choix
> 6. Résultats
>     1. visualisations
>     2. Analyse des résultats (et des limites si nécessaires)
>     3. conclusion relativement aux hypothèses
> 7. Conclusion globale intégrant les perspectives à ce travail

# Introduction

Ce chapitre présente notre étude de la maintenabilité sur les projet de dataforgoodfr. Nous étudierons celle-ci sous l’aspect de la qualité des tests et de la qualité de la documentation. Ces 2 critères sont primordiaux pour assurer la qualité et donc la continuité d’un projet.  

# Objectifs

Notre projet est voué à :

- Définir une méthodologie pour analyser les tests et la documentation d’un projet,
- Appliquer cette méthodologie aux dépôts de Data For Good,
- Poser des hypothèses concernant l’ensemble des projets des projets de DFG basées sur les dépôts analysés
- Produire des recommandations dédiées à chaque problèmes repérés dans l’interprétation.

Ce protocole permettra d’analyser de répondre à la question

> Comment varie la qualité de tests & de la documentation du code dans les projets de Data For Good ?
> 

Nous y répondrons en étudiant indépendamment plusieurs critères pouvant avoir un impact sur la qualité des tests et sur la qualité de documentation. Ceux ayant retenu notre attention sont : l’utilisation d’AI générative dans la réalisation du projet, le nombre de contributeurs ayant participer à celui-ci et son lien ou non avec l’IA. Nous cherchons donc à répondre aux questions suivantes :

1. L’entrée de l’IA générative dans le quotidien a-t-elle eu un impact dans ces deux domaines ?
2. Quel impact le nombre de collaborateurs a-t-il sur la qualité des tests et la documentation ?
3. Quelles différences de qualité de tests sont présentes entre les projets incluant des composants d'IA et les autres ?

# Méthodologie

## Choix des dépôts

Les dépôts à analyser ont été choisis de manière aléatoire tout en s’assurant d’avoir un projet par triplet de catégorie : réalisation par rapport à la sortie public de l’IA générative, nombre de collaborateurs, lien avec des composants d’IA.

### Catégories de la question 1

La première question cherche à étudier l’impact de l’IA générative sur les projets de dataforgoodfr. À défaut de pouvoir analyser précisément quels projets ont utilisé de tels outils, nous nous sommes basé pour cette étude sur la capacité des développeurs à utiliser l’IA générative. En nous basant sur la courbe [trends.google.com](http://trends.google.com) des termes “chatgpt", “chat gpt” et “gpt” disponible ci-dessous. Ces termes concernent tous l’outil leader de l’IA générative ChatGPT.com. 

![image.png](attachment:a5760369-f6c8-4c9b-89a1-0d9757f42e22:f53d7461-e759-4dfa-b959-1d46d6f11c22.png)

Celle-ci met en évidence deux dates : novembre 2022, date de publication du site [ChatGPT.com](http://ChatGPT.com) et août 2024 qui semble correspondre à une nouvelle hause d’utilisation de la plateforme. Ces dates, par extension, indiquent les phases qu’à connu l’IA générative : avant l’entrée dans le quotidien de la GenAI, pendant cette entrée et après l’admission à grande échelle de cette technologie.

Après vérification de la répartition de l’ensemble des projets de dataforgoodfr projets, nous avons pu observer que leurs dates d’activités sont quasiment répartie avec un tier des projets avant l’émergence de l’IA générative, un tier pendant son émergence et un dernier tier après celle-ci. Cette répartition justifie ainsi de prendre le même nombre de projet pour chacune de ses catégories dans notre échantillon.

### Catégories de la question 2

La deuxième question permet de se pencher sur l’impact du nombre de contributeurs sur les projets. Nous avons développer un script python (‼️disponible ici ‼️) qui nous a permis de calculer des statistiques sur le nombre de contributeurs des projet de l’organisation dataforgoodfr. les résultats sont présents ci-dessous.

![image.png](attachment:9eb9c0e6-345a-4c77-8794-f4aac3fe497f:28c11324-f7a3-4408-858c-4a3283ff2477.png)

On peut voir que la moyenne du nombre de contributeurs sur un dépôt est de 5.54. Nous considérons donc les projet ayant 5 contributeurs ou moins comme des petits projets et ceux ayant 6 ou plus comme des gros projets.

### Catégories de la question 3

La dernière question concerne la proximité du projet avec des composant d’IA. Les dépots peuvent ainsi être AI-related ou ne pas l’être. Sont définis comme en rapport avec l’IA les projets permettant d’entraîner des réseaux de neurones, ceux utilisant des APIs d’IA, ou encore ceux qui concerne des RAGs.

Ainsi nous avons choisi 12 dépôts car chaque projet peut : 

- s’être passé avant/pendant/après la sortie de l’IA générative (trois possibilités),
- avoir été réalisé par peu ou beaucoup de contributeurs (deux possibilités),
- concerner ou pas des composants d’IA (deux possibilités).

Nous aurons donc 4 projet par catégorie de la question 1, 6 par catégorie de la question 2 et 6 par catégorie de la question. 

### Projets choisis

- **Avant GenAI (… < 12/2022 )**

|  | **Peu de contributeurs** | **Beaucoup de contributeurs** |
| --- | --- | --- |
| **AI-related** | [offseason_missiontransition_categorisation](https://github.com/dataforgoodfr/offseason_missiontransition_categorisation/tree/main) | [batch4_diafoirus_fleming](https://github.com/dataforgoodfr/batch4_diafoirus_fleming) |
| **Non AI-related** | [website2022](https://github.com/dataforgoodfr/website2022) | [batch5_phenix_happymeal](https://github.com/dataforgoodfr/batch5_phenix_happymeal) |
- **Pendant GenAI (12/2022 < … < 08/2024 )**

|  | **Peu de contributeurs** | **Beaucoup de contributeurs** |
| --- | --- | --- |
| **AI-related** | [batch11_cartovegetation](https://github.com/dataforgoodfr/batch11_cartovegetation)  | [bechdelai](https://github.com/dataforgoodfr/bechdelai) |
| **Non AI-related** | [protectclimateactivists](https://github.com/dataforgoodfr/protectclimateactivists) | [offseason_ogre](https://github.com/dataforgoodfr/offseason_ogre) |
- **Après GenAI ( 08/2024 < … )**

|  | **Peu de contributeurs** | **Beaucoup de contributeurs** |
| --- | --- | --- |
| **AI-related** | [13_ia_financement](https://github.com/dataforgoodfr/13_ia_financement) | [13_democratiser_sobriete](https://github.com/dataforgoodfr/13_democratiser_sobriete) |
| **Non AI-related** | [14_PrixChangementClimatique](https://github.com/dataforgoodfr/14_PrixChangementClimatique/tree/main)  | [shiftdataportal](https://github.com/dataforgoodfr/shiftdataportal) |

## Notation

Chaque projet sera noté selon la qualité de sa notation et selon la qualité de ses tests. Nous sommes 4 jury, nous utiliserons tous les 4 les mêmes grilles de notations décrite ci-dessous.

### **Description des critères**

Afin d’assurer la qualité des notes et la reproductibilité de notre protocoles, nous avons fixé ensemble des définitions claires et précises des nos axes d’études : la qualité des tests et la qualité de la documentation. 

**Critères de la qualité de la documentation**

- Présence d’un README.md incluant
    - **Description fonctionnelle du projet** permettant de comprendre l'intérêt du repo
    - **Explication de l’architecture du repo** justifiant l'arborescence des fichiers
    - **Des instructions d’installation** mentionnant potentiellement la configuration requise et surtout le protocole pour exécuter le projet
    - **Les cordonnées des contributeurs** pour permettre d'avoir des informations supplémentaires
- Un **CHANGELOG.md** traçant l'historique des modifications faites sur le depot
- Un **CONTRIBUTING.md** expliquant la stratégie de branche, la convention de commit, les règles de pull request/fork
- Un fichier **LICENCE.md est présent** et décrit la licence protégeant le projet

**Critères de la qualité de tests**

- Des **tests sont présents**, dans un dossier tests, des fichiers avec l'extension de test selon le langage utilisé
- Des test de **niveaux de test pertinent sont présents** selon les besoins (unitaires, intégration, de bout en bout)
- Les **tests sont de bonne qualité** s’ils sont :
    - Lisibles
    - Bien nommés
    - Pertinents
    - Utiles
- Le pourcentage de **coverage des test est indiqué** (5 points), la couverture des tests (si indiqué ou mesurable) **est bon (**de 0 à 20)
- Les **tests sont entretenus dans le temps**, la date de modification d'un fichier de tests est égale ou ultérieure à celle du fichier testé

### **Barèmes de notation**

Une fois les définitions fixée, nous avons attribué un nombre de point maximal à chacun des critères en fonction de leur importance et la profondeur attendue en nous basant sur nos avis de futurs ingénieurs logiciels.

**Barème de la qualité de la documentation**

| **Critère** | **Nombre de point** |
| --- | --- |
| README description fonctionnelle du projet | 25 |
| README Explication architecture technique | 20 |
| README instruction installation | 25 |
| README nom et contact des contributeurs | 5 |
| CHANGELOG à jour | 10 |
| CONTRIBUTING complet | 10 |
| LICENCE présente | 5 |

**Barème de la qualité des tests**

| Critères | Poids |
| --- | --- |
| Présence de tests | 15 |
| Niveaux présents (UT/IT/E2E) | 25 |
| Qualité des tests | 25 |
| Coverage dispo + couverture | 25 |
| Entretien dans le temps (Git) | 10 |

## Moyennage des notes

Une fois chaque projet noté avec les critères définis, nous avons sommé les notes de chaque axe d’étude pour chaque jury afin d’avoir la note moyenne par axe par projet par jury.

 À partir des fichiers mentionnés ci-dessus, nous avons agrégé en une moyenne.

Une fois chaque projet noté avec les critères définis, les notes de tous les jurys sont agrégés en une moyenne des notes pour la qualité de la documentation et une moyenne pour la qualité des tests à l’aide ainsi qu’un [coefficient de corrélation intraclasse](https://en.wikipedia.org/wiki/Intraclass_correlation) qui permet de juger l’accord entre les jurys. Nous avons donc développer un script permettant de faire l’ensemble des calculs pour chaque projet et ainsi obtenir pour chaque projet :

- La moyenne de notes de qualité des tests
- Le coefficient ICC sur la moyenne des notes de test
- La moyenne de notes de qualité de la documentation
- Le coefficient ICC sur la moyenne des notes de la documentation

Enfin nous pourrons utiliser ces moyennes pour répondre à chacune de nos questions en les groupant selon les critères de la question. Par exemple, pour la question 2, les 6 projets avec peu de collaborateur verront leur moyennes analysées comme une seule et les 6 autres ensemble aussi.

## Limites

### Portée de l’étude

On ne peut pas analyser l’intégralité des dépôts pour produire des conclusion visant l’ensemble des projets de Data For Good France par manque de temps dans le cadre de cette étude. Plusieurs aspects de notre méthodologie doivent être automatiser pour atteindre un tel but :

- La classification de chaque projet selon les critères mentionnés dans chaque question (avant/pendant/après l’entrée de l’IA générative dans le quotidien, le nombre de collaborateur, le lien du projet avec l’IA)
- L’application de la notation de la qualité de la documentation
- L’application de la notation de la qualité des tests

### Comptage du nombre de contributeurs

Notre script de comptage du nombre de contributeurs par projet ne prend malheureusement pas en compte les contributeurs utilisant plusieurs comptes différents, car en effet le script compte le nombre de compte différents ayant commit sur le dépôt.

# ‼️ ITERATION ICC (ET SI ON A L’HISTORIQUE DES ITERATIONS)‼️

# Résultats

L’ensemble des notes brutes sont disponibles ici.

Les notes moyennes de chaque dépôts sont été calculées et sont disponibles ici. 

## Question 1

## Question 2

## Question 3