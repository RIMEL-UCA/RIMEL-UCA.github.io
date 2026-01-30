# Analyse de l’impact des contributeurs sur la qualité des dépôts de code

## Étude de cas : *Data For Good France*

---

## Contexte et problématique

Depuis 2023, quels facteurs liés aux contributeurs influencent la qualité des dépôts de code de l’organisation **Data For Good France** ?

Cette étude vise à analyser les pratiques de développement récentes de l’association afin de mieux comprendre comment :

* la **taille des équipes**,
* l’**activité des contributeurs**,
* la **nature des contributions**

impactent la **qualité du code** produit dans un contexte associatif, bénévole et hétérogène.

L’analyse se concentre exclusivement sur le **code source** des dépôts GitHub. Les tests et pratiques associées font l’objet du travail d'un autre groupe.

---

## Objectifs de l’étude

L’étude s’articule autour de trois axes principaux :

1. **Qualité du code**
   Évaluer et comparer le niveau global de qualité des dépôts de code étudiés.

2. **Nombre de contributeurs**
   Identifier à partir de quel seuil de contributeurs la qualité d’un dépôt s’écarte significativement de la médiane observée pour des projets comparables.

3. **Activité de contribution**
   Analyser comment le volume et la nature des contributions (feat, fix, refactor, etc.) influencent la qualité des dépôts.

---

## Sélection des dépôts étudiés

Afin d’obtenir un échantillon pertinent et représentatif, les dépôts analysés respectent les critères suivants :

* dernier commit datant de **2023 ou plus récent** ;
* appartenant à une initiative **Saison 10 ou plus récente**, **batch-6 ou plus récent**, ou **hors-saison** (avec activité sur l’année en cours) ;
* **au moins 20 commits** ;
* exclusion des templates, dépôts quasi vides et du site web de l’association.

Ces critères permettent de se concentrer sur des projets encore maintenus, suffisamment développés et représentatifs des pratiques actuelles de Data For Good France.

L’échantillon final est composé de **22 dépôts de code**.

---

## Définition de la qualité d’un dépôt de code

La qualité est évaluée à partir de cinq dimensions complémentaires :

* **Fiabilité** : présence de bugs et impact fonctionnel
* **Maintenabilité** : code smells et dette technique
* **Sécurité** : vulnérabilités détectées
* **Duplication** : proportion de code dupliqué
* **Complexité cognitive** : lisibilité et compréhension du code

Les métriques de fiabilité, maintenabilité et sécurité sont issues de **SonarQube**, en utilisant ses seuils industriels reconnus. Les métriques de duplication et de complexité cognitive sont transformées en scores numériques afin d’obtenir une échelle homogène.

Ces cinq dimensions sont ensuite agrégées (avec pondération) pour produire un **score global de qualité sur 100**, comparable entre dépôts.

---

## Pipeline d’analyse automatisée

L’ensemble de l’étude repose sur une **pipeline entièrement automatisée et dockerisée**, garantissant :

* la reproductibilité des résultats,
* l’indépendance vis-à-vis du système d’exploitation,
* la traçabilité des données et des traitements.

La pipeline couvre les étapes suivantes :

1. collecte des données GitHub,
2. calcul du nombre de contributeurs par dépôt,
3. analyse des types de commits,
4. génération de visualisations graphiques.

---

## Lancer l’analyse

### Prérequis

* Docker
* Docker Compose

### Exécution

Depuis la **racine du dépôt**, lancer simplement :

```bash
./run_pipeline.sh
```

Le script orchestre automatiquement l’ensemble de la chaîne de traitement. Certaines étapes (notamment la collecte des données brutes GitHub) ne sont exécutées que si les fichiers nécessaires ne sont pas déjà présents, afin d’éviter des calculs inutiles.

---

## Consulter les résultats

À l’issue de l’exécution de la pipeline, les résultats sont disponibles sous forme de fichiers CSV et de graphiques :

* **Nombre de contributeurs par dépôt**
  `2-nombre-contributeurs/data/contributors.csv`

* **Répartition des types de commits**
  `3-activite-contributeurs/data/commits_types.csv`

Les graphiques produits permettent notamment :

* d’analyser le lien entre le nombre de contributeurs et la proportion de commits correctifs
* d’étudier le ratio entre refactorisation et ajout de fonctionnalités en fonction de la taille des équipes
* de comparer l’activité des projets selon différents profils de contribution

---

## Références

* *Software Quality Measurement: A Systematic Review*
  [https://link.springer.com/article/10.1007/s11219-011-9144-9](https://link.springer.com/article/10.1007/s11219-011-9144-9)

* *Assessing Software Quality*
  [https://onlinelibrary.wiley.com/doi/full/10.1046/j.1365-2575.2002.00117.x](https://onlinelibrary.wiley.com/doi/full/10.1046/j.1365-2575.2002.00117.x)

---

## Perspectives

Les travaux futurs pourront inclure :

* une analyse temporelle de l’évolution de la qualité,
* l’intégration des données issues des issues et pull requests,
* une comparaison avec d’autres organisations open-source.
