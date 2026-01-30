# Scripts et outils d’analyse

Ce dossier regroupe l’ensemble des scripts utilisés pour l’analyse des évaluations des projets, depuis la collecte et la structuration des notes jusqu’aux analyses statistiques et à l’étude des désaccords entre jurys.

## Scripts d’analyse et de traitement

### `analyse_number_of_contributors.py`

[`codes/analyse_number_of_contributors.py`](./codes/analyse_number_of_contributors.py)

Ce script analyse la distribution du nombre de contributeurs pour l’ensemble des dépôts d’une organisation GitHub (dans notre cas DataForGood).
Il interroge l’API GitHub afin de récupérer tous les repositories, puis calcule précisément le nombre de contributeurs.

À partir de ces données, il produit des statistiques globales (moyenne, médiane, écart-type, minimum et maximum)
et propose une catégorisation des projets selon leur taille.
Les résultats détaillés sont exportés dans un fichier CSV contenant, pour chaque dépôt, le nombre de contributeurs et son URL.

> L’utilisation d’un *GitHub Personal Access Token* est requise afin d’éviter les limitations de l’API.

### `extract_notes.py`

[`codes/extract_notes.py`](./codes/extract_notes.py)

Ce script extrait et restructure les notes brutes des jurys à partir du fichier unique `all.csv`.
Il regroupe les évaluations par juge, par projet et par critère, puis calcule séparément les scores liés aux axes
tests et documentation.

Pour chaque membre du jury, un fichier CSV dédié est généré, contenant le détail des notes par dépôt et par critère.
Ces fichiers constituent la base des analyses statistiques ultérieures et sont enregistrés dans le dossier
`results/1_notes_per_judge/`.

### `statistiques_par_projet.py`

[`codes/statistiques_par_projet.py`](./codes/statistiques_par_projet.py)

Ce script agrège les notes attribuées par les quatre jurys pour chaque projet.
À partir des fichiers de notes individuels, il calcule :

- les scores moyens pour les axes tests et documentation,
- l’ICC (*Intraclass Correlation Coefficient*) afin d’évaluer le niveau d’accord inter-juges.

L’ICC est automatiquement interprété (*poor*, *moderate*, *good*, *excellent*) pour faciliter la lecture des résultats.
Le script génère un fichier CSV récapitulatif par projet, utilisé ensuite comme base pour les analyses par question de recherche.

### `analysis_per_question.py`

[`codes/analysis_per_question.py`](./codes/analysis_per_question.py)

Ce script permet d'analyser et et de répondre aux trois questions de recherche de l’étude.

À partir du fichier `statistiques_par_projet.csv` et du fichier de correspondance `mapping_projets.csv`,
il regroupe les projets selon :

- la période temporelle (Avant / Pendant / Après GenAI),
- le volume de contributeurs (peu / beaucoup),
- le type de projet (AI-related ou non AI-related).

Pour chaque question, il génère des fichiers CSV intermédiaires ainsi que des visualisations synthétiques
(courbes d’évolution temporelle et matrices comparatives).
Les résultats sont sauvegardés dans le dossier `results/3_analysis/` et correspondent aux figures finales de l’étude.

### `mapping_projets.csv`

[`codes/mapping_projets.csv`](./codes/mapping_projets.csv)

Ce fichier CSV définit les métadonnées utilisées pour catégoriser chaque projet analysé.
Il associe à chaque dépôt trois dimensions clés :

- la période temporelle (Avant, Pendant ou Après l’émergence de la GenAI),
- le volume de contributeurs (peu ou beaucoup),
- la nature du projet (AI-related ou non AI-related).

Ce mapping est utilisé par le script `analysis_per_question.py` afin de regrouper les projets
et de produire les analyses comparatives et visualisations finales.
Il constitue un élément central pour relier les résultats statistiques aux questions de recherche.


### `analyse_desaccords_juges.py`

[`codes/analyse_desaccords_juges.py`](./codes/analyse_desaccords_juges.py)

Ce script est dédié à l’analyse des désaccords entre les membres du jury lors de la notation des projets.
À partir des fichiers de notes individuels, il calcule les scores globaux par juge pour les axes tests et documentation,
puis analyse la sévérité relative de chaque juge (moyennes, médianes, écarts-types).

Il mesure également les écarts de notation entre chaque paire de juges et identifie les projets générant le plus de désaccords.
Le script produit des fichiers CSV synthétiques ainsi que des visualisations (graphiques de sévérité et matrices de désaccord),
sauvegardés dans le dossier `results/4_inter_rater_analysis/`.

### `visualisation_detaillee_desaccords.py`

[`.codes/visualisation_detaillee_desaccords.py`](./codes/visualisation_detaillee_desaccords.py)

Ce script génère des visualisations détaillées des notations, projet par projet, à partir des résultats de l’analyse
des désaccords entre juges.
Il produit une représentation de type tableau comparatif, proche d’un export Excel,
montrant les scores attribués par chaque juge ainsi que les moyennes et écarts-types
pour les axes tests et documentation.

Il identifie également, pour chaque projet, les écarts minimum et maximum entre juges
et établit un classement des projets selon leur niveau de désaccord.
Les figures et fichiers CSV générés sont sauvegardés dans le dossier `results/4_inter_rater_analysis/`.

---

## Structure du dossier `results/`

Le dossier `results/` regroupe l’ensemble des données produites au cours du pipeline d’analyse,
depuis les notes brutes des jurys jusqu’aux visualisations utilisées dans l’étude.

### `0_input/`

* **all.csv**
  Fichier source contenant l’ensemble des notes brutes saisies par les membres du jury.

### `1_notes_per_judge/`

* **notes_antoine.csv**, **notes_baptiste.csv**, **notes_roxane.csv**, **notes_theo.csv**
  Fichiers de notes nettoyés et structurés, un par juge, contenant le détail des scores par projet
  et par critère pour les axes tests et documentation.

### `2_statistics/`

* **statistiques_par_projet.csv**
  Fichier récapitulatif, incluant les scores moyens par axe et les ICC par projet.

### `3_analysis/`

Résultats des analyses correspondant aux trois questions de recherche.

#### `question_1/` – Impact de la période temporelle

* **impact_periode.csv**
* **evolution_temporelle.png**

#### `question_2/` – Impact du volume de contributeurs

* **impact_volume_contributeurs.csv**
* **matrice_volume_contributeurs.png**

#### `question_3/` – Impact du type de projet

* **impact_type_ai.csv**
* **matrice_type_projet.png**

### `4_inter_rater_analysis/`

Résultats dédiés à l’analyse de l’accord et des désaccords inter-juges.

#### `csv/`

* **comparaison_detaillee.csv**
* **desaccords_par_projet.csv**
* **differences_min_max_par_projet.csv**
* **differences_paires_tests.csv**, **differences_paires_doc.csv**
* **statistiques_par_juge.csv**

#### `images/`

* **severite_juges.png**
* **matrices_desaccord.png**
* **tableau_scores_detailles.png**
* **classement_desaccords.png**
