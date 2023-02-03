
# Sujet 1 : Quelle qualité logicielle dans les codes des notebooks?
## Authors

Nous sommes quatre étudiants ingénieurs en dernière année à Polytech Nice Sophia, spécialisés en Architecture Logicielle :

* Thomas De Sousa Vieira &lt;thomas.de-sousa-vieira@etu.unice.fr&gt;
* Nicolas Lacroix &lt;nicolas.lacroix@etu.unice.fr&gt;
* Mickaël Lamasuta &lt;mickael.lamasuta@etu.unice.fr&gt;
* Chenzhou Liao &lt;chenzhou.liao@etu.unice.fr&gt;


## I. Contexte de recherche

Les Jupyter Notebook sont de plus en plus populaires, ils sont utilisés dans de nombreux domaines, notamment la recherche en science des données, la formation en programmation et plus généralement dans l'enseignement. Les notebooks permettent de créer et de partager des documents qui combinent à la fois code, résultat d'exécution, explications, graphiques et visualisations en un seul endroit.

Au cours de ces dernières années de nombreux notebooks ont été rédigés et grand nombre d’entre eux peuvent être utilisés comme référence, certains ont pu évoluer au cours du temps et d'autres non. Au vu de cette utilisation croissante, il devient intéressant d'essayer d’analyser si des consensus se sont formés autour de ce format et si la qualité du code est mesurable avec des outils comparable à celle des programmes python traditionnels.


## II. Observations et questions générales

Avec l'augmentation de leur utilisation, il est devenu important de se demander s'il existe des bonnes pratiques à suivre lors de la rédaction de notebooks.

La question qui se pose alors est la suivante :

**Est-il possible d'identifier des bonnes pratiques autour de la rédaction de notebooks ?**

Cela nous amènera aux sous-questions suivantes :

1. **La qualité du code dans les notebooks est-elle mesurable et si c’est le cas est-elle bonne ?**

2. **Est-ce qu'on peut distinguer un motif commun entre les notebooks ?**

3. **La popularité des notebooks est-elle un indicateur du respect d’un éventuel consensus ?**

## III. Collecte d'informations

### Littérature


### Collecte

#### Démarche et postulats

Nous avons dans un premier temps cherché à extraire un maximum de notebook qui pourrait être disponible sur de multiples plateformes. Notre objectif à été de constituer un jeu de données suffisamment important pour obtenir un échantillon représentatif des notebooks. Nous nous sommes concentrés sur les deux grandes plateformes réputées : GitHub et Kaggle. Elles disposent d’une quantité et d’une diversité importantes de notebooks venant d'acteurs variés (data scientist, développeur, spécialiste en IA, etc.).

Notre échantillon témoin, qualifié de “notebooks populaires”, repose sur le nombre d'étoiles des projets hébergés sur le site GitHub et sur le nombre de "upvotes" (comparable en français à des votes favorables) sur les projets hébergés sur le site Kaggle. Le reste de l’échantillon est plus représentatif des notebooks publiés sur les plateformes grâce à une sélection aléatoire.
Notre dataset est constitué de :

- Echantillon Populaire
- Échantillon représentatif 

Nous avons sélectionné GitHub et Kaggle comme références car ces plateformes sont réputées, de par leurs engagements sur les dépôts et leur popularité auprès des data scientists.

#### Méthodologie

Afin de procéder à l'extraction des données, 2 scrapers (en français, un extracteur de données) ont été créés. Écris en Python, ils permettent d'extraire les données provenant de Kaggle et de GitHub en se connectant aux API misent à disposition par les plateformes. Chaque scraper récupère uniquement les notebooks présents sur les répertoires. En plus de la récupération du fichier, un fichier TOML est associé pour y inscrire les métadonnées (URL, nom, date, popularité, etc.) permettant aux étapes suivantes d'identifier le notebook.

![Figure 1 - Processus de collecte de résultats (scraping, analyse, visualisation)](assets/images/collect_pipeline.png "Figure 1 - Processus de collecte de résultats (scraping, analyse, visualisation)")

#### Analyse

Dans un second temps, nous avons cherché à mettre en place un outil d’analyse permettant d’obtenir des métriques à partir du jeu de données extrait précédemment. Pour effectuer notre analyse, nous nous sommes basés sur l'outil d'analyse de code statique Pylint ainsi que de notre propre analyseur structurelle du notebook.
L'analyse produit autant de fichiers JSON que de notebooks analysés. Ceux-ci contiennent un score qualité (produit par Pylint), le nombre d'erreurs identifiés par Pylint et le nombre de cellules de code et de markdown ainsi que leur fréquence.

#### Visualisation

Grâce aux fichiers JSON générés, il est possible d'en obtenir des visualisations. Ces visualisations, écrites en Python, lisent les fichiers JSON pour en créer un graphique à l'aide de la librairie Matplotlib.

## IV. Hypothèse et expériences

Tout d'abord, nous avons fait l'hypothèse forte que la popularité des notebooks nous servira de référence dans l’identification des bonnes pratiques. Pour cela, nous avons orienté nos sources de données vers les fichiers venant de références (Jupyter Notebook, Google research, Tensorflow et Pandas).

Puis, d’une manière plus générale, nous faisons l'hypothèse que la qualité d’un Jupyter Notebook est mesurable à la manière d’un code source Python traditionnel. Nous appliquerons alors des outils d’analyses standards tels que l’analyse statique Pylint ou encore une analyse structurelle du Jupyter Notebook (enchaînement de cellules de markdown/code) qui nous donneront un score pouvant être assimilé aux critères de qualité de code.

### Hypothèse 1 : la qualité du code dans les notebooks est bonne.

#### Expérience

Il s'agit d'identifier les notebooks exploitables pour le reste des expériences. Nous identifions ici les notebooks ayant un score Pylint acceptables, c'est-à-dire un score strictement supérieur à 0. Nous appliquons cette sélection afin de ne pas impacter les résultats suivants.

Enfin, parmi ces notebooks exploitables, il s'agit d'identifier la répartition des scores Pylint pour en connaître la tendance. Cette répartition s'intéressera en particulier aux notebooks populaires, c'est-à-dire les dépôts ayant le plus d'engagements sur GitHub et Kaggle.

#### Résultats

![Figure 2 - Résultat de l'analyse pour l'hypothèse 1 (nb pylint score == 0)](assets/images/hypothesis_1_result_pylint_score_0.png "Figure 2 - Résultat de l'analyse pour l'hypothèse 1 (nb pylint score == 0)")

Parmi les 626 notebooks extraits de GitHub, seuls 223 sont considérés comme exploitables. Cela présente un ratio d'exploitation d'environ 35,62%.

Et, parmi les 299 notebooks provenant de Kaggle, 246 notebooks ont un score Pylint satisfaisants. Le ratio d'exploitation est ici supérieur, avec environ 82,27% des notebooks exploitables.
Bien que Kaggle ait un ratio d'exploitation plus élevé, lorsqu'on s'intéresse, dans le détail, aux scores des notebooks exploitables, la répartition des scores est relativement partagée.

![Figure 3 - Résultat de l'analyse pour l'hypothèse 1 (pylint scores)](assets/images/hypothesis_1_result_pylint_scores.png "Figure 3 - Résultat de l'analyse pour l'hypothèse 1 (pylint scores)")

Cependant, on retrouve une majorité de notebooks avec un score Pylint compris entre 4 et 7. Entre 6 et 10, le nombre de notebooks ayant un score élevé décroît, ce qui signifie que l'obtention d'une bonne qualité Pylint est difficile.

### Hypothèse 2 : on peut distinguer un motif commun entre les notebooks.

#### Expérience

La structure d’un Jupyter Notebook est un élément important (équitabilité des cellules de markdown/code). Nous avons fait l’hypothèse qu’une structure commune peut être dégagée de l’ensemble des notebooks. Pour cela, nous avons sourcé des notebooks de popularité différente pouvant représenter un échantillon représentatif des notebooks déposées sur les différentes plateformes en ligne (GitHub et Kaggle). Pour notre expérience nous avons développé un analyseur syntaxique permettant d’obtenir la répartition entre code et markdown.

#### Résultats

![Figure 4 - Résultat de l'analyse pour l'hypothèse 2 (nombre cellules markdown/code entre github et kaggle)](assets/images/hypothesis_2_result_nb_markdowns_cells_github_kaggle.png "Figure 4 - Résultat de l'analyse pour l'hypothèse 2 (nombre cellules markdown/code entre github et kaggle)")

Le diagramme ci-dessus nous renseigne sur les tendances entre GitHub et Kaggle, permettant de situer l’équilibre entre cellules de code et de markdown.

Parmi l’ensemble des notebooks représentés, nous pouvons constater qu’un certain équilibre est trouvé entre cellule de code et de markdown. En effet, le consensus voudrait qu’avant chaque bloc de code, une explication préphase ce qu’il va suivre pour expliquer au mieux le déroulé du programme. C’est bien ce que nous pouvons en partie constater, en majorité il y a autant de bloc de code que de bloc de markdown dans les notebooks. De ce diagramme il est compliqué d’affirmer qu’il y ait bien cette alternance entre code et markdown mais cet équilibre pourrait indiquer ce modif structurel.

### Hypothèse 3 : les notebooks populaires sont de qualités.

#### Expérience

L'objectif de cette expérience est d'être en mesure de savoir s'il existe une différence notable de qualité entre un notebook populaire et un notebook non populaire.

#### Résultats

![Figure 5 - Résultat de l'analyse pour l'hypothèse 3 (nombre cellules markdown/code entre populaires et moins populaires)](assets/images/hypothesis_3_result_nb_markdowns_cells_popular_lower.png "Figure 5 - Résultat de l'analyse pour l'hypothèse 3 (nombre cellules markdown/code entre populaires et moins populaires)")

Concernant l’équilibre code/markdown, nous constatons que la popularité des notebooks n'influence pas la qualité structurelle d'un notebook. Qu'un notebook soit populaire ou non, la tendance est équivalente.

Il s'agira ici d'observer la qualité Pylint et la structure de chaque notebook.

![Figure 6 - Résultat de l'analyse pour l'hypothèse 3 (score pylint entre populaires et moins populaires)](assets/images/hypothesis_3_result_pylint_scores_popular_lower.png "Figure 6 - Résultat de l'analyse pour l'hypothèse 3 (score pylint entre populaires et moins populaires)")

Toutefois, la qualité relevée avec l'outil Pylint démontre que la qualité du code Python chez les notebooks populaires est globalement plus élevée que les notebooks non populaires. En effet, entre 0 et 2 de score, la majorité des notebooks est non populaire. Et, entre 5 et 10 de score, la majorité des notebooks est populaire.

Cette expérience permet de constater que malgré l'absence de différence  structurelle d'un notebook populaire à un notebook non populaire, la qualité de code est quant à elle significative.

## V. Résultat d'analyse et conclusion

### Interprétation

Les notebooks ont pour la plupart un score moyen avec l'outil Pylint. Il existe cependant des différences notables de qualité Pylint lorsqu'on s'intéresse à la popularité des notebooks. En effet, un notebook populaire aura de plus grande chance d'être de meilleure qualité qu'un notebook non populaire.
Concernant la qualité structurelle (proportionnalité de cellules de markdown/code), les notebooks tendent à adopter 1,3 à 1,6 cellule de code pour 1 cellule de markdown.

### Prise de recul

### Limites

## VI. Outils

Stack technique:

* Python 3.10 et pipenv

Organisation des scripts de l’exploitation:

* main.py: script principal faisant office de command line
* scripts/github/: scraper.py pour la récupération de notebooks sur GitHub et fichiers textes contenant les URLs des différents dépôts
* scripts/kaggle/: scraper.py pour la récupération de notebooks sur Kaggle et fichiers textes contenant les URLs des différents notebooks
* scripts/analysis.py: script analysant les notebooks (principalement avec Pylint) récupérés grâce aux scrappers
* scripts/diagrams/: scripts dédiés à la génération de graphiques pour visualiser les résultats de l’analyse

Organisation des résultats de l’exploitation:

* notebooks/github/ et notebooks/kaggle/: ces deux répertoires contiennent les notebooks extraits selon la source
     * Chaque notebook collecté (.ipynb) est associé à un fichier de métadonnées (.toml)
* results/: répertoire contenant les résultats de l’analyse sous la forme de fichiers JSON au format {auteur}-{nom_du_notebook}.toml.json


![Figure 1: Logo UCA, exemple, vous pouvez l'enlever](assets/images/logo_uca.png){:height="25px"}


## VI. References

[Debret 2020] Debret, J. (2020) La démarche scientifique : tout ce que vous devez savoir ! Available at: https://www.scribbr.fr/article-scientifique/demarche-scientifique/ (Accessed: 18 November 2022).
[Pimentel2019] Pimentel, João Felipe, et al. “A Large-scale Study about Quality and Reproducibility of Jupyter Notebooks.” 2019:
http://www.ic.uff.br/~leomurta/papers/pimentel2019a.pdf

[DanielEtzold] Best Practices for Writing Reproducible and Maintainable Jupyter Notebooks:
https://towardsdatascience.com/best-practices-for-writing-reproducible-and-maintainable-jupyter-notebooks-49fcc984ea68

Weinmeister, Karl. “Jupyter Notebook Manifesto: Best practices that can improve the life of any developer using Jupyter notebooks.” Google Cloud, 12 June 2019, https://cloud.google.com/blog/products/ai-machine-learning/best-practices-that-can-improve-the-life-of-any-developer-using-jupyter-notebooks

[NotebookFormat] Format des notebooks Jupyter : https://ipython.readthedocs.io/en/3.x/notebook/nbformat.html

[Casseau2021a] Casseau, C. et al. (2021) ‘Immediate Feedback for Students to Solve Notebook Reproducibility Problems in the Classroom’, Proceedings of IEEE Symposium on Visual Languages and Human-Centric Computing, VL/HCC. IEEE Computer Society, 2010-Octob. doi: 10.1109/VL/HCC51201.2021.9576363. voir sur lms

[Chattopadhyay2020] Chattopadhyay, S. et al. (2020) ‘What’s wrong with computational notebooks? Pain points, needs, and design opportunities’, in {CHI} ’20: {CHI} Conference on Human Factors in Computing Systems. Honolulu, HI, USA: Association for Computing Machinery, pp. 1--12. doi: 10.1145/3313831.3376729.. voir sur lms

[Head2019] Head, A. et al. (2019) ‘Managing Messes in Computational Notebooks messy notebook execution time’, in Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems. Glasgow, Scotland, UK: ACM, pp. 1–12. doi: 10.1145/3290605.. voir sur lms

[Quaranta2022] Quaranta, L., Calefato, F. and Lanubile, F. (2022) ‘Eliciting Best Practices for Collaboration with Computational Notebooks’, Proceedings of the ACM on Human-Computer Interaction. ACM PUB27 New York, NY, USA, 6(CSCW1). doi: 10.1145/3512934.. voir sur lms

[Rule2019] Rule, A. et al. (2019) ‘Ten simple rules for writing and sharing computational analyses in Jupyter Notebooks’, PLoS Comput. Biol., 15(7), pp. 1--8. doi: 10.1371/journal.pcbi.1007007.. voir sur lms
