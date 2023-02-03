# Sujet 1 : Quelle qualité logicielle dans les codes des notebooks?

## Auteurs

Nous sommes quatre étudiants ingénieurs en dernière année à Polytech Nice Sophia, spécialisés en Architecture Logicielle :

* Thomas De Sousa Vieira &lt;thomas.de-sousa-vieira@etu.unice.fr&gt;
* Nicolas Lacroix &lt;nicolas.lacroix@etu.unice.fr&gt;
* Mickaël Lamasuta &lt;mickael.lamasuta@etu.unice.fr&gt;
* Chenzhou Liao &lt;chenzhou.liao@etu.unice.fr&gt;

## I. Contexte de recherche

Les Jupyter Notebook sont de plus en plus populaires. Ils sont utilisés dans de nombreux domaines, notamment la recherche en science des données, la formation en programmation et plus généralement dans l'enseignement. Les notebooks permettent de créer et de partager des documents qui combinent à la fois code, résultats d'exécution, explications et visualisations en un seul endroit.

Au cours de ces dernières années, de nombreux notebooks ont été rédigés et un certain nombre d’entre eux font office de référence pour la recherche ou le développement. Au vu de cette utilisation croissante, il devient intéressant d’analyser si des consensus se sont formés autour de ce format et si la qualité du code est mesurable avec des outils comparables à ceux employés pour des scripts Python plus “classiques”.

## II. Observations et questions générales

La rédaction de notebooks est devenue une pratique de plus en plus courante dans de nombreux domaines, notamment dans le monde de la science des données et de l'analyse de données. Avec l'augmentation de leur utilisation, il est devenu important de se demander s'il existe des bonnes pratiques à suivre lors de la rédaction de notebooks.

La question qui se pose alors est la suivante :

**Est-il possible d'identifier des bonnes pratiques autour de la rédaction de notebooks ?**

Cette étude répondra donc aux sous-questions suivantes :

1. **La qualité du code dans les notebooks est-elle mesurable ? Auquel cas, est-elle bonne ?**

2. **Peut-on distinguer un motif commun entre les notebooks ?**

3. **La popularité des notebooks est-elle un indicateur du suivi d’un éventuel consensus ?**

## III. Collecte d'informations

### Littérature

Tout d’abord, nous avons analysé la littérature scientifique sur le sujet afin d'avoir une première idée des résultats attendus.

Nous avons donc sélectionné les articles suivants traitant de la qualité et de la reproductibilité:

L’article de [Pimentel2019] fournit une analyse détaillée de plus d'un million (1 053 653 analysées) de notebooks. Il a trouvé que seuls 24,9% d'entre eux ont pu être réexécutés sans ordre de passage ambigu, et ce chiffre tombe à 4% lorsque deux exécutions donnaient le même résultat. Cela montre que la reproductibilité des notebooks laisse une grande marge de progression et que sa qualité est liée à de nombreux facteurs. Lors de simulations de reproductibilité, les exceptions les plus courantes sont "ImportError" et "NameError". Cela signifie donc que c'est une bonne pratique d'utiliser des déclarations de dépendances explicites telles que !pip install, requirements.txt ou setup.py.

Un autre article de 2021 [Pimentel2021] s'en inspire et résume, en tant que bonne pratique : 69,07% des notebooks ont des cellules Markdown. 70,90% des notebooks qui ont des boucles ou des structures de condition ont aussi des définitions de fonctions. Il est à noter également que la plupart des notebooks ont des noms de fichiers descriptifs.

En revanche, parmi les mauvaises pratiques qui affectent la qualité, l'auteur a trouvé très peu de notebooks important des modules de test connus. De plus, beaucoup ont des cellules désordonnées (phénomène défini comme un 'split' étant une réexécution, une édition ou la suppression d'une même cellule à la suite d'une première exécution [Pimentel2019]) avec parfois du code non exécuté. En résumé, plusieurs raisons fatales amène à un notebook non reproductible : manque de dépendances, données inaccessibles, exécution désordonnée.

Enfin, un blog [DanielEtzold] récent (13/07/2022) suggère plusieurs bonnes pratiques :
* Utiliser des noms expressifs ;
* Éviter les ordres d'exécution ambigus ;
* Utiliser des modules de test ;
* Rendre le notebook modulaire ;
* Distribuer toutes les données utilisées dans le notebook avec le notebook ;
* Distribuer un notebook avec ses résultats ;
* Créer des fichiers de dépendance.

Il faut également noter que les notebooks sont utilisés de plus en plus dans de grands projets d'ingénierie. [Jupyter Notebook Manifesto](https://cloud.google.com/blog/products/ai-machine-learning/best-practices-that-can-improve-the-life-of-any-developer-using-jupyter-notebooks?hl=en) de Google résume également les bonnes pratiques dans les environnements de production : 

* Faire attention à la POO ;
* Des guides de style et la documentation comme l'ingénierie logicielle ;
* Établissement d'un contrôle de version ;
* Paramétrage des notebooks ;
* CI/CD pour les notebooks.

### Collecte

#### Démarche et postulats

Nous avons dans un premier temps cherché à extraire un maximum de notebooks qui pourraient être disponibles sur de multiples plateformes. Notre objectif à été de constituer un jeu de données suffisamment important pour obtenir un échantillon représentatif des notebooks. Nous nous sommes concentrés sur les deux plateformes populaires : GitHub et Kaggle. En effet, ces dernières disposent d’une quantité et d’une diversité importante de notebooks venant d'acteurs variés (data scientists, développeurs experts et amateurs, développeurs de librairies de Data Science, etc.).

Notre premier échantillon est constitué de “notebooks populaires”. Cette qualification de popularité repose sur le nombre d'étoiles des projets hébergés sur le site GitHub et sur le nombre de "upvotes" (comparable en français à des votes favorables) sur les projets hébergés sur le site Kaggle. Un second échantillon est constitué de notebooks provenant de dépôts Github considérés comme des “références” (indépendamment de la notion de popularité) au sein de la communauté de la Science des données. On retrouve ainsi entre autres des notebooks écrits par les organisations Jupyter, Tensorflow, Pandas. Le reste de l’échantillon est constitué grâce à une sélection aléatoire afin de représenter la diversité des notebooks publiés sur ces plateformes.

Notre dataset est donc constitué de :

* Échantillon des notebooks populaires
* Échantillon des références
* Échantillon aléatoire représentatif

#### Méthodologie

Afin de procéder à l'extraction des données, 2 scrapers (en français, un extracteur de données) ont été créés. Développés en Python, ils permettent d'extraire les données provenant de Kaggle et de GitHub en se connectant aux API mises à disposition par les plateformes. Chaque scraper récupère uniquement les notebooks présents sur les répertoires. En plus de la récupération du fichier, un fichier TOML est associé pour y inscrire les métadonnées (URL, nom, date, popularité, etc.) permettant aux étapes suivantes d'identifier le notebook.

![Figure 1 - Processus de collecte de résultats (scraping, analyse, visualisation)](assets/images/collect_pipeline.png "Figure 1 - Processus de collecte de résultats (scraping, analyse, visualisation)")

#### Analyse

Dans un second temps, nous avons cherché à mettre en place un outil d’analyse permettant d’obtenir des métriques à partir du jeu de données extrait précédemment. Pour effectuer notre analyse, nous nous sommes basés sur l'outil d'analyse de code statique Pylint ainsi que de notre propre analyseur structurelle du notebook. Le choix de Pylint se motive par le fait que cet outil est une référence dans la mesure de la qualité de base de code en Python.

L'analyse produit autant de fichiers JSON que de notebooks analysés. Ceux-ci contiennent d’une part un score qualité (produit par Pylint), les catégories et nombre d’erreurs identifiées par Pylint et d’autre part le nombre de cellules de code et de markdown ainsi que leur fréquence.

#### Visualisation

Grâce aux fichiers JSON générés, il est possible d'en obtenir des visualisations grâce à des scripts dédiés. Développés en Python, ces scripts lisent les fichiers JSON pour en créer un graphique à l'aide de la librairie Matplotlib.

## IV. Hypothèse et expériences

Tout d'abord, nous avons fait l'hypothèse forte que la popularité des notebooks nous servira de référence dans l’identification des bonnes pratiques. Nous avons également sélectionné les sources dites “références" en nous basant sur leur réputation (différente de leur popularité au sens défini dans cette analyse) et leur impact sur la communauté de la science des données.

De plus, nous faisons l'hypothèse que la qualité d’un Jupyter Notebook est mesurable à la manière d’un code source Python traditionnel. Nous appliquerons alors des outils d’analyses standards tels que l’analyse statique Pylint ou encore une analyse structurelle du Jupyter Notebook (enchaînement de cellules de markdown/code) qui nous donneront un score pouvant être assimilé aux critères de qualité de code.

### Hypothèse 1 : la qualité du code dans les notebooks est bonne.

#### Expérience

Il s'agit d'identifier les notebooks exploitables pour le reste des expériences. Nous identifions ici les notebooks ayant un score Pylint acceptable, c'est-à-dire un score strictement supérieur à 0. Nous appliquons cette sélection afin de ne pas impacter les résultats suivants.

Enfin, parmi ces notebooks exploitables, il s'agit d'identifier la répartition des scores Pylint pour en connaître la tendance. Cette répartition s'intéressera en particulier aux notebooks populaires, c'est-à-dire les dépôts ayant le plus d'engagements sur GitHub et Kaggle.

#### Résultats

![Figure 2 - Résultat de l'analyse pour l'hypothèse 1 (nb pylint score == 0)](assets/images/hypothesis_1_result_pylint_score_0.png "Figure 2 - Résultat de l'analyse pour l'hypothèse 1 (nb pylint score == 0)")

Parmi les 626 notebooks extraits de GitHub, seuls 223 sont considérés comme exploitables. Cela présente un ratio d'exploitation d'environ 35,62%.

Et, parmi les 299 notebooks provenant de Kaggle, 246 notebooks ont un score Pylint satisfaisants. Le ratio d'exploitation est ici supérieur, avec environ 82,27% des notebooks exploitables.

Bien que Kaggle ait un ratio d'exploitation plus élevé, lorsque l’on s'intéresse dans le détail aux scores des notebooks exploitables, la répartition des scores est relativement partagée.

![Figure 3 - Résultat de l'analyse pour l'hypothèse 1 (pylint scores)](assets/images/hypothesis_1_result_pylint_scores.png "Figure 3 - Résultat de l'analyse pour l'hypothèse 1 (pylint scores)")

Cependant, on retrouve une majorité de notebooks avec un score Pylint compris entre 4 et 7. Entre 6 et 10, le nombre de notebooks ayant un score élevé décroît, ce qui signifie que l'obtention d'une bonne qualité Pylint est difficile.

### Hypothèse 2 : on peut distinguer un motif commun entre les notebooks.

#### Expérience

La structure d’un Jupyter Notebook est un élément important (équitabilité des cellules de markdown/code). Nous avons fait l’hypothèse qu’une structure commune peut être dégagée de l’ensemble des notebooks. Pour cela, nous avons sourcé des notebooks de popularité différente pouvant représenter un échantillon représentatif des notebooks déposées sur les différentes plateformes en ligne (GitHub et Kaggle). Pour notre expérience nous avons développé un analyseur syntaxique permettant d’obtenir la répartition entre code et markdown.

#### Résultats

![Figure 4 - Résultat de l'analyse pour l'hypothèse 2 (nombre cellules markdown/code entre github et kaggle)](assets/images/hypothesis_2_result_nb_markdowns_cells_github_kaggle.png "Figure 4 - Résultat de l'analyse pour l'hypothèse 2 (nombre cellules markdown/code entre github et kaggle)")

Le diagramme ci-dessus nous renseigne sur les tendances entre GitHub et Kaggle, permettant de situer l’équilibre entre cellules de code et de markdown.

Parmi l’ensemble des notebooks représentés, nous pouvons constater qu’un certain équilibre est trouvé entre cellule de code et de markdown. En effet, le consensus voudrait qu’avant chaque bloc de code, une explication préface de ce qu’il va suivre pour expliquer au mieux le déroulé du programme. C’est bien ce que nous pouvons en partie constater, en majorité il y a autant de blocs de code que de blocs de markdown dans les notebooks. De ce diagramme il est compliqué d’affirmer qu’il y ait bien cette alternance entre code et markdown, mais cet équilibre pourrait indiquer ce motif structurel.

### Hypothèse 3 : les notebooks populaires sont de qualités.

#### Expérience

L'objectif de cette expérience est d'être en mesure de savoir s'il existe une différence notable de qualité entre un notebook populaire et un notebook non populaire.

Il s'agira ici d'observer dans un premier temps la structure de chaque notebook puis la qualité évaluée par Pylint.

#### Résultats

![Figure 5 - Résultat de l'analyse pour l'hypothèse 3 (nombre cellules markdown/code entre populaires et moins populaires)](assets/images/hypothesis_3_result_nb_markdowns_cells_popular_lower.png "Figure 5 - Résultat de l'analyse pour l'hypothèse 3 (nombre cellules markdown/code entre populaires et moins populaires)")

Concernant l’équilibre code/markdown, nous constatons que la popularité des notebooks n'influence pas la qualité structurelle d'un notebook. Qu'un notebook soit populaire ou non, la tendance est équivalente.

![Figure 6 - Résultat de l'analyse pour l'hypothèse 3 (score pylint entre populaires et moins populaires)](assets/images/hypothesis_3_result_pylint_scores_popular_lower.png "Figure 6 - Résultat de l'analyse pour l'hypothèse 3 (score pylint entre populaires et moins populaires)")

Toutefois, la qualité relevée avec l'outil Pylint démontre que la qualité du code Python chez les notebooks populaires est globalement plus élevée que les notebooks non populaires. En effet, entre 0 et 2 de score, la majorité des notebooks est non populaire. Et, entre 5 et 10 de score, la majorité des notebooks est populaire.

Cette expérience permet de constater que malgré l'absence de différence structurelle d'un notebook populaire à un notebook non populaire, la qualité de code est quant à elle significative.

## V. Résultat d'analyse et conclusion

### Interprétation

Les notebooks ont pour la plupart un score moyen avec l'outil Pylint. Il existe cependant des différences notables de qualité Pylint lorsqu'on s'intéresse à la popularité des notebooks. En effet, un notebook populaire aura de plus grandes chances d'être de meilleure qualité par rapport à un notebook non populaire.

Concernant la qualité structurelle (proportionnalité de cellules de markdown/code), quelque soit leur popularité, les notebooks tendent à adopter 1,3 à 1,6 cellule de code pour 1 cellule de markdown. Ainsi, la communauté semble avoir adopté la pratique d'annoter des commentaires afin d'éclaircir le lecteur dans la compréhension de l'exécution du notebook.

### Prise de recul

Classement des 10 erreurs les plus répandues sur l'échantillon populaire

Erreurs les plus répandues | Occurrences
---|---
Trailing whitespace | 5 774
Invalid name | 5 193
Line too long | 3 773
Redefined outer name | 3 415
Wrong import position | 3 350
Bad indentation | 1 938
Missing function docstring | 1 833
Pointless statement | 1 581
Consider using "f" string | 1 374
Unused import | 973
Autres | 9 386

Classement des 10 erreurs les plus répandues sur l'échantillon aléatoire

Erreurs les plus répandues | Occurrences
---|---
Trailing whitespace | 3 447
Invalid name | 2 562
Pointless statement | 1 588
Missing function docstring | 1 577
Redefined outer name | 1 113
Unnecessary semicolon | 851
Missing class docstring | 653
Wrong import position | 614
Function redefined | 550
Consider using "f" string | 455
Autres | 4 076

Malgré un nombre de signalements de Pylint plus importants sur les notebooks populaires, le score reste plus élevé que celui des notebooks non populaires. Si on s'intéresse au calcul du score de Pylint, la formule appliquée est la suivante :

```python
max(0, 0 if fatal else 10.0 - ((float(5 * error + warning + refactor + convention) / statement) * 10))
```

Ainsi, plus il y a de "statements" (instructions en français), plus l'impact d'un signalement est faible sur le score Pylint.

### Limites

Pour cette étude, nous avons identifié certaines limites :

* Les notebooks non populaires peuvent éventuellement être un fork d'un notebook populaire. Ainsi, une partie des pratiques d'un notebook populaire peut être utilisée dans le fork, ce qui peut amener à biaiser les résultats. Il s'agirait, à l'avenir, de détecter si un dépôt extrait provient d'un fork afin de l'exclure de la sélection des données. Ce phénomène est toutefois assez rare.
* Une fuite mémoire présente dans l’outil Pylint lorsqu’il est utilisé programmatiquement limite le nombre d’analyses simultanées impactant ainsi le temps d’analyse.
* Certains notebooks (<1%) étant écrits en R, nous avons fait le choix de ne pas les supporter.
* Bien qu'il soit possible qu'un auteur de notebook privilégie les commentaires dans les cellules de code, nous avons fait le choix, pour la qualité structurelle d'un notebook, de ne pas identifier les commentaires dans les cellules de code Python.

## VI. Outils

Stack technique:

* Python 3.10 et [pipenv](https://pipenv.pypa.io/en/latest/index.html)

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

![Figure 1: Logo UCA, exemple, vous pouvez l'enlever](assets/images/logo_uca.png){:height="25px"}
