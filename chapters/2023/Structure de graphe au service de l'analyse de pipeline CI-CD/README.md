**_Février 2023_**

## Auteurs
Nous sommes quatre étudiants ingénieurs en dernière année à Polytech Nice Sophia, spécialisés en Architecture Logiciel :

* Gabriel Cogne &lt;gabriel.cogne@etu.unice.fr&gt;
* Marine Demonchaux &lt;marine.demonchaux@etu.unice.fr&gt;
* William Fernandes &lt;william.fernandes@etu.unice.fr&gt;
* Taha Kherraf &lt;taha.kherraf@etu.unice.fr&gt;
    
## I. Contexte de la recherche / projet
De plus en plus, les projets sont accompagnés d'un pipeline CI/CD afin de les construire, tester et déployer automatiquement.
Ce pipeline est composé de plusieurs jobs[^job] pouvant produire ou nécessiter des artefacts pour s'exécuter. Ainsi, les
différents jobs d'un pipeline peuvent être dépendant les uns des autres sans que ces dépendances soient explicitées. Cela
peut provoquer des erreurs lors de l'exécution d'un pipeline, par exemple si un artefact est modifié entre l'exécution de
deux jobs. On peut donc se poser la question suivante :

> Est-il possible par l'analyse des "actions" d'identifier des dépendances entre celles-ci sous la forme d'artefacts (par
exemple : la génération d'un fichier Icov pour une publication de coverage) ? Peut-on identifier des tests de présence
dans les pipelines ?


## II. Observations / Question générale
Plus précisément, nous allons nous demander : Est-il possible d'établir des graphes de dépendances orientés acycliques 
intra-job[^intra-job] et inter-job[^inter-job] à la suite de l'analyse de pipelines ayant une phase de construction, de 
tests et déploiement ? Peut-on identifier des patrons de gestion de dépendances notables entre les différents pipelines ?

Cette question est intéressante dans la mesure où elle permettrait de connaître les dépendances entre les étapes. On 
pourrait ainsi de savoir si un changement apporté à un artefact (modification ou suppression) va impacter l'exécution du
pipeline. Ou alors savoir qu'un pipeline va échouer parce qu'une dépendance requise à une étape est générée par une étape
ultérieure.

Afin de ne pas trop complexifier notre étude, nous allons nous limiter aux pipelines GitHub Action.

## III. Collecte d'informations
Préciser vos zones de recherches en fonction de votre projet, les informations dont vous disposez, ... :

1. les articles ou documents utiles à votre projet
2. les outils
3. les jeux de données/codes que vous allez utiliser, pourquoi ceux-ci, ...

Pour notre recherche, nous comptions nous baser sur trois articles :
* [Who broke the build?: automatically identifying changes that induce test failures in continuous integration at Google scale](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45794.pdf)
  * Auteurs : Celal Ziftci, Jim Reardon
  * Date de publication : 2017/5/20
  * Conférence Proceedings of the 39th International Conference on Software Engineering: Software Engineering in Practice Track 
  * Pages 113-122
  * Description
    * Quickly identifying and fixing code changes that introduce regressions is critical to keep the momentum on 
    software development, especially in very large scale software repositories with rapid development cycles, such as at
    Google. Identifying and fixing such regressions is one of the most expensive, tedious, and time consuming tasks in
    the software development life-cycle. Therefore, there is a high demand for automated techniques that can help 
    developers identify such changes while minimizing manual human intervention…
* [Mining Metrics to Predict Component Failures](http://linyun.info/micode/micode.pdf)
  * Auteurs : Yun Lin, Guozhu Meng, Yinxing Xue, Zhenchang Xing, Jun Sun, Xin Peng, Yang Liu, Wenyun Zhao, Jinsong Dong 
  * Date de publication : 2017/10
  * Conférence The 32nd IEEE/ACM International Conference on Automated Software Engineering
  * Pages 394–404
  * Description
    * In this paper, we propose an approach to detecting project-specific recurring designs in code base and abstracting
    them into design templates as reuse opportunities. The mined templates allow programmers to make further 
    customization for generating new code. The generated code involves the code skeleton of recurring design as well as 
    the semi-implemented code bodies annotated with comments to remind programmers of necessary modification. We 
    implemented our approach as an Eclipse plugin called…
* [When Life Gives You Oranges: Detecting and Diagnosing Intermittent Job Failures at Mozilla](https://www.se.cs.uni-saarland.de/publications/docs/LJA+21.pdf)
  * J. Lampel, S. Just, S. Apel, and A. Zeller,
  * in ESEC/FSE 2021 - Proceedings of the 29th ACM Joint Meeting European Software Engineering Conference and Symposium on the Foundations of Software Engineering, 2021,  
  * vol. 21, pp. 1381–1392, doi: 10.1145/3468264.3473931.
  * Continuous delivery of cloud systems requires constant running of jobs (build processes, tests, etc.). One issue 
  that plagues this continuous integration (CI) process are intermittent failures-non-deterministic, false alarms that 
  do not result from a bug in the software or job specification, but rather from issues in the underlying infrastructure.
  At Mozilla, such intermittent failures are called oranges as a reference to the color of the build status indicator. 
  As such intermittent failures disrupt CI and lead to failures, they erode the developers' trust in the jobs. We present
  a novel approach that automatically classifies failing jobs to determine whether job execution failures arise from an 
  actual software bug or were caused by flakiness in the job (e.g., test) or the underlying infrastructure. For this
  purpose, we train classification models using job telemetry data to diagnose failure patterns involving features such 
  as runtime, cpu load, operating system version, or specific platform with high precision. In an evaluation on a set of 
  Mozilla CI jobs, our approach achieves precision scores of 73%, on average, across all data sets with some test suites 
  achieving precision scores good enough for fully automated classification (i.e., precision scores of up to 100%), and 
  recall scores of 82% on average (up to 94%).

Pour notre étude, nous allons analyser les projets GitHub suivant :
* [Audacity](https://github.com/audacity/audacity)
* [Django API](https://github.com/awaisanjumx2/django-api)
* [TS down sample](https://github.com/predict-idlab/tsdownsample)
* [Cloudy Weather API](https://github.com/iiTONELOC/cloudyWeatherAPI)
* [Juice Shop](https://github.com/miguelmemm16/juiceshop)
* [Tail Wind CSS](https://github.com/tailwindlabs/tailwindcss)
* [DGS Framework](https://github.com/Netflix/dgs-framework)

Pour chacun de ces projets, nous allons générer des graphes de dépendances en utilisant un script Python et une visualisation
en graphe faite avec [Graphviz](https://graphviz.org/).

     :bulb: Cette étape est fortement liée à la suivante. Vous ne pouvez émettre d'hypothèses à vérifier que si vous avez les informations, inversement, vous cherchez à recueillir des informations en fonction de vos hypothèses. 
 
## IV. Hypothèse et expériences
Avant la réalisation de notre recherche, nous allons expliciter nos suppositions quant au résultat que nous allons 
obtenir.

Il sera sans doute plus facile de détecter des dépendances intra-job[^intra-job]. En effet, nous pensons que trouver,
parmi les étapes d'un job, des besoins pour la suite d'un job est plus simple. Exemple : une étape consistant à construire
un projet, qui crée un binaire dans un dossier "build", et l'étape suivante utilisant ce dossier. Si le nom du dossier
se trouve dans un fichier de config, et, qu'entre deux soumissions, il est changé, il y aura un problème. Dans ce cas,
le fichier de config peut également être considéré comme une dépendance du job.

Concernant la topologie des graphes, on s'attend à obtenir deux types de graphe :
* Pour les graphes de dépendances intra-job[^intra-job], nous nous attendons à obtenir un graphe orienté acyclique et de 
  taille variable.
  * Orienté acyclique, car les étapes d'un job s'exécutent dans un ordre précis, étape par étape. En théorie, une étape A
  précédent une étape B ne peut pas dépendre d'un artefact produit par l'étape B. Si cela se produit, alors l'exécution 
  du pipeline risque fortement d'échouer.
  * La taille du graphe devrait dépendre du nombre d'étapes au sein d'un job.
* Pour les graphes de dépendance inter-job[^inter-job], nous nous attendons à obtenir un graphe orienté acyclique et de
  taille petit.
  * Orienté acyclique, car un pipeline a un début et une fin, que les jobs soient exécutés séquentiellement ou 
  parallèlement. Si le graphe avait un cycle, alors un job A pourrait dépendre du résultat d'un job B qui n'aurait pas été
  exécuté or c'est impossible.


1. Il s'agit ici d'**énoncer sous forme d'hypothèses** ce que vous allez chercher à démontrer. Vous devez définir vos hypothèses de façon à pouvoir les _mesurer/vérifier facilement._ Bien sûr, votre hypothèse devrait être construite de manière à _vous aider à répondre à votre question initiale_. Explicitez ces différents points.
2. Vous **explicitez les expérimentations que vous allez mener** pour vérifier si vos hypothèses sont vraies ou fausses. Il y a forcément des choix, des limites, explicitez-les.

     :bulb: Structurez cette partie à votre convenance : Hypothèse 1 => Expériences, Hypothèse 2 => Expériences ou l'ensemble des hypothèses et les expériences....


## V. Analyse des résultats et conclusion

1. Présentation des résultats
2. Interprétation/Analyse des résultats en fonction de vos hypothèses
3. Construction d’une conclusion 

     :bulb:  Vos résultats et donc votre analyse sont nécessairement limités. Préciser bien ces limites : par exemple, jeux de données insuffisants, analyse réduite à quelques critères, dépendance aux projets analysés, ...

## VI. Outils \(facultatif\)

Précisez votre utilisation des outils ou les développements \(e.g. scripts\) réalisés pour atteindre vos objectifs. Ce chapitre doit viser à \(1\) pouvoir reproduire vos expérimentations, \(2\) partager/expliquer à d'autres l'usage des outils.

![Figure 1: Logo UCA, exemple, vous pouvez l'enlever](./images/logo_uca.png)


## VII. Références
[^Debret 2020] Debret, J. (2020) La démarche scientifique : tout ce que vous devez savoir ! Available at: https://www.scribbr.fr/article-scientifique/demarche-scientifique/ (Accessed: 18 November 2022).

## VIII. Glossaire
[^job]: Job : Action décrite dans un pipeline, composée de plusieurs étapes.
[^intra-job]: Dépendance intra-job : Une dépendance entre deux étapes d'un même job
[^inter-job]: Dépendance inter-job : Une dépendance entre des étapes de deux jobs différents