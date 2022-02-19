
---

layout: default

title : Quelles sont les bonnes pratiques de mise en place d’un système de packaging ?

date: 2022-01-08 20:00:00 +0100

---

  

**_janvier 2022_**

  
  

## Auteurs

  

Nous sommes cinq étudiants en dernière année à Polytech Nice Sophia Antipolis, dans la spécialité Architecture Logicielle :

  

* Bruel Martin &lt;martin.bruel@etu.univ-cotedazur.fr&gt;

* Esteve Thibaut &lt;thibaut.esteve@etu.univ-cotedazur.fr&gt;

* Lebrisse David &lt;david.lebrisse@etu.univ-cotedazur.fr&gt;

* Meulle Nathan &lt;nathan.meulle@etu.univ-cotedazur.fr&gt;

* Ushaka Kevin &lt;kevin.ushaka-kubwawe@etu.univ-cotedazur.fr&gt;

  
  

## I. Contexte de recherche / Projet

  

Le sujet de notre recherche porte sur les systèmes de packaging en Javascript.

  

Un système de packaging permet de faciliter le développement et la gestion de sites/d'applications web.

Il récupère les modules avec des dépendances et génère des ressources statiques représentant ces modules.

  

Les systèmes de packaging permettent également d'optimiser la taille de code en minifiant le code, réutilisant certains blocs.

  

Ce contexte de recherche est particulièrement intéressant pour nous puisque nous avons tous été amenés à travailler sur des projets incluant du Javascript. De plus, ce langage reste aujourd'hui l'un des plus utilisés.

  

Il existe une multitude de librairies développées en Javascript : de nombreux projets utilisent des dépendances vers ces librairies Javascript. Il faut donc gérer et organiser ces librairies, ce qui devient de plus en plus compliqué. Cela explique l'intérêt croissant des systèmes de packaging.

  
  

## II. Observations/Question générale

  

Nous avons remarqué, après une rapide recherche sur Github, que la plupart des grands projets Javascript utilisent des systèmes de packaging (afin de gérer les dépendances, répondre à des besoins d'optimisations, etc.). Ces gestionnaires de packages permettent, une fois bien configurés, de réduire les temps de développement et de faciliter le passage en production de l'application réduisant ainsi les coûts pour les entreprises.

  

Nous nous sommes ainsi demandés **Quelles sont les bonnes pratiques de mise en place d’un système de packaging**.

  

Tout d'abord, nous trouverons quels sont les types de configurations possibles et les plus utilisés dans les projets en les identifiant manuellement puis en récupérant les taux d'occurrence parmi différents projets. Les projets utilisent par exemple des linters ou minifieurs.

  

Dans une seconde étape, après avoir trouvé les pratiques les plus utilisées, nous nous demanderons si ces pratiques sont effectivement de bonnes pratiques. Est-ce que ce sont des pratiques poussées par la communauté de développeurs ou par les systèmes de packaging ? Est-ce qu’il y a des contraintes à utiliser les pratiques de gulp ou celles présentées par les développeurs (dans les forums, blogs...) ?

  

Enfin, si nous trouvons des pratiques différentes, quels sont les paramètres d'un projet qui influent sur ces différentes étapes ? Existe-t-il des paramètres qui poussent les développeurs à adopter une pratique plutôt qu’une autre? On se penchera particulièrement sur la taille d'un projet à l'égard de l'utilisation d'un minifieur. On étudiera également l'ancienneté d'un projet vis à vis de l'utilisation d'un concaténateur et transcompilateur.

  

Pour ce faire, nous avons prévu d'adopter la démarche suivante :

- Monter en compétence sur les différents systèmes de packaging (comment ils sont utilisés, pourquoi c’est utilisé)

- Trouver les bonnes pratiques (documentation, littérature, forum,...)

- Chercher des projets de différentes tailles utilisant Gulp

- Trouver les pratiques mises en place dans ces projets

- Evaluer les pratiques dans les projets

  

## III. Collecte d'informations

  

Les données de nos recherches sont collectées sur des projets Open Source hébergés sur Github, et des générateurs de fichiers de configuration ainsi que les forums/blogs.

  

Nous avons utilisé l'API de Github à l’aide d’un script python afin d'obtenir un échantillon conséquent. Nous avons ainsi recherché tout projet contenant un fichier intitulé gulpfile : `g.search_code('filename:gulpfile')`.

  

Ensuite, nous avons filtré ces échantillons après avoir remarqué que des fichiers de configuration (gulpfile.js) étaient présents dans des node_modules et faussaient nos premiers résultats. Par ailleurs, l’API de Github limite le nombre de résultats à 1000 réponses par requêtes, nous avons donc dû exécuter plusieurs fois notre script et agréger les données obtenues à notre *dataset* en supprimant les doublons. Nous avons finalement obtenu un *dataset* de 1016 projets comportant un gulpfile.js.

  

## IV. Hypothèses & Expériences

### Intuitions initiales

  

La documentation Gulp met en avant certains plugins : gulp-uglify (minifieur), gulp-concat (concatenation). Nous nous attendons donc à retrouver ces plugins et plus généralement nous pensons voir des minifieurs, de plugins de concaténation, des linters, et enfin des transcompilateurs javascript (le système de packaging rend les nouveaux projets utilisant ES6 compatible avec les anciens navigateurs).

  

Concernant les bonnes pratiques, nous pensons pouvoir les définir en nous appuyant sur l'occurrence et la pertinence dans les projets, les recommandations dans les forums, et la présence dans les générateurs de fichiers de configuration pour systèmes de packaging.

  

Nous supposons enfin que la taille d'un projet (en terme de taille du projet, nombre de contributeurs, nombre de commits), et que la date influent sur ces étapes. Ainsi nous pensons voir des minifieurs dans de grands projets (où il y a un besoin d'optimiser la taille du code) et des concaténateur/transcompilateurs avant une certaine date car ces derniers permettent de supporter les anciens navigateurs.

  
  

### Expérimentation

  

#### Les pratiques récurrentes

  

Pour vérifier nos hypothèses, nous avons mis en place un premier script Python analysant les fichiers de configurations de Gulp. Nous avons fait le choix de commencer par Gulp puisque les fichiers de configuration (gulpfile.js) sont facilement détectables au sein de projets et exploitables rapidement. En effet, Gulp n’est qu’un exécuteur de tâches, ainsi pour réaliser une tâche de concaténation il doit importer le plugin gulp associé (gulp-concat) : il est ainsi facile de déterminer les actions effectuées en regardant les imports. Notre script Python permet donc de comptabiliser et catégoriser les imports de plugins et les dates/tailles du projet.

  

Outils utilisés :

- API Github pour la recherche de gulpfile.js

- Python pour lancer les requêtes, lire les fichiers de configuration et regrouper les données dans des data set

  
  

#### Définition des critères de recherche (taille/date)

Pour établir une corrélation entre les bonnes pratiques identifiées et la taille ou la date d’un projet nous avons, après analyse de notre *dataset*, défini ce qu’est un gros/petit projet et un ancien/récent projet.

  

Pour trouver une limite entre ce que l’on pourrait considérer comme un petit projet et un grand projet, nous avons tout d’abord effectué une analyse statistique de notre dataset. Pour avoir une idée de la répartition de notre *dataset*, nous avons réalisé une boîte à moustache (*boxplot*) de celui-ci. Les *boxplots* permettent en effet d’avoir une idée de la répartition des données de notre *dataset*.

  

![](https://lh4.googleusercontent.com/yFrY1FyFwsQkT88yOuTKX-wVGc3DucPPIm-91IdM89EfDLPvaE5PV0py96_kcKOrVUCGNCxas3NKDG45SiwMVZbt5MMsSbDOpw7qk5KSbszTZ5mNh1dIaIMBgZzTaSCSAdSs066-)

  

Comme on peut le constater, le *boxplot* de notre *dataset* est totalement déformé. Les nombreux cercles que l’on observe correspondent à des valeurs dites “aberrantes”, c'est-à-dire que la taille de ces projet est très importante par rapport au reste du *dataset* et que ceux-ci peuvent perturber les statistiques. Pour s’en convaincre il suffit de comparer la moyenne et la médiane de nos données, en effet nous obtenons une moyenne de taille de projet de 39 734 ko contre une médiane de 7 389 ko. La moyenne est donc fortement perturbé par les valeurs aberrantes. Pour avoir un *dataset* un peu plus “propre”, il est possible de retirer des valeurs aberrantes. Pour ce faire, une méthode consiste à définir un seuil limite correspondant à l’écart entre le troisième quartile et le premier quartile, puis de multiplier cette écart par 1,5 et enfin de supprimer toute les données qui sont supérieures au troisième quartile additionné au seuil de l'écart interquartile fois 1,5. Dans notre cas, le premier quartile est de 1518 et le troisième quartile de 29612, ainsi on retire tous les projet d’une taille supérieure à 71753 ko (29612 + 1,5 x (29612 - 1518 ) = 71753). On obtient ainsi le *boxplot* suivant :

![](https://lh6.googleusercontent.com/HxLklldbR-3VFWJQqivKL1b4YwTfrZJFo9Rpi9uwqHspWC9LqkpKbBk2-1WwC5z5rUBDlp3DPS2z6gZKXfWrK80wCuSkhCqaIYvFXrAyJ0m0PVDQ0fIvC2QhnE9LYpSpsYtfeulr)

  

Même s’il y a une amélioration, celui-ci dispose toujours de beaucoup de valeurs aberrantes. Ainsi, on répète le calcul précédent pour retirer les projets avec une taille trop importante.

Or à force de vouloir éliminer les projets qui posent problème nous nous retrouvons à supprimer 341 projets soit plus d’un tiers de notre *dataset*.

  

Cette approche ne s'avérant pas viable, nous avons alors opté pour une autre approche plus “manuelle”, en choisissant pour point de départ le séparateur obtenu précédemment. Nous avons petit à petit ajusté ce dernier. Pour ce faire, nous avons sélectionné arbitrairement des projets parmi les éléments détectés grand (ou petit) du *dataset* et déterminé manuellement s’ils le sont effectivement. Si ce n’était pas le cas, nous ajustions le séparateur pour inclure le projet en question dans la bonne catégorie.

  

Ainsi nous avons défini un gros projet comme un projet comportant une taille supérieure à 20 000, un nombre de commit supérieur à 40, ainsi qu’un nombre de contributeurs supérieur à 2. Les projets ayant plus de 100 commits ou plus de 7 contributeurs sont directement considérés comme importants et donc inclus dans le *dataset* des gros projets. Nous aurions souhaité inclure le nombre de lignes de code d’un projet mais cet attribut n’était pas proposé par l’API Github (compatibiliser nous même le nombre de lignes aurait augmenté de manière significative le temps d'exécution)

  

Concernant la date, nous avons fixé la date limite séparant nos anciens et jeunes projets à 2016. En effet, ES6 a été publié fin 2015, nous nous attendons donc à retrouver des transcompilateurs sur les projets modifiés après 2015.

  
  
  
  
  

## V. Result Analysis and Conclusion

  
  

### Résultats sur les pratiques identifiées

  

Le premier script d’analyse des pratiques nous a permis d’obtenir le diagramme suivant :

![](https://lh3.googleusercontent.com/GpQxakITKlamY7jv9E5rj-ImxX6NbRdj_RSHuvfNjnBuruKxEEt8KcOFTM7gXH-iJn6Cw1RK0M9zSU4KvTyn_BsKkVAifg4hYeDnyijJlyHkEHT_b6qK0zf8F72gIL4PcbV7BMEU)

  

En étudiant l'occurrence des plugins gulp, nous remarquons que la plupart des projets incluent un plugin de minification (uglify/imagemin), et de concaténation (concat) qui constituent les fonctionnalités principales des systèmes de packaging. Les plugins de traduction de fichiers SCSS/SASS vers css (sass) et d'auto préfixage (autoprefixer) sont également beaucoup utilisés car ils permettent de faciliter le développement.

  

De cette première expérimentation nous en avons ressorti que dans les configurations des systèmes de packaging, les plus utilisés étaient la minification et la concaténation. Quant au plugin de nommage (rename) ainsi qu’aux plugins de traduction et d’auto préfixage des fichiers de styles, il ne nous a pas semblé intéressant de les prendre en compte dans la suite de l’étude puisqu’ils n’agissent pas sur le javascript.

  

### S’agit-il de bonnes pratiques ?

  

La documentation de Gulp présente des plugins pour réaliser les tâches les plus communes. On peut notament retrouver, parmi l’analyse de l’utilisation présentée juste au-dessus, les plugins suivants : **gulp-uglify** et **gulp-rename**<sup>5</sup>. De plus, Google un des plus gros acteur du web présente un tutoriel pour configurer un Gulpfile afin d’automatiser des tâches de compilation avant le déploiement<sup>1</sup>. Dans ce tutoriel, on peut retrouver les plugins : gulp-babel (transcompilateur Javascript ES6 vers ES5), gulp-uglify et gulp-rename.

Cependant, est-ce que les pratiques présentées par ces différentes documentations sont-elles remises en question ou validées par la communauté ?

Nous avons mené des recherches sur des forums et des articles scientifiques pour comprendre ce qu’apporte l’utilisation des différentes tâches gulp utilisées.

-   La **minification** améliore les performances de chargement des pages web<sup>8</sup>. De plus, une minification manuelle est dénoncée. En effet, les plugins de minification proposés par les systèmes de packaging utilisent des algorithmes plus efficaces<sup>7</sup>
    
-   La **concaténation** améliore également les performances de chargement. En effet, l'envoi de plusieurs fichiers via le protocole HTTP/1.1 est plus long et la résolution de ceux-ci par le navigateur prend du temps. Elle permet également une compatibilité pour les navigateurs ne supportant pas les modules. Par contre, dans un contexte plus récent avec l’utilisation du protocole HTTP/2, la concaténation ne montre pas clairement de meilleure performance au chargement<sup>3</sup>. 
    
-   La **transcompilation** est utilisée pour assurer l’utilisation des applications sur les anciens navigateurs. Les différents forums consultés considèrent que les clients potentiels qui utilisent ces navigateurs représentent une part négligeable par rapport au coût de mise en place de cet outil<sup>4</sup>.
    Nous avons voulu vérifier si cette approche n’était effectivement pas utilisée en pratique.
  

Finalement, en regroupant les pratiques présentées par les documentations et les analyses de performance trouvées, on peut en conclure que l’utilisation de minification, de concaténation et de transcompilation sont de bonnes pratiques selon le contexte dans lequel on les utilise.

  
  

### Facteurs influençant les bonnes pratiques

  

En recoupant les bonnes pratiques avec les données analysées dans les projets utilisant gulp.Nous avons obtenu les résultats suivants :

![](https://lh4.googleusercontent.com/cspN91tOR-66zc8GvyQm0IHOoGascfiJ8OtIOH47R9HDKaR0eN2ZUDYwfOaPYcNCS_oXtmQoMxDEyraxXv2ON6C0ZZ4ksI3uQPY3KUBGw5TvdrHglQQZ0E0e4fqOpndgefnzB-nQ)

Nous constatons clairement un lien entre la taille du projet et la taille d’un minifieur.

  
  
  
  

![](https://lh3.googleusercontent.com/MyrqLJmGEiTSKZpPn4F6UV0aHMbyyP900iAq2DIa7VWGNgafBY6o2QAxMFGcH4oVyuoYKCWZRkFkaeX4Q9wx0clUKXLBiFTq9oOhQySmVJUicIciBkauoVwzDjyemUVeUgVuMjtc)

  
  

![](https://lh6.googleusercontent.com/qF62hFDfHveZCoTABHR-kIYDvaGUdCzUxZKuJeEA3sAglkH3cqID1xFAI0TL3CdIAyaWw5o6B0PEnxvmbGuX2pj8m5CMpeTCgDLeO6HGjOjWGTSYyI9q6LCFco3_ZjrXldM04omw)

Nous remarquons ici une corrélation entre la récence d’un projet et la présence d’un transcompilateur : les projets mis à jour après 2016 possèdent des tâches gulp de transcompilation.

Cependant le pourcentage obtenu reste très faible : il s’avère, qu’en pratique, mettre en place un plugin de transcompilation ne fait perdre qu’une infime part du marché (les vieux navigateurs ne supportant pas ES6).

  
  
  
  
  
  
  

## VI. Tools

- [Pygithub](https://pygithub.readthedocs.io/en/latest/github_objects/Repository.html)

  
  

## VI. References

  
  

[1]

« Lab: Gulp Setup | Web », Google Developers. [https://developers.google.com/web/ilt/pwa/lab-gulp-setup](https://developers.google.com/web/ilt/pwa/lab-gulp-setup) (consulté le 19 février 2022).

[2]

« The JavaScript Packaging Problem », Zero Wind :: Jamie Wong. [http://jamie-wong.com/2014/11/29/the-js-packaging-problem/](http://jamie-wong.com/2014/11/29/the-js-packaging-problem/) (consulté le 19 février 2022).

[3]

M. Jung, « The Right Way to Bundle Your Assets for Faster Sites over HTTP/2 », Medium, 24 avril 2019. [https://medium.com/@asyncmax/the-right-way-to-bundle-your-assets-for-faster-sites-over-http-2-437c37efe3ff](https://medium.com/@asyncmax/the-right-way-to-bundle-your-assets-for-faster-sites-over-http-2-437c37efe3ff) (consulté le 19 février 2022).

[4]

L. Wang (Ex-Uber), « Transpilers: Do you really need it ? », Medium, 3 septembre 2018. [https://medium.com/@pramonowang/transpilers-do-you-really-need-it-e9c63686e5fe](https://medium.com/@pramonowang/transpilers-do-you-really-need-it-e9c63686e5fe) (consulté le 19 février 2022).

[5]

« Using Plugins | gulp.js ». [https://gulpjs.com//docs/en/getting-started/using-plugins](https://gulpjs.com//docs/en/getting-started/using-plugins) (consulté le 19 février 2022).

[6]

« webpack », webpack. [https://webpack.js.org/](https://webpack.js.org/) (consulté le 19 février 2022).

[7]

« What is Minification | Why minify JS, HTML, CSS files | CDN Guide | Imperva », Learning Center. [https://www.imperva.com/learn/performance/minification/](https://www.imperva.com/learn/performance/minification/) (consulté le 19 février 2022).

[8]

« Why minify JavaScript code? », Cloudflare. [https://www.cloudflare.com/learning/performance/why-minify-javascript-code/](https://www.cloudflare.com/learning/performance/why-minify-javascript-code/) (consulté le 19 février 2022).

[9]

« You might not need to transpile your JavaScript », freeCodeCamp.org, 19 mai 2017. [https://www.freecodecamp.org/news/you-might-not-need-to-transpile-your-javascript-4d5e0a438ca/](https://www.freecodecamp.org/news/you-might-not-need-to-transpile-your-javascript-4d5e0a438ca/) (consulté le 19 février 2022).
