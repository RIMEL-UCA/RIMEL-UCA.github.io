# Où et comment sont utilisées les Java properties ?

## Auteurs

We are four students in last year of Polytech' Nice-Sophia specialized in Software Architecture :

* Florian Bourniquel
* Doryan Bonifassi
* Zineb El Haouri
* David Sene

Lien github : [https://github.com/doryanB/RIMEL](https://github.com/doryanB/RIMEL)

## I. Context de recherche

La création de la classe Properties remonte à Java 1.0 bien avant que des concepts objet plus ou moins équivalent soient introduits comme les maps. Les properties jouent un rôle important dans un grand nombre de projets Java. Elles sont généralement utilisées pour sauvegarder les données de configuration des projets Java. Par exemple, elles peuvent indiquer l'URL de la base de données et les informations pour s'y connecter, ou encore quel logger utiliser et quel niveau de log on souhaite avoir lors de l'exécution.  


Le moyen le plus répandu pour les définir est de les inscrire dans des fichiers .properties qui seront par la suite lues par le programme, cependant elles peuvent aussi être créé dans le code à l'aide de la méthode setProperty\(\).

Un autre avantage des properties stockées dans des fichiers est que lors d'un changement dans un des fichiers il n'est pas nécessaire de recompiler le code java pour prendre en compte cette modification. Elles sont alors souvent utilisées pour modifier la configuration des projets au déploiement.

Même si cette "technologie" date de 1996 elle reste indispensable et il serait très difficile de la remplacer non seulement par sa longévité et l'habitude des développeurs à l'utiliser, mais surtout, car sa disparition entraînerait d'importants problèmes de compatibilité.

Cependant même si les properties sont omniprésentes dans les projets java on peut se demander où sont utilisées les properties, uniquement dans certaines parties des projets comme le code ou à l’inverse, sont-elles présentes dans d'autres parties également comme les tests et le déploiement. Un autre point que nous voulons aborder est l’utilisation que font les développeurs des properties. Les utilisent-ils par le biais des méthodes présentes de base dans le JDK \(Java.utils.Properties\) ou bien par des librairies externes ? Ce sont ces aspects-là que nous souhaitons approfondir.

## II. Observations et question générale

### A Question

Nous avons constaté que la plupart des projets Java utilisent à un moment donné des properties. Nous nous sommes donc demandé jusqu'où les développeurs poussent l’utilisation de cette technologie.

En effet, nous trouvons intéressant de comprendre et de pouvoir quantifier l’utilisation que font les développeurs de cette brique élémentaire de java.

Nous avons découpé ce thème en trois questions :

* Où les développeurs utilisent les properties ? Uniquement dans le code ou bien dans l’ensemble du projet ? Par exemple, un projet peut utiliser les properties uniquement dans son code, mais pas dans ses tests, ou à l'inverse les utiliser dans ses deux parties.
* Comment les properties sont utilisées ? Directement grâce aux méthodes présentes de base dans le JDK \(Java.utils.Properties\) de java ou sont-elles manipulées par le biais de librairies ou de wrappers créer par les développeurs?
* Est-ce que la majorité des projets utilisent les properties Java dans un même but ? Par exemple, pour indiquer l'URL de la base de données ou le type de logger à utiliser. Ou à l'inverse pour des besoins propres au métier de leur projet

### B KPI

Pour répondre à ces trois questions, nous avons défini les KPI suivants.

#### La localisation des appels aux méthodes interagissant avec les properties

Cette KPI consiste à lister la localisation \(chemin complet de la classe et son package\) de l’ensemble de l’appel aux méthodes permettant d’accéder ou de modifier les properties.  

Grâce à la localisation précise que nous fournit ce KPI, nous allons pouvoir définir dans quelle partie des projets sont utilisées les properties\(code / test / déploiement\). Que ce soit grâce aux méthodes présentes dans le jdk de java ou bien à l’aide d’une librairie / wrapper externe.

Nous avons défini que les properties sont utilisées dans une des parties d’un projet \(par exemple la partie test\) si le rapport entre le nombre d’appels aux méthodes trouvées dans la partie courante sur le nombre total d’utilisations des properties doit être supérieur à 0,05 \(5%\) . Nous avons défini cette limite a 5% pour éviter de nous retrouver dans le cas extrême ou un projet qui appellerait 40 000 fois les properties dans le code, mais qui appellerait seulement 4 fois les properties dans ses tests ne soit considéré comme un projet qui utilise les properties dans ses tests. Cependant, nous avons défini ce seuil assez bas, car nous estimons qu’en dessous de ce seuil, il n’y a pas assez d’appels pour considérer la partie comme utilisant réellement les properties. Une fois que nous aurons listé la position de ces appels grâce au nom des classes / package qui les contiennent, nous serons en mesure de répondre à cette première question.

#### Utilisation ou non d’un wrapper/library externe pour manipuler les properties

Cette KPI consiste à compter le nombre de properties créé dans les fichiers properties et le nombre d’appels à la méthode présente de base dans le JDK de Java qui permet d’accéder aux properties System.getProperty\(\).

En comparant le nombre de properties créé et le nombre d’appels a l’accesseur, nous allons pouvoir différencier les projets qui utilisent un wrapper / une librairie externe d’un projet qui utilise uniquement les méthodes présente dans le JDK.

Nous allons considérer qu’un projet utilise des librairies / wrapper à partir du moment ou le rapport nombre d’appels au getter trouvé sur le nombre de properties déclaré dans le fichier properties est inférieur à 0,10 \(10%\) et le nombre total d’appels à System.getProperty\(\) est inférieur à 2. Sinon, nous considérons que le projet utilise les méthodes du JDK. Nous avons choisi 10% comme seuil par une première observation rapide de notre dataset. Le seuil de 2 appels à la méthode System.getProperty\(\) vient tout simplement du constat que si un développeur utilise une librairie/ wrapper l’accesseur ne doit apparaître qu’une ou deux fois dans la librairie, mais pas plus, car dans le code nous utiliserons la librairie et non plus System.getProperty\(\).

#### Fréquence d’utilisation d’une property donnée

Cette KPI consiste à lister le nom des properties utilisé dans l’ensemble des projets. En vérifiant si le nom des properties est redondant dans plusieurs projets, nous allons pouvoir répondre à la question : est-ce que les projets utilisent des properties pour répondre souvent au même besoin ou bien est-ce que chaque projet les utilise pour répondre à des besoins qui leur sont propres ?

Nous avons conscience que cette méthode de validation reste limitée et naïve puisqu'il est possible que deux projets utilisent une property pour les mêmes raisons, mais nommée différemment. Cependant, cette métrique nous donnera une première réponse partielle.

## III. Récolte d'informations

### A. Dataset

Nous avons fait le choix de l'imiter le scope des projets que nous allons analyser afin de nous créer un point d'entrée et de rendre le projet réalisable en deux mois.

Nous nous sommes dans un premier temps restreints au projet GitHub, car c’est une des sources de données les plus rapidement utilisables et les plus populaires. De plus, nous avons préféré nous concentrer sur les cent premiers projets Java qui avaient le plus d’étoiles. Nous avons fait ce choix dans l’optique d’obtenir un dataset contenant seulement des “gros” projets Java qui ont du vécu et qui donc, pour nous, sont bien plus représentatif dans l’utilisation qu’ils vont faire des Properties que si nous avions sélectionné des projets de manière aléatoire par l’API GitHub. Au vu de l’ancienneté des Properties \(Java 1.0\), nous n’avons pas exclu les projets qui ne sont plus maintenus, car ils restent pertinents dans notre contexte. Enfin, nous avons réduit une seconde fois notre dataset en nous focalisant uniquement sur des projets Maven puisqu’ils offrent une structure définie et constante, ce qui nous permet de plus facilement parcourir et localiser les appels aux properties.

On peut cependant préciser qu’il serait simple d’augmenter le nombre de projets présents dans notre Dataset. Il suffirait de rajouter leur lien GitHub à notre dictionnaire de lien.

### B. Protocole

Notre protocole est découpé en trois parties distinctes.

#### Récupération et filtrage des répositories

* Dans un premier temps, on collecte l’ensemble des repository \(100 premiers projets java qui ont le plus d’étoiles sur GitHub\).

* Afin d’épurer le dataset, nous allons lancer plusieurs scripts bash qui ont pour but de nous indiquer les projets qui ne sont pas pertinents pour notre étude.
  * Un premier script bash nous indique quels projets ne sont pas des projets Maven.
  * Un second script qui permet de vérifier la présence de properties java pour éliminer les projets qui n’en contiennent pas.

#### Analyse et cartographie des projets

* Dans un premier temps, nous parcourons l’ensemble des fichiers properties afin de lister les properties et récupérer leurs noms. 
* Ensuite, grâce à Spoon, nous allons effectuer un premier passage pour répertorier la localisation des différents appels aux méthodes Java “System.setProperties” et "System.getProperties”. Cette étape est importante, car elle nous permet de différencier les deux types de projets que nous allons traiter. Ceux qui utilisent directement les méthodes de base du JDK \(“System.setProperties” et System.getProperties”\) des projets qui utilise une librairie ou un wrapper pour interagir avec les properties.   Cette différenciation est cruciale, car dans le premier cas, nous ne devons effectuer qu’un seul passage avec spoon pour localiser les appels aux méthodes du JDK. Alors que dans le second cas, nous ne devons pas cette fois-ci localiser les méthodes du JDK Java, mais identifier quelles méthodes du wrapper / librairie encapsulent ces méthodes. Puis, effectuer une seconde passe avec spoon pour localiser l’utilisation de ces méthodes encapsulantes. 
* Nous exportons ces résultats au format CSV pour qu’ils puissent être utilisés par notre générateur de graphique.

#### Génération des graphiques

Nous passons nos CSV précédemment générés à un script R qui va construire les graphiques pour chaque projet analysé, mais également, des graphiques pour l’ensemble des projets.

### C. **Limites de notre dataset**

Nous avions dans la partie filtrage des repositories un script bash qui nous permettait de vérifier la présence de fichiers liés au déploiement \(DockerFile, JenkinsFile, docker-compose.yml…\) afin de vérifier s’il était possible de cartographier l’utilisation des properties dans le déploiement. Nous avons constaté que sur les cent projets traités, seulement dix comportaient ce type de fichier. Ces projets n’avaient pas de cohérence au niveau de la rédaction de ces fichiers ce qui rendait l’analyse automatique impossible. De plus, après une analyse manuelle nous n’avons trouvé aucune mention des properties Java. Pour toutes ces raisons, nous avons décidé de ne plus prendre en compte la partie déploiement dans notre étude.

### D. Le bug de spoon

Durant notre exécution nous nous sommes rendu compte d’un important problème avec spoon. Normalement, spoon nous permet de tester des projets multi maven et c’est le cas. Cependant si dans deux sous projets maven nous retrouvons la même classe spoon plante, car il n'accepte pas de construire ces résultats avec deux fois la même classe même si vue qu’ils sont dans deux projets maven différents il devrait n’y avoir aucun problème. Malheureusement c’est le cas de nombreux projets de notre dataset. Pour donner un ordre d’idée sur 100 projets nous en avions 60 après un premier filtrage, mais si l’on considère le bug de spoon il nous reste 20 projets à analyser. Ces 20 projets permettent d’avoir cependant des résultats cohérents, mais sont moins représentatifs que si nous avions pu exécuter spoon sur l’ensemble de notre dataset. On peut cependant noter que notre dataset n’est pas la cause de ce problème, mais bien un bug avéré de spoon.

## IV. Hypotheses & Experiences

Intuitivement, d’après notre expérience et une analyse rapide des projets de notre dataset, nous avons formulé les hypothèses suivantes.

#### Les properties Java sont principalement utilisées grâce aux méthodes Java natives

Nous pense qu’au vu de la simplicité de l’utilisation des properties java \( appel à un simple getter ou setter\), nous pouvons supposer qu’il n’est pas nécessaire de mettre en place une encapsulation pour utiliser ces méthodes.

Pour vérifier cette hypothèse, nous allons nous appuyer sur le rapport entre le nombre de properties set et le nombre d’utilisations de “System.getProperty”

#### Seules quelques properties sont présentes dans les tests 

Nous avons l’intuition que seul un sous-ensemble des properties est utilisé dans les tests. Notamment les properties de configuration par exemple celles qui indiquent quel type base de données utiliser ou encore son adresse.

Pour valider cette hypothèse, nous allons utiliser notre cartographie des projets afin de vérifier qu’effectivement, le nombre d’appels aux properties dans les tests est supérieur à 5% du nombre d’appels total, mais est bien inférieur au nombre de properties utiliser dans le code.

#### Une partie des properties sont génériques, un sous-ensemble des properties java sont donc utilisé de la même manière dans tous les projets

Nous avons constaté en parcourant les ReadMe des différents projets qu’en plus des properties spécifiques au métier de chaque projet, une partie était identiques dans plusieurs projets. Notamment les properties qui permettent de configurer les éléments communs au projet tel que les bases de données, les loggers ou la localisation des fichiers en entrée et en sortie.

Pour répondre à cette question , nous allons lister le nom des properties utilisées dans chaque projet. Nous allons ensuite vérifier que les mêmes noms apparaissent dans plusieurs projets.

## V. Résultats et conclusion

### Résultats

#### Les properties Java sont principalement utilisées grâce aux méthodes de base du JDK

![](https://lh5.googleusercontent.com/JOvA5Sj9IMf9i8wA97NrvG0pl_UCrnQc8aq6kzZDYCb2frLmS5rpqRGZOeX0d0uysbagLP_0yae_1D2lKfiOBNphTvuHzVTQSoo7gaYimV7_RvHHfaEMGNt2z6RCKCV_V9J4_U1p)

Sur cette figure qui affiche les projets qui utilisent ou non un wrapper externe, on peut constater que l’ensemble des projets testés utilise directement les properties de java grâce aux méthodes présentes dans le JDK. Ce résultat semble logique, car les properties étant simple à manipuler il n’y a pas de raisons à introduire une surcouche.

#### Seules quelques properties sont présentes dans les tests

Pour répondre à cette question nous devons dans un premier temps si dans les projets nous utilisons bien les properties dans les codes comme dans les tests.

![](../.gitbook/assets/image%20%2817%29.png)

Sur cette figure on constate bien que sur 60 % des projets de notre dataset c’est le cas. Même si nous n’avons pas un nombre assez conséquent de projets pour l'affirmer on peut supposer que de manière générale les projets utilisent les properties dans les tests et le code. Certains projets utilisent les properties uniquement dans le code et enfin peu de projets utilisent les properties uniquement dans les tests. Sur notre dataset nous avons les proportions suivantes :

* Projets qui utilisent des properties uniquement dans les tests : 10%
* Projets qui utilisent des properties uniquement dans le code : 30%
* Projets qui utilisent des properties dans les tests et le code : 60%

Maintenant pour répondre à notre hypothèse initiale : seules quelques properties sont présentes dans les tests, on peut regarder ces données que nous avons extraites de notre csv où nous avons uniquement conservé les projets qui utilisent les properties dans le code et les tests et les colonnes qui dénombrent le nombre de properties dans le code et les tests.

![](../.gitbook/assets/image%20%288%29.png)

Ce que l’on peut constater c’est que contrairement à notre hypothèse aucune tendance ne se dégage des autres 42% de ces projets utilisent plus de properties dans le code que dans les tests et à l’inverse 58% en utilisent plus dans les tests. De plus les écarts peuvent être importants comme avec le projet hazelcast qui contient 39% de properties dans ses tests que dans son code. Ou encore swagger-core dont la différence entre le nombre de properties dans le code et les tests n’est que 33%. Alors que dans joda-time nous n’avons qu’une seul property en plus dans les tests que dans le code.

#### Une partie des properties sont génériques, un sous-ensemble des properties java sont donc utilisés de la même manière dans tous les projets

Un diagramme nommé CountProperties dans le dossier Global affiche les properties qui ont été retrouvées dans plusieurs projets cependant ce diagramme est trop large pour être inséré dans le rapport. Nous allons donc nous concentrer sur l’extraction des données du csv propertiesNames qui contient les mêmes informations.

Sur les properties 567 référencées, 549 ne sont présentes que dans un seul projet, ce qui représente 97% de l’ensemble des properties. On peut donc en déduire qu’effectivement une sous parties est utilisée dans plusieurs projets.

![](https://lh3.googleusercontent.com/x8-APYCrkrjSn1I3k5ESjcQUMo341-vsyCaPtfev3zMtsjlfKX5vuTjmsPHMSll_OH4CcXTMN7-qFE85YvAXaZyN_vj39vKXbIwHqFyHdfvgnlq1O7MnyuR5hGrIB-Vedvbss-Qa)

Si l’on regarde ces données extraites de notre csv on peut constater que comme nous l’avions supposé les properties utilisées dans plusieurs projets sont bien des properties de configuration comme la configuration du rootLogger de log4j qui revient 4 fois dans nos 20 projets analysés.

### Conclusion

Si l’on reprend nos trois questions initiales :

* Où les développeurs utilisent les properties ?
* Comment les properties sont utilisées ?
* Est-ce que la majorité des projets utilisent les properties Java dans un même but ?

Nous pouvons conclure en indiquant qu’il ne semble pas y avoir de réelles tendances dans les projets que nous avons analysés quant à la localisation des properties même si les projets les utilisant dans les tests et le code dominent.

Cependant nous pouvons dire qu’aucun de nos projets analysés n’utilise de wrapper externe, tous les accès aux properties se font grâce aux méthodes du JDK.

Quant à la dernière question de façon générale les projets utilisent les properties afin de répondre à des besoins qui leur sont propres, mais un sous ensemble de properties peut se retrouver dans plusieurs projets et ce sont des properties de configuration comme celles concernant les paramètres d’un logger.

![](../.gitbook/assets/image%20%2819%29.png)

