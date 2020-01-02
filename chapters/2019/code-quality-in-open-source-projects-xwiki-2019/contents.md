# How to improve contributors onboarding

## Auteurs

Nous sommes quatre étudiants en dernière année de Sciences Informatiques à Polytech Nice-Sophia en spécialité Architecture Logicielle :

* BOUNOUAS Nassim &lt;nassim.bounouas@etu.unice.fr&gt;
* CANCELA VAZ Joël &lt;joel.cancela-vaz@etu.unice.fr&gt;
* MORTARA Johann &lt;johann.mortara@etu.unice.fr&gt;
* ROUSSEAU Nikita &lt;nikita.rousseau@etu.unice.fr&gt;

## I. Les contributeurs, la vraie ressource de l'open source

Le sujet ayant retenu notre attention concernait le projet XWiki. Le sujet initial, était subdivisé en deux questions :

* Comment améliorer la qualité de code chez XWiki ?
* Comment améliorer l'_onboarding_ des contributeurs chez XWiki ?

En partant de ces sujets nous avons redéfini une nouvelle problématique : Comment introduire une dynamique de contribution dans un projet open source et pérenniser sa communauté ?

L'intérêt principal de cette recherche réside dans l'importance de la communauté dans le monde open source. De nombreux projets open source constituent le socle de l'informatique moderne, que ce soit des éditeurs de texte, aux outils de _build_ jusqu'aux bibliothèques "usuelles" : Visual Code Studio, Maven, Gradle, JUnit, Mockito, Apache Kafka, Jenkins, Docker...

L'ensemble de ces projets constitue le paysage technologique de notre époque et ils sont devenus incontournables pour la plupart des développeurs. Derrière ces projets et organisations se cachent souvent des centaines de contributeurs, rémunérés ou non par des sociétés pour participer à l'avancée desdits projets. Le noyau Linux peut être considéré comme le projet open source ayant le plus impacté le monde de l'informatique ces 30 dernières années. Compte tenu de son usage, le noyau Linux regroupe de nombreux contributeurs professionnels \(_i.e._ rémunérés par une entreprise pour contribuer\).

Qu'en est-il des projets plus petits ? Comment peuvent-ils faire grandir leur communauté ? Comment peuvent-ils inciter les contributeurs à participer activement et sur le long terme au développement de leur projet ?

## II. Collecte d'informations

Nous avons ciblé 4 projets de taille comparable à XWiki afin d'en étudier les contributions :

* JUnit 5
* Mockito
* Hibernate ORM
* Log4j2

Afin de cadrer notre recherche nous avons dû définir la notion de **contributeur**. Nous avons donc restreint la définition de contributeur à toute personne ayant proposé un _commit_ accepté sur la branche principale \(`master`\) du projet.

Un entretien avec Vincent Massol \(CTO de XWiki SAS\) nous a permis d'ajouter une nuance dans notre analyse puisque XWiki est maintenu par la société XWiki SAS. Il ne faut donc pas nous attendre à avoir une forte participation de la communauté comparé aux membres de la société. Ce point est néanmoins nuancé par l'implication d'industriels dans l'ensemble de ces projets.

### Articles de recherche

Nous avons cherché des articles de recherche traitant de l'open source et des dynamiques de contributions. Nous avons traité à la fois des papiers de recherche, mais aussi des articles rédigés sur des blogs. Nous allons exposer ici une synthèse des points qui ont retenu notre attention, ainsi que les idées d'indicateurs qui en ont découlé.

Ces différents articles ont permis de guider nos recherches. Les points d'intérêts ont été relevés et intégrés dans les analyses. Enfin, le questionnaire que nous avons mené a grandement été construit autour des remarques et idées retenues dans l'ensemble de ces articles.

#### How to discourage open source contributions

> [danluu.com - Discourage-OSS](https://danluu.com/discourage-oss/)

L’auteur de cet article évoque les problèmes d’intégration des pull requests \(c'est à dire des contributions entrantes\) sur les projets open source.

En outre, il évoque le volume des `pull requests` pour les projets open source et comment elles sont traitées par les mainteneurs des projets. Il semble raisonnable d’extrapoler que les projets avec un grand volume de pull requests restant en “attente” de validation, ou d’intégration dans les branches de développement, sont des projets qui ne portent pas attention à ses contributeurs.

Potentiellement, le travail que peut engager un individu peut ne jamais être intégré. C’est un paramètre qui peut influencer un contributeur sur son engagement. Les développeurs peuvent craindre que leur potentiel travail soit dévalorisé et oublié.

Il n’est pas déraisonnable d’avoir des centaines de pull requests en attente si proportionnellement autant de pull requests sont intégrées.

On peut donc déduire un `KPI` qui est le taux `PULL REQUESTS EN ATTENTE / PULL REQUESTS INTÉGRÉES`. Si le projet possède une maintenance et un support pour sa communauté raisonnable, alors ce taux doit rester compris dans une fourchette entre 15 % et 25 %. Environ 80 % des pull requests seraient alors intégrées au projet.

Il peut donc être intéressant de voir l’impact de ce taux sur l'enrôlement de nouveaux contributeurs.

#### Dear Open Source Project Leader: Quit Being A Jerk

> [lostechies.com - Dear Open Source Project Leader: Quit Being A Jerk](https://lostechies.com/derickbailey/2012/12/14/dear-open-source-project-leader-quit-being-a-jerk/)

Cet article discute du facteur psychologique d’une personne à contribuer à un projet. L'article reprend plusieurs points :

* le fait qu’un ticket \(`issue`\) reçoive une réponse dans un temps raisonnable \(de même pour une `pull request`\)
* la manière dont les mainteneurs perçoivent la question \(ou la contribution\) et comment ils y répondent.

On peut se poser alors les questions :

> Est-ce que les projects leaders sont tous bienveillants ? Peuvent-ils influencer le choix d'un potentiel contributeur à venir collaborer sur un projet ?

Cet article soulève le fait que certains mainteneurs, qui ne sont pas bienveillants, ne poussent pas à la contribution.

Le niveau des contributeurs est hétérogène, mais ce n’est pas une raison pour exposer publiquement les erreurs de certains, menant à une sorte de `Wall of Shame`. Cette attitude est un frein évident à la contribution pour des développeurs qui sont effrayés d'être exposés publiquement à la critique.

> “I can’t think of a better way to get people to stop contributing to open source projects. Seriously… there is nothing more demotivating and demoralizing than this kind of high-school-bully response. It needs to stop.”

Pour tenter de mesurer ce comportement, on peut tenter d’analyser le champ lexical des réponses aux `issues` et `pull requests`. Est-ce que des mots comme “lol” “wtf” etc. apparaissent ? Est-ce que le mainteneur prend une position agressive ? En soit, une sorte d'appréciation morale de la réponse. On peut tenter d’extraire depuis les issues les réponses “officielles” faites par les mainteneurs et les analyser. On peut en déduire un score de “Karma” de bienveillance subjectif.

#### The Economics of Technology Sharing: Open Source and Beyond

La dynamique des entreprises à contribuer dans les projets open source est aussi un paramètre à prendre en compte dans la motivation des contributeurs.

Généralement, les contributeurs ne sont pas rémunérés et vont travailler, de leur plein grès, sur le composant de leur choix dans le contexte d’un projet open source.

Dans le cadre des projets OSS industriels, les entreprises peuvent motiver leurs employés à contribuer sur leur temps de travail sur ceux-ci. La collaboration est donc différente du volontariat : l’entreprise rémunère et investit sur les produits OSS.

Le papier aborde aussi le sujet des incitations sociologiques qui poussent à contribuer à un projet.

Ces incitations sont les points clés pour faire évoluer la performance de groupe :

* la visibilité de cette performance pour le public
* l'impact de l'effort sur la performance
* la reconnaissance des talents  

Ces incitations sont cependant difficilement évaluables dans le cadre de notre étude.

Les auteurs continuent et évoquent le rôle moteur d'un chef de projet. Un projet mené par un "leader" efficace sera plus fort. Les contributeurs seront incités à suivre son exemple et à rendre le projet meilleur. Le respect et l'autorité naturelle des mainteneurs proviennent de leur manière de gérer le projet, mais aussi de la manière dont ils interagissent avec les membres de la communauté.

Les auteurs poursuivent en ouvrant une discussion sur les intérêts économiques personnels.

Typiquement, la possibilité d'une augmentation des revenus dans le futur est un facteur de motivation pour contribuer à des projets open source. En effet, l'expertise que l'individu va développer au travers de ses contributions va permettre de challenger de manière pécuniaire son employeur ou ses futurs employeurs.

D'autre part, un individu peut spontanément contribuer afin de répondre à un besoin auquel le produit ne répond pas. Il va alors développer sa solution pour répondre à ses besoins spécifiques, et pourra la partager avec la communauté.

Enfin, la curiosité intellectuelle fait consensus comme étant le facteur poussant le plus un individu à développer sur un projet open source.

#### Motivation Of Contributors In Open Source Software Development Projects

> Ce papier traite plutôt de l'aspect psychologique des personnes contribuant à l'OSS.

Dans une première partie, les auteurs du papier discutent du concept de motivation, qu'est-ce que c'est et qu'est-ce que ça signifie. Dans un second temps, l'article propose une étude des facteurs motivationnels du développement open-source. Le papier poursuit et effectue une analyse des motivations intrinsèques \(émanant de l'individu\) et extrinsèques \(hors de l'individu, provenant de son environnement social\). Les auteurs concluent sur les limitations et ouvrent sur les poursuites de recherches.

L'étude ouvre sur le fait que l'apprentissage dans les projets open source offre une satisfaction "intrinsèque" et "extrinsèque" au développeur.

L'apprentissage est mis en avant : c'est le principal facteur de participation des personnes dans le monde OSS. C'est une motivation intrinsèque. C'est l'apprentissage par le _savoir-faire_ : moins théorique avec une approche pragmatique. Les idées de projet proviennent d'une volonté de créer des solutions _différentes_ de ce qui existe. Ces projets permettent d'étendre les connaissances sur les disciplines connexes au projet. Les auteurs insistent sur le fait que les mainteneurs du projet doivent construire un environnement permettant l'épanouissement des participants au projet dans leur acquisition de connaissances.

Le papier stipule que la contribution à l'OSS permet au développeur de "satisfaire son ego". Cela veut dire que même s'il n'est pas rémunéré, le développeur a un "bon" sentiment lorsqu'il contribue au projet quand il est reconnu et apprécié pour son travail par les autres membres de la communauté. L'estime des autres est un facteur psychologique important. De manière analogue, le souhait de gagner le respect d'une institution ou d'un groupe d'individus peut être un moteur de motivation pour contribuer \(accomplissement de soi\). Souvent, ils ne sont pas attirés par l'aspect financier, mais plutôt dirigés par des facteurs compétitifs de statut \(le gain en renommée et en réputation\). De manière liée, certains développeurs peuvent contribuer à l'open source dans un combat ouvert contre les solutions propriétaires et fermées \(idéologie\).

La motivation à contribuer en équipe est plus élevée si la personne sent que son travail compte, mais aussi si l'objectif de l'équipe à laquelle il est attaché a de la valeur pour lui.

**Résumé des motivations intrinsèques :**

* plaisir de coder.
* augmenter l'expertise et les connaissances techniques
* le désire d'aider les autres en partageant ses connaissances
* volonté de garder un produit libre
* contribuer est pour certaines personnes un échange équivalent pour leur usage de solutions OSS

**Résumé des motivations extrinsèques \(sociales\) :**

* corriger un bug ou implémenter une nouvelle fonctionnalité pour répondre à leurs besoins dans le cadre d'un logiciel OSS qu'ils utilisent
* rémunération financière

**A propos des rémunérations financières :**

Celles-ci peuvent être vues comme une manière de contrôler la contribution au lieu de laisser au libre arbitre de chacun le choix de participer ou non dans les projets. Pour beaucoup de développeur, avoir le choix de pourvoir moduler le temps et l'effort à consacrer pour les projets open-sources est important. L'argent peut influencer ces décisions organisationnelles.

## III. Une chambre bien rangée est-elle plus accueillante ?

En débutant cette étude chaque membre de l'équipe a posé sur le papier ses impressions et ses préjugés. Nous nous sommes alors rendu compte que nous arrivions à la même hypothèse : pour nous, les principaux facteurs permettant d'attirer des contributeurs dans un projet open source résidaient dans la bonne tenue de celui-ci, une bonne documentation et des règles de contributions mises en avant et détaillées.

En partant de cette hypothèse nous avons défini des métriques pouvant, d'après nous, la valider ou l'invalider. Les métriques retenues étaient donc :

* Présence d'un `README.md` détaillé avec ces informations :
  * Présence d'une section _"Getting started"_ ou ce qui s'en rapproche afin de proposer un point d'entrée à la contribution.
  * Présence d'une section _"Getting help"_ qui détaille comment contacter les mainteneurs du projets en cas de questions.
  * Présence de badges \(ou [_shields_](https://shields.io/)_\)_ montrant l'état du _build_, la couverture de tests ou d'autres informations utiles.
  * Présence d'une section _"How to build"_ ou ce qui s'en rapproche détaillant comment construire la solution localement.
* Présence d'exemples de code.
* Présence d'un `CONTRIBUTING.md` qui définit les conventions de nommages, les messages de _commits_, etc.
* Présence d'un outil d'intégration continue \(Travis CI, Jenkins, etc.\)
* Présence d'une documentation tenue à jour
* Nombre de jours depuis la dernière mise à jour de la documentation, si disponible
* Temps moyen de réponse aux _issues_
* Nombre de _commits_
* Nombre de contributeurs
* Analyse des contributeurs et de leur contributions

Nous souhaitions initialement réaliser un sondage en ligne afin de pouvoir confronter nos résultats à l'avis d'un groupe plus ou moins grand de développeurs. Nous avons décidé de réaliser ce sondage en fin d'analyse afin d'une part de ne pas être guidé dans nos recherches par les résultats de ce dernier et d'autre part pour nous permettre d'affiner les questions.

## IV. Analyse des résultats

### JUnit5

JUnit5 est un _framework_ de test unitaire, un des plus utilisés pour le langage Java. Cette version majeure 5 succède à la version 4 et apporte beaucoup de nouvelles fonctionnalités majeures. Cette version 5 est aussi une refonte du framework et par conséquent se trouve sur un _repository_ à part.

#### Analyse des KPI \(analyse faite le 27 Janvier 2019\)

* Présence d'un README.md :
  * Présent et est assez complet, contient les parties \(_Contributing, Getting Help, Continuous Integration Builds, Code Coverage_ et _Building from Source_\).
  * Il est possible de discuter en ligne avec l'équipe de développement sur Gitter ou de manière indirecte via StackOverflow.
* Méthode de contribution :
  * Chercher les issues taguées avec "_up-for-grabs_" \(qui sont très peu nombreuses, 10 seulement au moment de la vérification\)
* Badges :
  * Travis.CI et Appveyor, tous les deux au vert au moment de la vérification avec le label "_Build passing_"
* Présence d'une intégration continue sur Travis.CI et Appveyor.
* Présence d'un CONTRIBUTING.md
  * Présent et détaille très bien les conventions de nommage et de formatage du code.
* Exemples de code :
  * Présent sur un autre _repository_, lien référencé dans le README.md
* Présence de Javadoc, à jour et mise à jour automatiquement à chaque _commit_
* Temps moyen de réponse aux issues: environ 1h
* Nombre de commits sur master : 5417 au moment de la vérification
* Nombre de contributeurs : 95

JUnit5 semble remplir la très grande majorité de nos critères pour être un projet attirant, voyons maintenant avec l'analyse des contributions si le projet est porté par les membres de l'équipe JUnit ou par sa communauté.

#### Analyse des contributions

Le projet a démarré en octobre 2015, comporte plus de 5400 _commits_ et 95 contributeurs.

Parmi ces contributeurs :

* 44 contributeurs avec 1 _commit_ \(46,3%\)
* 43 contributeurs avec 2 à 16 _commits_ \(45,3%\)
  * dont seulement 6 contributeurs avec au moins 2 _commits_ espacés d’au moins une semaine.
  * et un membre de Neo4j \(un autre projet Java de gestion de base de données basé sur les graphes\).
* 8 principaux contributeurs: \(8.4%\)
  * Top 8: un compte nommé “junit-buildmaster” qui sert pour des opérations de gestion du code \(indentation de code, renommage de variables, mise à jour de _headers_ dans la documentation\).
  * Top 7: un contributeur lambda qui ne contribute plus depuis 2 ans \(dernier _commit_ octobre 2017\)
  * Top 6 : un autre contributeur lambda qui ne contribue plus depuis janvier 2017
  * Top 5 : un contributeur lambda encore très actif \(dernier _commit_ 24 janvier 2019 au moment de la vérification\)
  * Top 4 : un membre de JUnit, son premier commit a été plus tardif que les autres membres mais, il est plus actif \(dernier _commit_ 14 janvier 2019\)
  * Top 3: un contributeur lambda, grosse contribution entre fin 2015 et 2016, depuis plus rien.
  * Top 1 et 2: deux membres de JUnit qui ont contribué au projet depuis le début et qui continuent.

Une recherche sur le [site portfolio](https://blog.johanneslink.net/2016/04/16/goodbye-junit-5/) d'un des 8 contributeurs a montré que les contributeurs 2, 3, 5 et 7 se connaissent et ont travaillé en équipe ensemble, et que suite à des conflits dans l’équipe ils ont cessés de travailler ensemble. Les contributeurs 3, 5 et 7 ne sont donc pas si “étrangers” au projet. Il semble donc qu’il n’y ait qu’un seul “vrai” contributeur externe au projet dans les 8 principaux contributeurs, le contributeur 6, qui ne contribue plus.  
Le projet est donc porté par les membres de l'équipe JUnit en très grande majorité.

Une KPI qui n'a pas vraiment été prise en compte est la complexité du projet, en effet, le projet JUnit5 est composé d'une vingtaine de modules Java et le coût d'entrée dans le projet semble être assez conséquent, même les issues "_up-for-grabs_" sont parfois incompréhensibles pour un néophyte.

A l'instar du comportement des mainteneurs relevées dans le papier "Dear Open Source Project Leader: Quit Being A Jerk", Nous avons décidé d'étudier quelques commentaires de certains membres de l'équipe JUnit à l'égard de nouveaux contributeurs. Les réponses données à certains contributeurs qui demandent s'il peuvent essayer d'implémenter une fonctionnalité ne sont pas très encourageantes. Ce qui a pour effet de créer une sorte de "syndrome de la tour d'ivoire", ou en tout cas on relève un certain manque de tact.

![Pas de r&#xE9;ponse pour les questions de ce pauvre contributeur](../.gitbook/assets/no-answer-to-question-issue.png)

![Pull request ferm&#xE9;e, mainteneur &quot;qui n&apos;a pas le temps&quot; de review le code](../.gitbook/assets/denied-pull-request.png)

![Pas tr&#xE8;s encourageant de dire &quot;on promet rien&quot;](../.gitbook/assets/essaie-mais-on-promet-rien.png)

Pour confirmer mes propos je me suis rendu sur le **Gitter** de l'équipe JUnit, me suis fait passé pour un nouveau contributeur qui aimerait contribuer mais qui ne sait pas par où commencer. Le résultat est assez décevant... les membres de l'équipe JUnit ne m'ont pas répondu et ont continué leur discussion. C'est un contributeur externe qui m'a aiguillé sur une de ses _issues_ qu'il a fait il y'a quelques années et qui s'est proposé de m'aider si j'en avais besoin.

![Capture d&apos;&#xE9;cran de la discussion sur Gitter](../.gitbook/assets/snobe.png)

Avec tous ces exemples, on peut en déduire que les contributions externes ne semble pas être une priorité pour l'équipe.

### Mockito

#### Analyse des KPI \(analyse faite le 10 Février 2019\)

* Présence d'un README.md :
  * Présent et à jour. Le document contient les grandes lignes permettant l'accueil dans le projet \(Version courante, liens vers les documentations fonctionnelles et techniques et les différents moyens de contacter l'équipe\)
  * Il est clairement écrit que le projet désire des contributions externes et tout est fait pour qu'un nouvel entrant puisse construire le projet et proposer des modifications/envoyer du code.
* Exemples de code :
  * Présents sur le site officiel du projet
* Méthode de contribution :
  * Sur les 239 tickets ouverts, 14 portent un label "please contribute" et sont adaptés à un nouveau contributeur
  * Un fichier CONTRIBUTING.md est présent et commence par les différents endroits où un individu externe au projet peut entrer en contact avec la communauté pour obtenir du support. Ce fichier décrit clairement les deux branches principales du projet \(version courante et version à venir\). Les attentes en terme de pull request \(commits, coding style et procédures\) y sont décrites.
* Présence d'une intégration continue sur Travis.CI et Codecov.
* Badges \(ou _shields_\) :
  * Travis.CI \(build : passing\)
  * Codecov : Couverture en test unitaire de 88%
  * Versions : Licence, Release notes, Dernière version téléchargeable \(binaire et maven\), dernière version documentée
* Présence de Javadoc, à jour et mise à jour automatiquement à chaque _commit_
* Temps moyen de réponse aux issues: De nombreuses issues ne reçoivent jamais de réponses
* Nombre de commits sur master : 4960 au moment de la vérification
* Nombre de contributeurs : 138

#### Analyse des contributions

Lors de l'analyse des contributions sur le projet Mockito, on se rend assez rapidement compte de l'importance du fondateur. Szczpan Faber qui se fait appeler "Mockito Guy" sur GitHub et Twitter représente à lui seul 75% des commits sur la branche principale de la nouvelle version du framework \(`release/2.x`\).

Le second contributeur en nombre de commits, Brice Dutheil, représente 11% des commits, et les contributeurs placés de la 3ème à la 7ème place représentent chacun entre 1 et 2.5 % des commits disponibles sur la branche principale. Les 131 autres contributeurs ont tous un nombre de commits très largement inférieurs à 1%.

Ces proportions sont à nuancer avec la paternité réelle au sens du nombre de lignes de code. On retrouve deux contributeurs de tête qui représente à eux seuls 70% du code : Brice Dutheil et Szczpan Faber. Le point intéressant concerne le nombre de lignes de code où Brice prend 45% contre 25% pour Szczpan.

Cette inversion montre que les commits à eux seuls ne sont pas représentatifs cependant, la tendance reste la même : Brice et Szczpan ont le leadership du projet et les 136 autres contributeurs ont une présence plus atténuée.

**Conclusion :** On peut retenir de cette analyse que bien que le projet ne soit pas tenu par un groupe plus grand que ce binôme, il est ouvert aux contributeurs externes. Cependant les contributions externes restent ponctuelles et ne constituent pas l'ossature principale du projet.

### Hibernate

Le projet Hibernate est composé de 39 dépôts. Nous avons choisi de concentrer notre étude sur le dépôt de _Hibernate ORM_.

#### Analyse des KPI

* Présence d'un README.md :
  * Contient les consignes de base permettant de construire le projet.
  * Propose une redirection vers le CONTRIBUTING.md.
* Méthode de contribution :
  * Un fichier CONTRIBUTING.md est présent. Il apporte des précisions sur les licences utilisées dans le projet et fournit un guide des étapes à réaliser pour pouvoir contribuer au projet.
  * Les issues sont gérées avec un JIRA externe. Les premières réponses sont de l'ordre de quelques heures.
* Badges :
  * Jenkins : _build passing_
  * LGTM \(qualité du code\) : A
* Intégration continue : Jenkins, LGTM
* Documentation : présence d'une AsciiDoc sur le dépôt du projet.
* Moyen de contacter l'équipe de développement : oui, sur le site de Hibernate \([http://hibernate.org/community/](http://hibernate.org/community/)\)
* Temps moyen de réponse aux issues : de l'ordre de quelques heures
* Nombre de contributeurs au moment de l'étude : 346
* Nombre de commits sur `master` au moment de l'étude : 9436

#### Analyse des contributions

En analysant la répartition des commits sur la branche `master` du projet, nous pouvons dégager plusieurs profils de contributeurs :

* 1er et 2e top contributeurs : contributeurs réguliers
* Du 3e au 10e top contributeur : contributeurs ayant bien contribué sur une période
* Du 11e au 50e top contributeur : contributions occasionnelles
* 51+ : contribution unique

Il est intéressant de remarquer que les 18 premiers contributeurs font partie de l'organisation GitHub _Hibernate_. Hibernate étant développé par JBoss \(qui est une division de RedHat\), nous pouvons supposer que ces contributeurs sont rémunérés pour contribuer au projet. Le premier contributeur, Steve Ebersole, est _project lead_ de Hibernate ORM.

De par la quantité de contributions produite par les développeurs de l'organisation _Hibernate_, moins de 20% des contributeurs ont produit et maintiennent plus de 80% du projet \(les 19 premiers contributeurs, donc 5% des contributeurs\).

**Conclusion :** malgré le fait que la communauté soit assez présente pour remonter les bugs, ce sont le plus souvent les membres de l'organisation _Hibernate_ qui répondent aux issues et corrigent ces bugs. L'implication de la communauté est donc relativement minime, et joue plutôt un rôle de QA que de contributeur.

### Apache Log4j2

Log4j2 est un utilitaire de journalisation pour Java. Il est distribué par la fondation Apache. Log4j2 est une mise à jour majeure du projet "Log4j", apportant de nouvelles fonctionnalités ainsi que des corrections sur l'architecture du projet mère. Le dépôt de code contient tous les sous-projets en lien avec l'utilitaire. C'est une réécriture de zéro de Log4j.

#### Analyse des KPI \(Mise à jour 25 Février 2019\)

* **README.md** :
  * La première section décrit la manière dont on doit collaborer avec le projet. En outre, elle explicite clairement que les soumissions \(a.k.a `pull requests`\) doivent suivre la licence `Apache license`.
  * Contient une section d'usage de l'utilitaire
    * C'est un exemple de code simple avec la procédure pour intégrer l'outil à un projet
  * Contient une consigne permettant de construire le projet
    * Une documentation plus complète est disponible sur le site de la documentation de _Log4j_.
  * Redirection vers le `CONTRIBUTING.md` du projet
    * Il apporte une procédure complète à suivre pour les personnes souhaitant contribuer en relevant un bug ou en ayant une idée de fonctionnalité.
    * Le fichier expose aussi les règles à suivre pour contribuer au projet. On remarque qu'il y a une procédure pour les modifications triviales \(ceux concernant la documentation ou les commentaires du code\), ainsi qu'une procédure pour les changements.
    * Il peut être nécessaire de signer un `Contributor License Agreement` pour les changements considérés comme non-triviaux
    * Une directive sur le style de programmation est aussi disponible sur le site de la documentation principale.
  * Lien vers la documentation détaillée du projet
    * Celle-ci est fournie, mais ne dispose pas de section `Getting Started` permettant à l'utilisateur d'appréhender le projet de manière dirigée et structurée.
  * Lien vers l' `issue tracker`. C'est une plateforme _JIRA_ externe. Les réponses sur celle-ci sont de l'ordre de quelques heures ou quelques jours dans le pire des cas.
  * **Badges** :
    * Travis-CI : build passing
    * Maven-central : numéro de version courante sur la branche stable
* **Integration Continue** :
  * La plateforme d'intégration continue principale est hébergée sur _Travis_. Elle effectue une intégration de l'ensemble du projet _Log4j_, incluant ses sous projets.
* **Documentation** :
  * Complète, avec des cas d'usage et des exemples concrets
  * Elle est consultable sous forme de fichier PDF
  * Elle est maintenue à jour \(Dernière mise à jour : 03/02/2019\)
  * Une version `asciidoc` est disponible
* **Exemples** :
  * Le dépôt propose plusieurs `samples` illustrant l'usage de l'utilitaire
* **Quelques chiffres** :
  * Le projet compte à ce jour `10 550 commits`
  * Le projet dénombre `59 contributeurs` et plus de `660 stars` sur GitHub
  * Il y a `31 pulls requests` en attente, contre `217` fermées
  * Le dernier commit est âgé de 19 jours.
  * Il y a des contributions régulières tout au long des mois \(au moins un par jour\)
  * Le projet possède un code stable depuis 2015 \(pas de pics d'ajout/suppression de code\)

#### Analyse des contributions

Après une analyse des contributions sur la branche `master` du projet, ainsi que l'uniformisation des identités \(commits sous le nom de différents pseudonymes\) on peut dresser deux profils de contributeurs.

Un premier groupe leader actif composé de quatre personnes:

* `Gary Gregory` est le mainteneur le plus actif du projet, avec le plus de paternité \(`4610 commits`, soit 43% de tous les commits réalisés\).
* Il y a trois autres contributeurs importants `Ralph Goers` \(le fondateur du projet\), `Remko Popma` et `Matt Sicker`. Ils ont chacun entre `1500` et `1000` commits, soit 34% de tous les commits réalisés \(3588\).

Ces quatre personnes détiennent environ 80% de la paternité du code. Ils continuent de contribuer et d'aider à l'intégration des nouvelles fonctionnalités. Ils s'assurent aussi de l'entretien du backlog et de la gestion des tickets sur le Jira. \(Voir : [https://logging.apache.org/log4j/2.0/jira-report.html](https://logging.apache.org/log4j/2.0/jira-report.html)\).

`Gary Gregory` et `Ralph Goers` sont membres de la fondation Apache, mais ils sont aussi attachés à une entreprise, respectivement `Rocket Software` et `Nextiva` et sont les fondateurs du projet. `Remko Popma` est un contributeurs open-source engagés dans le projet, alors que `Matt Sicker` travaille pour CloudBees.

Tous ont un profil de professionnels ayant eu besoin du produit pour leur entreprise. Ils sont aujourd'hui senior manager ou expert technique dans leurs entreprises respectives.

L'évolution de Log4j depuis 2010 démontre plusieurs points :

* Le projet a commencé avec `Ralph Goers`. Il travailla sur le produit durant deux années sans aides extérieures.
* `Gary Gregory` a rejoint le développement en octobre 2012
* Depuis octobre 2012, les différents contributeurs arrivant sur le projet sont d'environ 5 à 6 par an.

On remarque dès lors un motif de contribution où les nouveaux développeurs ne contribuent pas plus de quelques semaines \(ou un mois au mieux\).

**Conclusion** C'est le noyau "dur" qui fait vivre le projet. Les contributions \(hors mainteneurs clés\) proviennent pour la grande majorité d'ajout de fonctionnalité pour des besoins spécifiques. Les utilisateurs implémentent leurs besoins et les font partager à la communauté. Il y a cependant un manque cruel de fidélisation. Cela doit provenir de la nature intrinsèque du projet qui ne reste qu'un "outil" de gestion des logs. Les contributions à ce projet restent donc assez limitées et le motif observé est normal. Le facteur limitant est ici probablement un code hérité assez vieux \(9 ans\), avec un engouement assez limité pour motiver les nouveaux développeurs. Il est probable que Log4j convienne à presque tous les cas d'usage du logging après 9 ans de travail.

### XWiki

Le projet XWiki est composé de 8 dépôts. Nous avons choisi de concentrer notre étude sur le dépôt de _XWiki Platform_.

#### Analyse des KPI

* Présence d'un `README.md` :
  * Redirige vers la documentation ainsi que des pages de guide à destination des développeurs, des utilisateurs et des administrateurs. Le `README.md` comporte également une redirection vers le _Getting started_ pour les contributeurs.
* Méthode de contribution :
  * Des sections `Contribute Designs`, `Sponsoring issues`, `Contribute code` permettent d'aiguiller les contributeurs aussi bien dans le sens de la production que de la suggestion d'améliorations et de fonctionnalités
  * Un post `Onboarding` est dédié aux nouveaux entrants et propose des pistes \(`Available Tracks`\) permettant d'intégrer le projet, voici des exemples :
    * Report or find an issue
    * Understand XWiki concepts of XClass & XObjects
    * Contribute to the core
    * Write or run a test
* Badges : Non
* Pas d'exemples de code
* Intégration continue : Jenkins
* Documentation : disponible sur le site de XWiki, dont le lien est présent dans le `README.md`
* Moyen de contacter l'équipe de développement : plusieurs moyens explicités dans le `README.md` du projet
  * Blog
  * Mailing lists
  * IRC
* Nombre de contributeurs au moment de l'étude : 96
* Nombre de commits sur `master` au moment de l'étude : 36.461

Le dépôt de XWiki Platform comporte un community profile qui permet de mesurer à quel point le dépôt est conforme aux attentes standards de la communauté :

![XWiki Platform community profile](../.gitbook/assets/xwiki-platform-community-standards.png)

#### Analyse des contributions

Parmi les 96 contributeurs du projet, les 20 premiers contributeurs \(dont 15 font partie de l'organisation XWiki\) sont ceux qui portent vraiment le projet \(totalisant 32.398 commits sur 36.461 commits, soit environ 89%\). La loi de Pareto est vérifiée.

**Conclusion :** XWiki possède beaucoup des caractéristiques permettant de facilement rejoindre le projet \(équipe de développement disponible, documentation à portée de main\). La communauté a même participé au Google Code-in, ce concours vise à intégrer dans un projet des jeunes entre 13 et 17 ans et leur confier des tâches bien définies. Cette activité démontre une réelle prise en considération de la démarche d'intégration des contributeurs. Lors de notre entretien Vincent Massol, ce dernier nous a confié que la grande difficulté réside dans la constitution d'une communauté de développeurs et non dans la constitution d'une communauté d'utilisateurs.

## V. Sondage

Nous avions quelques intuitions quant aux facteurs qui contredisaient notre hypothèse de départ :

* La complexité trop haute des projets analysés
* Le manque d'investissement de la part des mainteneurs des projets
* Le manque d'une communauté réellement soudée

Nous avons donc décidé de mettre en place un sondage afin de vérifier nos soupçons, il est composé de 3 parties :

* Des questions sur la personne interrogée \(sa situation professionnelle et si elle est développeur\)
* Des questions sur son expérience de développeur \(ancienneté et si cette personne a déjà contribué à un projet open source\)
* Enfin, des questions sur son expérience de contribution:
  * Est-ce que cette personne est ou a déjà été payée pour contribuer sur un projet open source ?
  * A quelle fréquence contribue-t'elle ?
  * Selon elle, est-ce que la personnalité du mainteneur du projet est importante ? \(gentillesse, attitude, charisme\)
  * Est-ce qu'elle contribue individuellement ou en entreprise ?
  * Est-ce que la personne préfère contribuer seule ou en équipe ?
  * Quelle est la taille moyenne des projets auxquels elle contribue ?
  * Qu'est-ce qui la motive à contribuer aux projets open source ?
  * Quel est le principal obstacle qui l'empêcherai de contribuer ?
  * Est-ce qu'elle préfère un projet rigoureux mais avec une communauté peu accueillante et étroite d'esprit ou un projet moins rigoureux mais avec une communauté bein plus accueillante et plus ouverte ?
  * Enfin, une "question" libre dans laquelle elle pourrait nous raconter sa meilleure expérience concernant une contribution open source

Le sondage a été posté sur les réseaux sociaux Facebook et Twitter en visant plus particulièrement les communautés de développeurs et sur le forum de développeurs [Dev.to](https://dev.to/nirousseau/poll--open-source-contributions-and-motivation-factors-3o4l). Nous avons réussi à obtenir 189 réponses ce qui nous donne un échantillon de personnes interrogées assez intéressant à étudier, surtout quand on connaît la diversité des personnes qui sont inscrites sur Dev.to.

Les deux premières parties du questionnaire sont surtout des questions pour filtrer et cibler le public que l'on veut interroger \(les personnes qui développent et ont contribué au moins une fois à un projet open source\).

### Situation professionnelle et question de filtrage

La première partie avec la question "Êtes-vous développeur ou avez vous déjà codé ?" fait passer les nombres de personnes interrogées qui nous intéressent de 189 à 175.

![Diagramme de la situation des personnes interrog&#xE9;es, certains ont le sens de l&apos;humour, des chatons et des poilus notamment](../.gitbook/assets/1%20%281%29.png)

![Premi&#xE8;re question de filtrage](../.gitbook/assets/2%20%281%29.png)

La répartition des niveaux des personnes interrogées est assez uniforme, ainsi notre sondage touche plusieurs catégories de développeurs sans pour autant donner les résultats d'une majorité.

### Niveau d'expérience en développement et question de filtrage

La seconde partie contient une question sur le niveau d'expérience en développement de ces personnes ainsi qu'une question de filtrage.

![Diagramme de la r&#xE9;partition de l&apos;exp&#xE9;rience en tant que d&#xE9;veloppeur des personnes interrog&#xE9;es](../.gitbook/assets/3.png)

![Seconde question de ciblage](../.gitbook/assets/4%20%281%29.png)

Nous avons utilisé cette seconde question pour filtrer notre panel de personnes interrogées et ainsi n'avoir que l'avis des personnes ayant déjà contribué, au moins une fois, à un projet open source. Notre échantillon passe alors à 139 personnes.

### Comportements autour de la contribution open source

Nous exposons uniquement ici les résultats permettant de confimer ou infirmer notre seconde hypothèse. Cette hypothèse envisage la complexité d'un projet et l'attitude de la communauté comme principaux freins à la contribution.

![Apparemment l&apos;attitude des mainteneurs a une importance pour la majorit&#xE9; des personnes interrog&#xE9;es](../.gitbook/assets/7%20%281%29.png)

![4 obstacles &#xE0; la contribution ressortent principalement avec la complexit&#xE9; du projet, l&apos;attitude des mainteneurs des projets, le manque de documentation et du code peu lisible.](../.gitbook/assets/9_4.png)

![Apparemment notre premi&#xE8;re hypoth&#xE8;se &#xE9;tant loin d&apos;&#xEA;tre vraie, le social semble plus important dans la vie d&apos;un projet open source que la rigueur de son d&#xE9;veloppement.](../.gitbook/assets/9_5.png)

## VI. Conclusion

Pour conclure, ce sondage va dans le sens de notre seconde hypothèse, l'attention portée à la communauté par les mainteneurs de projet et la complexité de leur projet semblent être les principaux facteurs influençant l'attractivité pour les contributeurs. Avec du recul ceci semble assez évident, corriger la complexité des projets semble néanmoins une tâche difficile mais améliorer la communication avec sa communauté semble moins ardu. Une autre idée de notre part serait de gamifier ces projets dans le but d'attirer et motiver les contributeurs, une solution encore en beta test mais pouvant être envisagée est ProMyze Themis. Cette conclusion tend à démontrer l'intérêt des évènements physiques telles que les conférences. Les évènements comme les conférences Devoxx ou encore le SpringOne Platform offre l'occasion de tisser des liens au sein d'une communauté. Ce sentiment d'appartenance à une communauté semble être un grand facteur de motivation. Ce sentiment constitue le 3ème étage de la Pyramide de Maslow et les résultats obtenus nous dirigent vers l'ensemble de ces derniers trois niveaux à savoir : le besoin d'appartenance, l'estime de soi et le besoin d'accomplissement personnel. L'ensemble de ces résultats se retrouve étroitement lié avec l'étude de Dan Pink intitulée [The Puzzle Of The Motivation](https://www.ted.com/talks/dan_pink_on_motivation/up-next).

## VII. Outils utilisés

Au début du projet, nous avons commencé par déterminer des KPIs simples à évaluer, et pouvant être récupérées de façon automatisée. Nous avons donc mis au point un script permettant d'automatiser le téléchargement de sources depuis un dépôt Git.

Cependant, nous nous sommes rapidement aperçus que ces KPIs n'étaient pas assez fins pour nous permettre d'étudier le problème convenablement. Les KPIs ont donc été affinés, cependant leur évaluation ne pouvait plus être automatisée. En effet, l'estimation de la qualité d'un `README.md` ne peut se faire automatiquement. Nous avons donc effectué la majeure partie de cette analyse à la main, en nous aidant de plusieurs outils :

* [git-fame](https://github.com/casperdcl/git-fame) afin d'avoir un premier aperçu de la répartition de la paternité de code entre les différents contributeurs.
* [git inspector](https://github.com/ejwa/gitinspector) nous a permis d'établir l'analyse des commits.
* [Webscraper](https://www.webscraper.io/) afin de récupérer des informations directement depuis l'interface de GitHub.
* [L'API GitHub](https://developer.github.com/v3/) dans l'optique d'obtenir des statistiques sur les issues

## VIII. Références

* [https://github.com/xwiki/xwiki-platform](https://github.com/xwiki/xwiki-platform)
* [https://github.com/junit-team/junit5](https://github.com/junit-team/junit5)
* [https://github.com/hibernate/hibernate-orm](https://github.com/hibernate/hibernate-orm)
* [https://github.com/apache/logging-log4j2](https://github.com/apache/logging-log4j2)
* [https://github.com/mockito/mockito](https://github.com/mockito/mockito)
* [https://promyze.com/themis/](https://promyze.com/themis/)
* [https://dev.to/nirousseau/poll--open-source-contributions-and-motivation-factors-3o4l](https://dev.to/nirousseau/poll--open-source-contributions-and-motivation-factors-3o4l)

![UCA : University C&#xF4;te d&apos;Azur \(french Riviera University\)](../.gitbook/assets/entete-3.png)

