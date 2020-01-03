# Comment les utilisateurs de Docker Compose s'adaptent-ils aux changements de version selon les spéci

**Février 2018**

## Auteurs

Nous somme cinq étudiants en cinquième année de science informatique à l'école Polytech Nice-Sophia Antipolis, et nous nous spécialisons en Architecture Logicielle.

* Maxime Carlier &lt;maxime.carlier@etu.unice.fr&gt;
* Rami Ajroud &lt;rami.ajroud@etu.unice.fr&gt;
* Danial Aswad &lt;danial-aswad.bin-ahmad-fazlan@etu.unice.fr&gt;
* Ahmed Fezai &lt;ahmed.fezai@etu.unice.fr&gt;
* Thomas Suignard &lt;thomas.suignard@etu.unice.fr&gt;

## Introduction

Dans le cadre de la matière Rétro-Ingénierie, Maintenance et Évolution des Logiciels \(RIMEL\), ce chapitre présente le travail que nous avons réalisé durant ce semestre. Ce document montre explicitement le contexte de notre travail et la problématique évoquée, suivie par nos hypothèses, puis les démarches effectuées et enfin, les résultats. À la fin de ce document nous établirons une conclusion sur la validité ou non de nos hypothèses ainsi qu'une réponse a notre question générale.

## I. Contexte de recherche

“Docker est un logiciel libre qui automatise le déploiement d’applications des conteneurs logiciel”  
-- Wikipédia

C’est ainsi que l’encyclopédie libre à choisie de présenter ce logiciel ayant connus un boom phénoménal dans les 5 dernière années.

Ainsi, Docker est un logiciel qui permet de déployer une stack logicielle indépendamment de la plateforme, mais qui offre en plus, d’autres outils permettant d’étendre au delà les capacités de base de la plateforme. Docker-Compose permet ainsi de déployer de manière groupé un ensemble de container pour contrôler de manière centralisé, le déploiement de plusieurs stack logicielle fonctionnant ensemble ou encore Docker Swarm qui vise à répondre au problématique de d'orchestration de plusieurs machines.

A l’heure actuelle, Docker est utilisé dans environs 10.000 projets Github, à levé environs 250 Millions de Dollar de fond et effectué déjà 8 acquisition \([https://www.crunchbase.com/organization/docker\](https://www.crunchbase.com/organization/docker%29\). Son succès, plus qu’académique pour les problématiques qu’il cherche à résoudre à su atteindre le monde industriel et attirer le regard de nombreux investisseurs. L’avenir semble donc radieux pour la boîte Californienne, et pourtant, une récente affaire vient ternire le blason de la baleine bleue.

Passé le domaine des conteneur dans lequel Docker à su s’imposer comme Leader incontesté, c’est au niveau de l'orchestration multi machine que la guerre fait rage. Pendant les deux années précédentes, ce sont Docker Swarm, Kubernetes et Mesos qui se sont livré bataille pour assouvir leurs règne. Et bien que Kubernetes soit clairement plus appréciés des utilisateurs \(33k star et 11.626 fork sur github contre 5k star et 1.023 fork pour Swarm\) c’est toujours Docker qui a la main sur le produit sous jacent.

Pourtant Docker à récemment décidé de faire marche arrière face à la pression d’utilisateurs soucieux d’un éventuel “Vendor Lock” qui se profilait à l’horizon, et choisis d’intégrer Kubernetes à son logiciel de Conteneurisation.

Ce revers qui peut paraître anodin, et même bienvenu de la part de l’entreprise Américaine, laisse pourtant entrevoir un certain point d’ancrage pour une argumentation en défaveur de la politique de développement que l’entreprise a choisis pour un tout autre logiciel de sa suite.

En effet les foudres n’ont pas finit de s’abattre sur Docker Compose et ses fameuse mises à jours destructrice. L’entreprise ayant fait le pari osé de ne pas garantir de rétro compatibilité entre versions, les utilisateurs ne sont pas à l’abris de voir tout bonnement disparaître l’un des opérateur qu’ils utilisent dans leurscompose fileentraînant ainsi un version lock de leurs système de déploiement a moins d’investir un temps-homme suffisant pour remédier à cela, lorsque cela est possible.

Mais au vu du revers de la politique de Docker sur Kubernetes, qu’adviendrait-il de sa politique de mise à jours pour Compose, s’il s'avérait que les utilisateurs ne tirait pas, ou peu de de bénéfice de cette approche ? Ou bien s’il ne mettait tout simplement pas à jour leurs Compose File ? Ou encore même, si l’implémentation des nouvelles version ne se limitait qu'à un groupe restreint d’utilisateurs, peu représentatif de la balance économique \(étudiants/académiques plutôt que industriels déboursant de l’argent\) ?

Ce sont ce type de questions auxquelles nous allons tenter d’apporter une réponse.

![](../.gitbook/assets/docker-compose-version-release.png)

_Figure 1: Figure 1: L'évolution de version de Docker depuis la première sortie \(20 décembre 2013\), version 1 \(en rouge\), version 2.X \(en orange\) et version 3.X \(en bleue\)._

## II. Observations et question générale

Pour faire suite aux problématiques soulevées dans notre contexte, nous commencerons par formuler une question de haut niveau, autour de laquelle nous construirons notre étude.

**Question générale : Comments les utilisateurs s'adaptent-ils aux mise à jour de Docker Compose en fonction des caractéristiques de leurs projets ?**

L’objectifs ainsi fixé, il s’agira donc pour nous, dans la suite de ce document de chercher si oui ou non il existe des critères propres à un projet, qui permettraient d’établir un lien de causalité entre leurs présence \(ou leur absence\) valué, et le dynamisme de changement de version des Compose File.

L'intérêt étant ici, que si il existe bel est bien un lien entre un tel critère, et cette action de mettre à jour son Docker Compose, nous serions en mesure de qualifier le groupe d’utilisateurs qui bénéficie le plus \(ou pas du tout\) de la politique de mise à jour adopté par Docker.

Ce résultat intermédiaire nous permettrait ainsi de conclure si le pari fait par Docker pour son outil Compose va à l’encontre ou non de l’usage qu’en font les utilisateurs.

## III. Rassemblement d'informations

Afin de procéder à cette étude, nous nous somme appuyé sur l’API Github. En l’intégrant au sein d’un projet Java disponible à l’URL suivante : [https://github.com/Maxime-Carlier/RIMEL-Docker](https://github.com/Maxime-Carlier/RIMEL-Docker), nous avons produit une application capable de récupérer un ensemble de projets nous servant de supports pour notre étude, puis d’appliquer pour chacun d’eux, un certains nombre de processeurs permettant d'en extraire des informations clé \(âge du repo, technologie, historique des commits\).

Le déroulement formel du rassemblement d’information est ainsi :

* Dans un premier temps, sélectionner l’ensemble des repository portant letopicDocker, et ayant la visibilité publique. \(n=1020\).
* A partir de ce premier ensemble, nous avons effectué un tris visant à écarter les projet, ne contenant pas au moins 1 fichier docker-compose.yml \(n=175\).
* Nous avons extraits des répo restant des informations pertinentes pour qualifier ces dernier \(nombre de commits, nombre de collaborateurs etc…\)
* Sur la base des fichiers restants, nous avons par la suite effectué un processus de drilling nous permettant pour chaque fichier Docker Compose d’un repository, d’en extraire l’ensemble des versions différentes ayant existé par le biais des commits qui les ont modifiées.
* Une fois l’ensemble de ces versions récupérées, nous avons récupéré pour chacune des versions du fichier ayant existé, la version spécifié dans contenu par le champ “version” du fichier docker-compose.
* Nous avons ensuite comparé un à un \(par ordonnancement chronologique\), chacune des version de docker compose, et lorsque celle-ci changeait d’un commit à un autre, nous avons sauvegardé la date du commit apportant ce changement
* Enfin nous avons calculé la durée en jours séparant la sortie de la version par Docker, et la date du commit sauvegardé précédemment.
* Sauvegarde des données au format JSON pour leurs exploitation

Une fois ces données produite, nous avons pu poursuivre notre étude en cherchant donc si cette durée en jours \(vélocité de mise à jour\) pouvait être reliée à un des critères.

Les figures suivantes montrent les caractéristiques principales de notre échantillon \(caractéristiques parmis nos critères choisis pour valider nos hypothèses\).

![](https://lh3.googleusercontent.com/AsMvXxqwjRnaLkxFeOn4nJczXegqjvkvAtqWpM9EDYQktfk8DQUqcyVMXG8pZvxHyldCyWviW-aXqw9aozKQ6Iak4Qd7j5NIdYnjgiwGh8h_Jq15tFLRAfXG3FcWMUEv6ZOZFrU)

![](https://lh5.googleusercontent.com/kIVT5nNwokQIee2IVzTdstvpTXYcNGesDlCFImxjRIFrUW8oxCk_gn-3RKfJ9XUuuBEKU2oU6r8I1ONavJhMXBwh64V2Occ_3MRL145B_vJj8PqxvzgITe1Mup7qTa_1g44rwoQ)

## IV. Hypothèses et expériences

En nous appuyant sur un sample de dépôt GitHub, et en les classifiants selon leurs caractéristiques disponibles telles que : le nombre de collaborateurs, le nombre de lignes de codes, le langage source, les technologies utilisées. nous tenterons donc d’établir la présence ou l’absence du lien de causalité décrits précédemment.

Intuitivement, nous sommes en mesure de formuler plusieurs hypothèses qui ont un lien direct avec la question générale, et qui ne traitent que d’une caractéristique spécifique.

**H1 : Les projet avec peu de collaborateurs ont une vélocité de mise à jour élevée.**

Nous pensons qu’en effet les projets avec peu de collaborateurs ont une dynamique différente d’un projet contenant de nombreux collaborateurs. Si ces dernier nécessite un grand nombre de personnes, c’est que les projets sont de grande envergure, et ainsi, la charge de travail nécessaire un faire évoluer un composant \(Docker Compose dans notre cas\) vers une nouvelle version n’utilisant pas les même commande, est trop importante, apporte peu de bénéfice, et n’est parfois simplement pas envisageable \(Compose File très complexes\).

**H2 : Les projets anciens \(+ 2 ans\) ne mette plus à jours leurs Compose File**

Nous avons l'intuition que des projets ayant atteint leurs maturité et se trouvant à présent dans le cycle de maintenance n’ont aucun intérêt à aller chercher le “petit plus” de la version suivante. Concrètement, nous nous attendons à observer que les projets datant de plus de 730 jours n'effectuent plus aucune mise à jour de la version de leurs compose file.

**H3 : La mise à jour V2.X -&gt; V3.X est peu implémentée**

La célèbre introduction de la syntaxe V3 des compose file qui supprimait tout bonnement l’utilisation de certaine commande liée au volume \(volume\_driver, volume\_from\) n’a pas été grandement implémentée.

Nous pensons que ces options sont indispensable pour certain projet les utilisants, et que par conséquent, la mise à jour vers la version 3.X qui supprime ces commandes n’a pas été grandement implémentée.

## V. Analyse des résultats et conclusion

### a. Temps moyen du changement de version \(jours\) en fonction du nombre de contributeurs

![](https://lh3.googleusercontent.com/VV_nXa-wgTE0aUmXkesRLek0nShRr3tAxKt1vRE0Y8cVwCOyzmHzxWCzen7RHZ8fOdkP03Ss7XfcbD7ORVky-hJEgrO_CccDjCrJMwi8ReqrLwfCt57IyCLOvaj_wRcddyMO5I4)

Dans cette figure 1, nous avons représenté par un point chaque repository. un point est représenté en abscisse par son âge et en ordonnée son temps moyen de changement de version, tous deux en jours. Cette figure nous montre un nuage de point fortement dispersé. Nous en déduisant qu’il ne semble pas y avoir de lien entre le nombre de contributeurs et le temps moyen de mise à jour pour un repository.

La meilleur approximation \(parmis les approximations linéaires, puissance, polynomiale et exponentiel\) est une approximation exponentielle \(tracé bleu\). Elle suggère une légère hausse du temps de mise à jours en fonction du nombre de contributeurs, cependant le coefficient de détermination associé étant très faible, cette tendance est peu représentative.

Nous constatons enfin que les repos mettent en moyenne environ 268 jours pours passer d’une versions à l’autre de docker-compose, ce qui est assez long à l'échelle du développement d’un projet. Cela nous pousse à penser que les mise a jours s’effectuent surtout à la suite d’une nouvelle release \(268 jours étant un temps raisonnable pour concevoir et livre un produit\) pour redémarrer sur des bonne bases.

### b. Répartition des mises à jours en fonction de l'âge des repository

![](https://lh3.googleusercontent.com/aNNflAoSkYlEeprebtz1gCVn57vhwp28KP6-FQfriwjzK-SOgo3hQtvpT98tod7uny7TH1WC_OF7MSEkKOGG_l8J6ovVla_KeCReOiAUdiceGutaqlcbIYybG6TXmP3Wv1XbAUM)Pour répondre à l’hypothèse 2, nous avons mesuré la proportion de repository ayant effectué au moins une mise à jour de fichier docker compose au cours de son existence. Si, chaque repository effectuez une mise à jour régulièrement, on s’attend à ce que leurs proportion augmente dans le temps. La figure 2 nous suggère le contraire, la répartition des repository ayant effectué au moins une mise à jour reste assez stable au alentour de 60%. Cette tendance nous permet de déduire qu’environ 40% des repository n'effectuent et n’effectuerons jamais de mise à jour, quelque soit leurs âge.

![](https://lh4.googleusercontent.com/Gni2Qsw1Aeqc_uLfsNAT3nquvP0ohHSNFKyhNr2510wEc-538QLjtdFfXO6G933-kLe51vy7hZs1UoX4LNHWgyrlzPYDS5_6ljFqrqtpYXbudiH2oH1zHIf4kBoSFItDaLLicd8)

![](https://lh4.googleusercontent.com/lcMf1GSC2mupGCvBmH35F7rGHw-jPiAN0k6GIhboeGCWAQQ-yu1C6t71q9HPIJ82hNKUi0O2vKZhSGzxtlv6CpsB9VyO0Xf0CeEwGBTEywhIMa7gQVEeH-fJ05ibpjLBUh8NTOU)

La proportion des non augmente légèrement avec l'âge à partir de 1440 jours. Les repository concerné sont donc ceux qui ont connus la première versions de docker-compose \(il y a 1530 jours\). On peut penser que ces repository ont adopté la technologie rapidement puis n’y ont pas vu d'intérêt et ont donc décidé de ne plus la mettre à jour. Ils ont cependant gardé leurs fichiers docker-compose, probablement pour des raisons de stabilité ou car la livraison de fonctionnalitées étant plus importante, la priorité n’a pas était donnée à cette migration de versions.

### c. Temps moyen du changement de version \(jours\) en fonction de l'âge du repository

![](https://lh5.googleusercontent.com/RDn3lnkZo_0CHqozUuyKW8_ypwBgbsX-hykdzkXWjDOsuL0Lrlm5ZVKkGVzQboOKEz2FmHd1c9D-x-LG5K83e9r-8-bVce2GVuvyUOd6gHgxK_kJrGCtE6qeA5IZJUIS9Yr2za0)

Nous avons mesuré le temps moyen de changement de version en fonction de l'âge des repository. Le figure représente chaque point comme un repository avec en ordonné le temps moyen de changement de version et en abscisse l'âge de celui ci. Cette fois encore, ce nuage nous suggère qu’il n’y a aucun lien entre l'âge et le temps de changement de version. Notre meilleur approximation \(une fonction puissance\) nous donne un coefficient de détermination très faible \(0.015\) qui nous permet de conclure qu’il n’existe aucun lien entre ces deux paramètres.

### d. Conclusion

Pour reprendre notre question générale posée en début de document qui était :

Comments les utilisateurs s'adaptent-ils aux mise à jour de Docker Compose en fonction des caractéristiques de leurs projets ?

En conclusion, et en nous appuyant sur les hypothèse formulé précédemment, ainsi que sur les résultats présentés ci-dessus nous sommes en mesure d’affirmer, que parmis les métriques de projets que nous avons étudiés, il n'existe pas de corrélation permettant d’affirmer qu’une quelconque causalité existe entre les critères de projets GitHub, et le temps moyen d’adoption de nouvelle version de Docker Compose.

Néanmoins, ayant observé un temps moyens de mise à jours \(tout repository confondus\) de 268 jours, nous somme en mesure d’affirmer que la stratégie adopté par Docker sortir beaucoup de version par année \(8 en 2017, soit une version tout les 46 jours environs\) avec parfois des changements non rétrocompatible, et en opposition directe avec l’utilisation qu’en font les utilisateurs. Cette opposition peut s’avérer risqué pour Docker comme expliquée dans le contexte, et pourrait amener à d’autres opposition dans le futur voir à l’abandon de la technologie docker-compose.

### e. Limites de l’étude

Bien qu’un effort est été fait afin de fournir un grand nombre de projets sample en entrés \(1.020\) \(topic:Docker et visibilité:public\) au détriment parfois du temps nécessaire à processer toute les informations \(35.000 commits par repository parfois\), , après application de tous nos filtres, notre sample se retrouve à 175 projets, dont seulement 108 ont fait au moins un changement de version.

Afin d’approfondir la question étudiée, il serait probablement nécessaire, une autre manière de recueillir des projets samples répondant directement à nos critères.

De plus, notre échantillon de départ est biaisé par notre technique d'échantillonnage elle même \(choix par topic Docker afin de maximiser le nombre de repository utilisant docker-compose\), ce qui nous amène à relativiser les conclusions faites sur le sujet.

