# Comment les projets s’adaptent à l’évolution de Docker ?

## **Comment les projets s’adaptent à l’évolution de Docker ?**

**Février 2018**

## **Authors**

Nous sommes quatre étudiants en dernière année à Polytech Nice-Sophia, spécialisé en Architecture Logicielle.

* César Collé &lt;[cesar.colle@etu.unice.fr](mailto:cesar.colle@etu.unice.fr)&gt;
* Loris Friedel &lt;[loris.friedel@etu.unice.fr](mailto:loris.friedel@etu.unice.fr)&gt;
* Loïck Mahieux &lt;[loick.mahieux@etu.unice.fr](mailto:loick.mahieux@etu.unice.fr)&gt;
* Thomas Munoz &lt;[thomas.munoz@etu.unice.fr](mailto:thomas.munoz@etu.unice.fr)&gt;

## **I. Research context /Project**

Nous aimons utiliser Docker dans nos projets au quotidien et nous constatons que Docker ajoute \(et dans une moindre mesure supprime\) régulièrement des fonctionnalités, de plus c’est une technologie jeune \(4 ans\) et souvent considéré comme “à la mode”, c’est pourquoi nous nous posons des questions concernant la légitimité des nouvelles fonctionnalités, la gestion faite par Docker de son produit ainsi que le niveau de maturité de cette technologie.

## **II. Observations/General question**

Docker est devenu un outil essentiel pour le développement et la mise en production en entreprise. Nous nous sommes posé la question de savoir comment la communauté s’adapte aux modifications de Docker. En effet, pour un outil devenu aussi essentiel dans le développement de produits en entreprise il est important de s’assurer qu’il évolue pour s’adapter aux usages qu’il en est fait.

Aussi, nous nous sommes demandés, au vu de notre expérience dans l’utilisation de cet outil si l’évolution lente que nous constations \(depuis 2 ans peu de nouveautés ont été introduites\) résultait d’un manque de besoin pour davantage de nouveauté ou d’un retard de la part de la société Docker \(et de la communauté\).

Pour résumer, les questions que nous nous sommes posées sont les suivantes :

1. **Est-ce un produit assez stable pour être utilisé en entreprise pour des projets en production ?** 
2. **Quelles sont les fonctionnalitées réellement adoptées/utilisées par les utilisateurs de Docker?**
3. **Comment les Dockerfile sont-ils maintenus au cours de la durée de vie d’un projet ?**

## **III. Information gathering**

Nous avons analysé 1469 projets originaire de Github pour mener à bien notre étude.

Pour être sélectionné, un projet doit :

* contenir un Dockerfile.
* être parmi les projets avec le plus d’étoiles et de contributeurs sur Github \(dans le top 1000 à chacune de nos recherches\).

Nous avons ainsi effectué plusieurs recherches, par langage \(Java, Go, Ruby, C/C++ etc.\), en récupérant à chaque fois les 1000 premiers projets \(ordonné par nombre d’étoiles\).

Ensuite, nous avons filtré les projets contenant au moins 1 Dockerfile \(afin de ne pas essayer d’analyser des projets n’en contenant pas\).

Ensuite, nous avons trié les projets les plus importants en deux catégories : ceux qui sont fortement liés à Docker \(tel que Traefik\) et ceux qui l’utilisent simplement pour avoir une image de leur application \(tel qu’une application web standard\).

## **IV. Hypothesis & Experiences**

### **Identifier les commandes les plus utilisées**

#### **Répartition des commandes**

Grâce à la répartition des commandes, il nous est possible d’identifier les commandes “indispensables” à la conception d’un Dockerfile et celles qui le sont moins.

![](https://lh4.googleusercontent.com/jsgJAKKjPTIR_u2KOT5Y7hJNlxONJeeDFYCJ4tAeNMWCD9iENB9447efANHhXMsJEa6OOUP4090KXoCroF48wDYy_cmCeRVjZTbWURtMF7JdUblcAUscYqW6puORE8Oej8SVvz4q)

#### **Commentaires sur les résultats**

Sur ce graphique, on peut constater qu’un noyau de commandes très utilisées, composé deFROMetRUNexiste. Aussi, les commandes les plus utilisées sont des commandes déjà disponible dès le début de Docker \(ou dans les mois qui suivent sa sortie\), il s’agit de WORKDIR, CMD, ENV, COPY, EXPOSE, ENTRYPOINT, ADD, MAINTAINER \(maintenant dépréciée\), etc. Les nouvelles commandes \(arrivées plus récemment comme ONBUILD, HEALTHCHECK ou LABEL\) sont peu utilisées.

On peut alors se demander si les nouvelles commandes proposées apportent des fonctionnalités suffisamment intéressantes pour être utilisées plus largement. Par exemple la commande MAINTAINER est plus utilisée que la commande LABEL \(alors que MAINTAINER est dépréciée au profit de LABEL\)

La forte utilisation d’une commande est un indicateur de son importance. Aussi, un noyau composé uniquement de commandes anciennes \(c’est-à-dire présentes depuis la création de Docker, comme FROM, RUN, etc.\)

### **Adoption de nouvelles commandes**

#### **Latence d’utilisation des commandes**

Grâce à cette métrique, il s’agit de savoir si les nouvelles commandes disponibles sont rapidement adoptées dans la communauté. L’intérêt ici est de constater notre hypothèse de départ \(renforcée par les résultats précédents\) que le noyau de commande est suffisant et que les nouvelles commandes n’étaient pour la plupart, pas nécessaires.

![](https://lh5.googleusercontent.com/ePSnnP63B1bLdV2mMOrb0NApKT5-AKzQ3Ox-9cTS460J3-rntxq1mLG4jP8OlviG8Q8UCzfkCpda-3e38_h7cFljiYVR4sX0gYsLH8Be7jEOaIR5BD1aBnBwhUjwSdMrEmEEI46i)

![](https://lh4.googleusercontent.com/dZ-3oT-a8-hlg2bMjStVDkKUmz2DpVRIVZyX0ypZzguvbBQjrWH0A3gIn-a7prChmDzpuTU5xw-PcrOSXc0KgLvvRDmdRzrHgKXRVP9TMl27pmdA_wmqFLrs-BvrEAfMo1b1bSYI)

#### ![](https://lh6.googleusercontent.com/1eOx8ncO_jSj5c_YYjTOz5cMR6yvEj6U9J4zgGrSJ4V_xZdJrjej0PCAacwo9aWRp5ljtHpJqQ3gX0VYi25EAA1gUPBev9VDOpRThTcqDBUJqEnsftRW-0UZZmXuglglWvuFh-20)

![](https://lh6.googleusercontent.com/MnlSoMv31guG7TdOP5hHiiWHkywRBZz2eyUbx7pDLgGSJ2pkWVj-W19GhGwLBkcq0ZioXERJCqJr1hETKjbGlH_ijgNvO_Iv4MAXSJdoVurRget3PB-FLqqRp5KInR39m4dhKHDh)![](https://lh6.googleusercontent.com/144PcmnhMPDg0hyIs8D1wVfSdzjPunewb_9Qqjz39nB8uPRvMCY4svtw_q7P8Ws0-m1cTEYxdXw88JTBrJjV7ygrjQJ3kL-IwBPM0X_yLMQNUnPc17nPAyae-CdDm3L3xup9_uVh)

#### **Commentaires sur les résultats**

Nous rappelons que les commandes les plus récentes de docker sont **: LABEL \(2016\), SHELL \(2016\), HEALTHCHECK \(2016\)**

Contrairement à ce que nous pensions, l’adoption de nouvelles commandes est assez rapide, cela peut alors signifier deux choses : soit les nouvelles commandes apportent des fonctionnalités répondant à un réel besoin, soit la communauté d’utilisateurs de Docker sont victimes de la “mode” et utilisent rapidement les nouvelles fonctionnalités. Le fait de ne s’intéresser qu’à des projets Open Source peut dans notre cas biaiser notre interprétation. En effet, l’adoption rapide de nouvelles commandes \(par exemple LABEL à la place de MAINTAINER\) peut correspondre à une volonté de suivre les bonnes pratiques, caractéristique souvent rencontrée dans les projets open source \(pull request systématique, code review etc.\).

#### **Catégorisation**

Nous avons exécuté un algorithme de clustering sur les set de commande utilisé par les Dockerfile. Nous avons pu identifier deux catégories bien distinctes, nous ne montrerons cependant pas les graphiques pour cause de difficultés techniques.

* Projet Web : FROM, RUN, EXPOSE, CMD, COPY
* Projet autre : FROM, RUN, VOLUME, LABEL, ENV, ADD

## Tools

### **GitHub Javascript Client**

[**https://github.com/pksunkara/octonode**](https://github.com/pksunkara/octonode)

Afin de pouvoir automatiser le clonage des dépôts, nous avons utilisé le client Github en JavaScript \(NodeJS\) qui nous a permis de récupérer une liste de dépôts à partir d’une recherche \(sur un topic particulier, comme par exemple Java ou Go\).

### **Repo-driller**

[**https://github.com/mauricioaniche/repodriller**](https://github.com/mauricioaniche/repodriller)

Toute nos expériences sont basées essentiellement sur l’utilisation de repo-driller. Repo-driller nous a permis notamment de pouvoir filtrer les commits provenant des dépôts afin de se concentrer seulement sur ceux impactant un dockerfile.

Sur ces commits, nous avons pu récupérer toutes les modifications et à quelle date elles ont été faites. Ces données ont ensuite été traité par un module d’analyse externe.    
****

### **V. Result Analysis and Conclusion**

Par rapport aux questions que nous nous posions au début de cette étude, nous nous sommes rendus compte \(et avons pu chiffrer\) la lenteur qu'avait Docker à ajouter de nouvelles fonctionnalités pertinentes. Cependant, les utilisateurs de Docker semblent utiliser un ensemble de commande fixe et stable, formant ainsi le noyau dont on supposait l’existence intuitivement.

### **VI. References**

[**https://www.infoq.com/articles/docker-future**](https://www.infoq.com/articles/docker-future)

[**http://www.redbooks.ibm.com/redpapers/pdfs/redp5461.pdf**](http://www.redbooks.ibm.com/redpapers/pdfs/redp5461.pdf)**\(p. 55-56-57\)**

