# Est-ce que les erreurs de merges viennent du code ajouté ou du code déjà présent ?

## Authors

* Chennouf Mohamed &lt;mohamed.chennouf@etu.unice.fr&gt; 
* Huang Shiyang &lt;shiyang.huang@etu.unice.fr
* Swiderska Joanna &lt;joanna.swiderska@etu.unice.fr&gt;
* Wilhelm Andreina &lt;andreina-simonett.wilhelm-garcia@etu.unice.fr&gt;

## I. Context

GIT est le logiciel de gestion de version le plus populaire au monde. Git gère l'évolution du contenu d'un projet et permet de synchroniser du code entre plusieurs collaborateurs. 

Durant nos études et nos expériences professionnelles, nous avons tous été confronté à travailler en équipe sur des projets divers et variés. Dans la majorité des cas, nous utilisons GIT pour versionner nos travaux. Au sein d’une équipe de travail, il est courant de créer des branches avec git afin que les membres de l’équipe puissent travailler sur des parties du projet sans impacter le projet actuel. Une fois le travaille sur la branche X accompli, il est nécessaire d’ajouter ce travail dans la branche principale du projet. Pour synchroniser ces branches, il existe deux méthodes : Merge et Rebase. 

Pour notre étude nous nous intéressons aux merges car c’est la synchronisation que l’on utilise le plus souvent durant nos projets. Un merge permet d’avancer la branche principale en incorporant le travail d’une autre branche. Cette opération join et fusionne les deux points de départ de ces deux branches dans un commit de fusion.

Voici un schéma pour illustrer un merge :

![1. Sch&#xE9;ma d&#x2019;un merge](../.gitbook/assets/image%20%284%29.png)





Un merge permet d’avancer la branche principale \( en blue \) en incorporant le travail d’une autre branche \(en vert\) . Bien que GIT soit un excellent logiciel de gestion de versions, les merges nous ont parfois causé quelques problèmes :

**Les conflits :** 

En effet, il nous est déjà arrivé d’avoir des conflits sur des merges. Les conflits se produisent lorsqu'on fusionne des branches ayant des commits concurrents. Git a besoin de notre aide pour décider quelles modifications seront intégrées à la fusion finale. La plupart du temps cela a lieu : lorsque plusieurs utilisateurs modifient la même section de code ou lorsqu'une personne modifie un fichier et une autre le supprime. Après résolution du conflit, les merges peuvent provoquer des erreurs dans le projet.

**Les erreurs de merges :** 

Qu’il y ai conflit ou pas, les merges peuvent mal se passer lorsque le projet ne se compile pas ou s'exécute de façon inattendue \(erreurs …\). Par exemple, cela peut se produire quand: 

* Du code ajouté est mal écrit
* Du code ajouté marche mais s’intègre mal avec le reste du projet
* Des tests ne sont pas mise à jour
* Des tests sont mal écrit

## II. Questions générales

**1. Existe-t'il vraiment des erreurs de merge ?**  


Nous nous sommes dit qu’il est possible qu’un merge se passe bien mais entraîne des erreurs dans le code. Afin de prouver que ce cas existe, nous avons pris les deux exemples suivants comme point appuis :

* Un merge se passe bien mais le projet ne se compile pas car il y a une ou plusieurs fautes de codes comme par exemple une erreur de frappes.
* Un merge se passe bien et il n’y a pas d’erreur de compilation mais certaines fonctions du projet ne marchent pas car il y a des modifications sur des interfaces ou des conditions qui ont impacté l’intégralité du système.

Nous avons démontré que ces deux exemples existent et trouvant à la main des cas dans divers projets : \(sonarqube , heron ...\). Ces deux exemples nous amène à nous poser la question suivante :  
****

**2. Comment repérer qu’un système ne se compile pas ou se compile mais a un comportement inattendu ?**  


De nos connaissances, le moyen le plus efficace pour savoir si un projet ne compile pas est de s'intéresser aux outils d'intégration continue qui propose, pour la plupart, des états du projet lors de chaque merge.  


Par exemple, sur les outils Jenkins et Travis, lorsqu’un projet ne compile pas un voyant rouge est affiché alors que lorsque les tests ne passent pas un voyant orange est affiché. Nous avons fait l’hypothèse que le projet compile et ne marche pas lorsque les tests ne passent pas.

De cette manière, en inspectant les outils intégrations, il nous est possible de déterminer quelles sont les merges qui sont en erreur. Pour débuter, nous avons choisi un projet qui:

* utilise un langage qu’on connais.
* a un accès public à son outils d'intégration continue.
* a suffisamment de merge qui se passe mal.

**3. Comment savoir si les erreurs de merges viennent du code ajouté ou du code déjà présent ?**

Intuitivement nous nous somme dit que les erreurs de merge peuvent venir :

* du code ajouté mal écrit
* du code ajouté qui marche mais qui s’intègre mal avec le reste du projet
* des tests qui ne sont pas mis à jour
* des tests mal écrit

A partir de ces idées nous avons décidé de prouver nos intuitions en créant des règles. Ces règles sont spécifique à chacune de nos intuitions que l’on testera sur plusieurs projet afin de conclure sur la provenance des erreurs \(codé ajouté ou code présent ? \) .

**4. A t’on accès aux informations qui nous intéresse ?**

Pour créer nos règles, nous avons supposé avoir :

* les commites en erreurs d’un Merge
* les commites qui précède et succède celui en erreur
* le nom des fichiers qui interviennent durant un commite

Mais a t-on ces informations et comment y avoir accès ?

Deux idées sont possible : Commande **Git shell** ou **API gitHub**. Nous avons préféré prendre API de github car grâce à l’API il est possible de  récupérer directement les objets que l’’on cherche sans avoir à faire du traitement sur les données reçus \(parsing, différence …\).

**5. Sur quelle projet tester nos règles?**

Nous avons décidé de prendre SonarQube comme premier projet test car il a :

* Un serveur d’intégration accessible : travis
* Un nombre de merge conséquent qu’il soit bon ou en erreur
  * 3185 pull request fermés
  * 2181 pull request mergés

**6. Comment pousser l’analyse plus loin  ?**

Nous avons trouvé un moyen de pousser notre analyse plus loin en inspectant le code des fichiers modifiés. De cette manière, on peut vérifier dans le fichier que le code modifié et le code qui a corrigé l'erreur viennent bien de la même section de code.

## III. Expériences et Analyse des résultats

**Règles :**

_**Tests pas mis à jour :**_ 

_`Commit_Error : files [ x.java, v.java] ---> Commit_Good : files [x.test ]`_

_**Tests mal écrit :**_ 

_`Commit_Error : files [ x.test, v.test] ---> Commit_Good : files [ x.test ]`_

_**Code ajouté défectueux :**_ 

_`Commit_Error : files [x.java, v.java] ---> Commit_Good : files [ x.java]`_

_`Commit_Good : files [ x.java, o.java] ---> Commit_Error : files [x.test, v.test, o.test] ---> Commit_Good : files [x.java]`_

_**Code ajouté marche mais produit des erreurs avec le reste du système :**_

_`Commit_Error : files [x.java , v.java] ---> Commit_Good : files [ o.java ]`_ 



Dans un premier temps nous nous sommes basé sur les nom des fichiers qui interviennent lors d’un Merge qui se passe mal. En raisonnant sur les commites avant et après celui qui a causé l’erreur, il est possible d’en déduire que :

* les tests n’ont pas été mis à jour ,
* les tests ont été mal écrit
* le code ajouté est défectueux
* le code ajouté marche mais produit des erreurs au sein du système.

Nous avons ensuite développé un logiciel qui exécute nos règles sur les pull requests en erreurs. Voici le résultat de l’expérience 1:  


![r&#xE9;sultat de l&apos;exp&#xE9;rience 1 sur sonarqube](https://lh5.googleusercontent.com/n4hr-T78q49VnqskW-aOdefKocb-Mste4XO-iCe7U5O5AMoJTuEae7zK4i5pCgA_muFc3tTu-51yoavkQX1jzDp1UwXYmsVSmgW5K766OjhnBdgdkt-TLezUjhYnpuEBSS_FHpw)

* les tests n’ont pas été mis à jour : 2
* les tests ont été mal écrit : 4
* le code ajouté est défectueux  : 12
* le code ajouté marche mais produit des erreurs au sein du système : 24

![Graphe mod&#xE9;lisant le r&#xE9;sultat de l&apos;exp&#xE9;rience 1](https://lh4.googleusercontent.com/KNJlrsbmL2GiHfV52jzCWTAn5Boj3v1oOy7KtOzWpV3BtKSgxdLpXUFBoz4WQyc10hvA6DFLy2XZQXDklvmlkLa0kmjTeTZL4SUR2nvaSf-8qgAVlVmLgKOqbm_oxge_SbdhG50)

Après cette première expérience, on peut se rendre compte que ⅔ des merges qui se passe mal sont dus à l’impact du nouveau code dans le reste du système. Afin de pouvoir confirmer cette première expérience nous avons décidé de lancer nos clauses sur d'autres projets :

Nous avons ensuite étendu nos expériences sur divers projets : Rxjava, mockito , incubator-dubbo, incubator-shardingsphere, incubator-skywalking, incubator-heron.

**1. Sonarqube raisonnement sur les noms des fichiers**

Nous avons décidé de nous intéresser au merge qui sont des pulls resquest afin de restreindre la taille des données manipulées. Dans un premier temps, nous avons extrait automatiquement 61 pulls requests contenant un total de 665 commis dont 36 commis sont en erreurs depuis le projet sonaqube.

**2. mockito raisonnement sur les noms des fichiers**

Ce projet à plus de 700 pulls requests, cependant pour cette expérience, les pulls request extraites viennent de deux versions du projet avec Jdk8 et Jdk11. Nous n'avons pas réussi à automatiser la séparation des différentes versions du code car la distinction des résultats de sous tests \( tests Jd8 et tests Jdk11\) sur Travis n'était pas possible. De ce fait les résultats obtenus ne sont pas très révélateurs. De plus, en inspectant à la main le github, nous avons remarqué que la version Jdk11 comportait la majorité des erreurs de merge.

* les tests n’ont pas été mis à jour : 34
* les tests ont été mal écrit : 27
* le code ajouté est défectueux  : 246
* le code ajouté marche mais produit des erreurs au sein du système : 128



![Graphe mod&#xE9;lisant le r&#xE9;sultat de l&apos;exp&#xE9;rience 2](https://lh3.googleusercontent.com/WHh-Q-s8ihTHBH8IFvyHX7L0-uBuDLJBiaWJYRz8jgUbs6b1CFcIj3LgYC21oEPVx9zUhYkk6pPVMPax36YQl3nQe0NGc2sfrU0q3f1Dtp_Av4sbuRKLwJ7Z5ornRzp-c1--igE)

**3. RxJava raisonnement sur les sections de code**

RxJava est un grand projet avec plus de 3000 pulls requests. Ce projet nous a étonnés du fait qu'il n'y a aucune pull request mergé en erreur sur son gestionnaire de version \(Travis\). De ce fait nous n'avons pas eux de résultat concluant pour ce projet.

**4. sonarqube raisonnement sur les sections de code**

Sur cette expérience nous avons décidé de raisonner sur les sections de code en utilisant l'API git qui nous permet d'avoir le numéro des lignes modifier. Ainsi en comparant les numéros de lignes, nous avons pu affiner nos résultats sur sonarqube :



![R&#xE9;sultat de l&apos;exp&#xE9;rience des sections de code modifi&#xE9;es sur sonaqube](https://lh4.googleusercontent.com/qAFEP7Wwk_kq7XOSmFTHUaUqMbW4i2x4fkgz1yexUe0rftJohcRZCg2k-ulVgxiRzhSmLZqE_1H30uz4McDj0fzdPzFEHuLF3keznh0jPgKb8ESXodMX5Op6_4i_fS53XNV2XSo)

* les tests n’ont pas été mis à jour : 4
* les tests ont été mal écrit : 4
* le code ajouté est défectueux  : 13
* le code ajouté marche mais produit des erreurs au sein du système : 23



![Graphe mod&#xE9;lisant le r&#xE9;sultat de l&apos;exp&#xE9;rience 4](https://lh6.googleusercontent.com/k63xbFeUeJfQFOKDVIvu6zex799L9_BQUWks9BW2k6kp7bl-_TkdLdSFCalESEYKrdSlxvOgo_K0WDKgTuB0LoV2MUE03hqm3GOHHosAMVz5AYvRP-gS1uUBJs5Wna4wAJgoaOA)

Dans cette expérience nous avons inspecté 2 commis d'erreur de moins que la précédente expérience avec sonarqube. Malgré tous on remarque que les résultats restent très similaires à la première expérience. Il y a, dans ce projet beaucoup d'erreur qui sont dus à l'interaction du code ajouté qui marche avec le reste du système.

**5. heron raisonnement sur les sections de code**

Heron est un projet avec plus de 2200 pulls requests. heron a une seule version du projet en Jdk8. Nous avons extrait 535 pulls requests sur lesquelles nous avons lancé notre système de détection de cause d'erreur.

![](https://lh6.googleusercontent.com/sXc9RxXjLLZKU0QIUTQGH-sTBYv9EjEJ5pGg4i3Mq_ckFtgOQ7LnUSWJNXsVhYwAz6PuUohDtQdDrWrrvTYtCCyMXHz-LOlqEzgEiLZdAEMKsDSmkjnivNs4oPHX7W2EHD0M-Nk)

* les tests n’ont pas été mis à jour : 62
* les tests ont été mal écrit : 58
* le code ajouté est défectueux : 417
* le code ajouté marche mais produit des erreurs au sein du système : 265



![Graphe mod&#xE9;lisant le r&#xE9;sultat de l&apos;exp&#xE9;rience 5](https://lh5.googleusercontent.com/9TUarI8UtLEIzJUVPraCZdSw6ymo5mCpO4lYnXz6IBRrahXUhS_p8yxdSC-5rpeIGckiUk3zrP1FudEJ7cW4WHy0b8FhTX0dAjYbYcHPy9uH61OH3b-vCOMiUOsINqEcoPg_h88)

Dans ce projet, il est plus courant de trouver des erreurs qui proviennent du code ajouté par le développer. En effet plus de la moitié des erreur vient du code ajouté qui est mal écrit.

## V. Conclusion

Durant ce projet, nous avons eu des résultats très divers et variés. Suivant les projets ces erreurs peuvent majoritairement venir du code qui a été ajouté ou du code déjà présent comme des tests pas mis à jour par exemple. Étant donné la diversité de nos résultats, il n'est pas possible de trancher sur une réponse concrète et définitive.

 En analysant plus de projet, environ 1000, on pourrait potentiellement remarquer une convergence ou pas vers des erreurs de codes ajoutés ou des erreurs de code déjà présent. Cependant avec seulement 5 projets analysés, il n'est pas possible de conjecturer une réponse viable. 

Malgré tous, si ces résultats restent aussi divers, il est possible qu'ils démontrent que les erreurs de merges dépendent de l'équipe qui mène le projet. En effet nos cinq résultats peuvent être aussi différent car les équipes de développer fonctionnent et collaborent différemment au sein des équipes. Ce qui peut expliquer ces divergences, avec dans certaines équipes plus erreur de merge qui sont introduits car les dévellopers se soucis moins de la totalité du système et des interactions qu'il produise lors de changements du code. 

Il serait judicieux d'approfondir les recherches en examinant plus de code et ainsi voir s'il y a une convergence vers telles ou telles métriques : nombre de codes ajoutés défectueux , nombre de codes ajoutés qui marche mais qui s’intègre mal avec le reste du projet, nombre de tests qui ne sont pas mis à jour, nombre de tests mal écrit



## VI. Références

Notre code : [https://github.com/huangshiyang/RIMEL](https://github.com/huangshiyang/RIMEL)

Sonarqube: [https://github.com/SonarSource/sonarqube](https://github.com/SonarSource/sonarqube)

heron: [https://github.com/apache/incubator-heron](https://github.com/apache/incubator-heron
)

RxJava: [https://github.com/ReactiveX/RxJava](https://github.com/ReactiveX/RxJava
)

mockito: [https://github.com/mockito/mockito](https://github.com/mockito/mockito
)

dubbo: [https://github.com/apache/incubator-dubbo](https://github.com/apache/incubator-dubbo
)

shardingshere: [https://github.com/apache/incubator-shardingsphere](https://github.com/apache/incubator-shardingsphere
)

skywaling: [https://github.com/apache/incubator-skywalking](https://github.com/apache/incubator-skywalking
)

