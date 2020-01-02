# Impact des contributeurs minoritaires sur la qualité du code des projets open-source

## Authors

We are four students in last year of Polytech' Nice-Sophia specialized in Software Architecture :

* Enzo Dalla-Nora &lt;enzo.dalla-nora@etu.unice.fr&gt;
* Florian Lehmann &lt;florian.lehmann@etu.unice.fr&gt;
* Tanguy Invernizzi &lt;tanguy.invernizzi@etu.unice.fr&gt;
* Alexandre Clement &lt;alexandre.clement@etu.unice.fr&gt;

## I. Contexte de recherche

Les logiciels open source trouvent leurs origines dans les années 1970 et 1980 \[5\]. Depuis le début, le mouvement social derrière le logiciel libre soutient que les utilisateurs de logiciels devraient avoir la liberté de voir, modifier, mettre à jour, corriger et ajouter du code source pour répondre à leurs besoins et être autorisés à le distribuer ou le partager librement avec d'autres. Cette philosophie a amené une grande diversité de personnes à participer à ces projets. Du noyau Linux aux différentes bibliothèques front-end JavaScript, n'importe qui peut contribuer et apporter des modifications aux outils dont il se sert chaque jour, à condition bien sûr que sa contribution soit acceptée.

On est donc en droit de se demander quels facteurs influent sur la qualité d'une contribution. On sait que les facteurs humains jouent un rôle important \[3\]. Des travaux ont déjà exploré l'influence de la paternité de code avec la qualité d'une contribution avec des résultats mitigés : alors que certains auteurs trouvent des relations entre les défaillances et la paternité de code, d'autres affichent des résultats plus nuancés \[2\]. Pour cette étude, nous allons donc nous intéresser au lien entre la paternité d'un projet et la qualité de code apportée par les contributeurs.

## II. Observations

1. Le but cherché est de déterminer si, sur un projet open-source, le **code ajouté** par des **contributeurs minoritaires** a un **impact** sur ce projet en termes de **qualité de code**. L'objectif principal avec notre étude est de mettre en lumière les problématiques que peuvent apporter les différents profils de contributeurs sur un projet open-source au fil de leurs participations.
2.   Il est intéressant de noter que, contrairement à certains aspects entraînant une mauvaise qualité de code tel que la complexité de la dépendance ou la taille, la propriété peut être modifiée de façon délibérée en modifiant les processus et les règles. \[1\] Si la propriété a un effet positif, les stratégies visant à appliquer une propriété forte du code peuvent être mises en place. Les gestionnaires peuvent également faire attention aux codes fournis par les développeurs dont l'expérience pertinente en amont est inadéquate.

   Afin de répondre à cette problématique, on vient mettre en avant 2 angles différents pour apporter des éléments et des conclusions plus complètes.

   **Est-ce que le code d'un contributeur minoritaire vient baisser la qualité logicielle du projet collaboratif ?** On se place ici dans un aspect "métriques", en souhaitant connaître si un delta de qualité sur un projet est visible avant et après un ajout d'un contributeur minoritaire.

3.   **Est-ce que le code d’un contributeur minoritaire a une pérennité dans le temps moins importante qu’un contributeur majoritaire ?** Le but ici est d’apporter une vision sur le devenir du code ajouté par les contributeurs minoritaires et de constater si une différence est visible sur un plan temporel, en considérant qu’un code “mauvais” sera moins durable dans le temps puisque rapidement remplacé à cause des problèmes qu’il peut engendrer.

   En répondant à ces questions, on sera capable d'aborder deux aspects différents qui seront capables d'attester de la qualité d'un code à leurs manières, et d'apporter une réponse à notre étude.

## III. Collecte d'informations

Les métriques de la propriété de code ont été récemment définies afin de distinguer les contributeurs majeurs et mineurs d'un module logiciel, et de déterminer si la propriété d'un tel module est forte ou partagée entre les développeurs. Cette relation a d'abord été étudiée et validée sur des projets logiciels propriétaires réalisés par Bird et al. \[1\]. Une autre étude portant sur des logiciels open-source réalisée par Foucault et al. \[2\] a suivie mais les résultats ne permettent pas de conclure sur une quelconque corrélation.

Cependant, nous remarquons que Foucault et al. ont utilisés le nombre de lignes de code \(LOC\) et le nombre de méthodes pondérées \(WMC\) comme mesurer la qualité des logiciels. Nous avons donc décidé de reproduire l'étude réalisée par Foucault et al. sur des logiciels open-source mais en utilisant le logiciel SonarQube pour analyser la qualité des logiciels. SonarQube permet d'analyser des projets avec de nombreuses métriques comme le nombre de défauts de code \(code smells\), les duplications de code, les vulnérabilités, etc. De plus, contrairement à l'étude originale de Bird et al. qui n'analyse que des fichiers dll et l'étude de Foucault et al. qui n'analyse que des fichiers Java, SonarQube nous permet d'analyser des projets en Java, mais aussi en JavaScript et Python.

Enfin, les deux études précédemment réalisées portent toutes deux sur des projets sélectionnés manuellement. Pour éviter de biaiser dans notre étude par le choix des projets, nous avons décidé d'utiliser un jeu de donnée issue de Kaggle pour la sélection des logiciels que nous allons analyser.

#### Lexique

**Contributeur minoritaire** : Contributeur sur un projet ayant moins de 5% des commits de celui-ci. \[1\]

**Proportion de propriétés** : La proportion de propriétés \(ou simplement la propriété\) d'un contributeur pour un composant particulier est le rapport du nombre de commit que le contributeur a effectué par rapport au nombre total de commit pour ce composant.

**Modification** : Dans un commit, on considera que le nombre de modifications correspond au delta entre le nombre de lignes ajoutées et le nombre de lignes supprimées.

Pour arriver à extraire les informations intéressantes, nous avons procédé en deux étapes :

**A. Sélection des projets**

Pour accélérer cette récolte d'informations, nous sommes partis, comme expliquer précédemment, de Kaggle. Le jeu de données utilisées regroupant plus de 2 millions de projets open-source, nous avons garanti une sélection aléatoire et diversifiée et donc nous éliminons un biais quant au choix des repos analysés.

Nous avons fait le choix de partir d'une base de données Kaggle contenant 2.8 millions de projets _github_.

Le 1er filtre appliqué vient sur les langages majoritaires des repos. Nous avons cherché à sélectionner les langages ayant des utilisations très différentes et populaires. On a tout d'abord _JavaScript_, principalement utilisé dans les projets "web", Python qui est porté plus sur un aspect "script" et Java. De cette façon, on s'astreint d'un nouveau biais qui aurait pu venir avec le profil "métier" des contributeurs et pourquoi pas argumenter sur ces derniers pour enrichir la précision de nos résultats. Nous avons obtenu près d'un million de dépôts correspondants.

Nous avons ensuite effectué une réduction à 5000 repos aléatoires parmi ce résultat. Cette réduction ne viendra que peu impacter les résultats étant donné le nombre très important de repos sélectionnés, tout en nous permettant de ne pas être contraints par l'API _github_ \(5000 appels/h\) que nous utilisons pour récupérer les informations traitées juste après.

Enfin, deux autres filtres ont été mis en place :

* Le dépôt doit contenir au moins un contributeur qui a moins de 5 pourcents des commits puisque notre étude se focalise sur l’impact des contributions d’auteur minoritaire. Il est donc indispensable qu'un dépôt contienne au moins un de ses contributeurs qui soit minoritaire afin de pouvoir le prendre en compte.
* Java, Python ou JavaScript doit représenter au minimum 51% du code dans le projet. Cela nous permet ainsi d’éliminer tous les projets qui contiennent l’un des 3 langages, mais dont la quantité de code reste trop faible pour être considérée comme majoritaire. Il peut notamment s’agir de code utilisé pour la réalisation de scripts et dont la vocation principale n’est pas d’implémenter le coeur du projet.

Au final, nous arrivons à 1356 dépôts contenant un langage majoritaire analysable par SonarQube et contenant au moins un contributeur minoritaire.

**B. Extraction des données**

Une fois nos données obtenues, nous avons procédé à l'extraction de diverses métriques, afin de pouvoir porter notre analyse sur ces dernières.

**a. Extraction de la qualité logicielle**

Le but de notre premier élément étant de se concentrer sur cette qualité logicielle, il nous a fallu choisir un logiciel qui répond à ces critères :

* Être automatisable pour éviter d'intervenir dans l'analyse des 1356 repos.
* Executable sur des projets contenant les 3 langages choisis \(Java/JavaScript/Python\) pour cette étude.
* Capable d'extraire des métriques standards afin de ne pas ajouter un biais par rapport à si nous en avions créer de toutes pièces.
* Fournissant un moyen simple de récupérer les résultats, toujours dans l'optique d'automatiser les analyses.

SonarQube répondant à nos exigences, nous avons choisi ce logiciel pour assurer l'analyse des projets.

**b. Métriques SonarQube**

Nous allons récupérer le delta des métriques entre chaque commit en analysant l'ensemble du projet. En effet, nous serons ainsi capables d'avoir une idée de l'impact global d'un changement étant donné que des modifications restent rarement sans conséquence sur les fichiers autres que ceux modifiés.

Nous avons choisi de conserver 4 métriques principales :

* Nombre de bug
* Nombre de vulnérabilités
* Code smell
* Nombre de ligne dupliquée

Ces métriques standards reflétant différents pans de la qualité logicielle, nous les avons choisis pour asseoir notre étude.

## IV. Hypothèses et expériences

Maintenant que nous connaissons les informations que nous allons collecter, il est important de décrire quels protocoles vont être mis en place pour répondre à nos questionnements.

**Est-ce que le code d'un contributeur minoritaire vient baisser la qualité logicielle du projet collaboratif ?**

La première hypothèse que l'on peut émettre serait qu'un contributeur minoritaire va, en moyenne, introduire une qualité de code plus mauvaise étant donné qu'il n'est pas aussi "conscient" de l'architecture et du projet de façon plus globale.

Nous avons donc mis en place un protocole afin de vérifier cette hypothèse. Il s'agit dans un premier temps de prendre un dépôt, de distinguer les profils de chaque contributeur en fonction du nombre de commit sur ce projet \(&lt;5% pour un minoritaire\). Ensuite, il suffit pour chaque commit de lancer une analyse SonarQube avant et après l'avoir joué et d'associer le delta des métriques au groupe de contributeur correspondant.

Pour chaque groupe, nous aurons ainsi le total de nos 4 métriques SonarQube à exploiter sur un projet donné et nous pourrons nous appuyer dessus afin d'en extraire des conclusions quant à cette première hypothèse.

Il est important de noter qu’ici, nous allons pondérer ces métriques par le nombre de lignes modifié \(lignes ajoutées - lignes supprimées\) totales de chaque groupe. Nous avons simplement regardé pour chaque commit le nombre de lignes modifiées et nous avons associé ce montant au groupe associé au contributeur. Cette pondération est importante pour apporter une base comparable entre les 2 groupes, sans lesquels les données seraient erronées.

**Est ce que le code d'un contributeur minoritaire a une pérénnité dans le temps moins importante qu'un contributeur majoritaire ?**

Ici, nous nous intéressons à un aspect plus temporel par rapport à la durée de vie d'une contribution. Si notre première hypothèse se confirme, on est alors en droit de se demander si un contributeur minoritaire apporte un code moins durable dans le temps étant donné son aspect "qualité de code" peu intéressant et plus enclin aux dysfonctionnements.

Un protocole très similaire a été mis en place, en regardant cette fois-ci non plus une analyse SonarQube pour chaque commit mais le delta de temps entre la dernière modification des lignes de code touchées par le commit et la date de celui-ci. Ainsi, on sera capable de connaître la durée de vie d'une ligne. Il suffira d'associer ce delta au groupe du contributeur et de diviser par le nombre de lignes modifié par chaque groupe afin obtenir la durée de vie moyenne d'une ligne de code pour chaque groupe.

Nous avions la possibilité de regarder la pérénnité en terme de commit, c'est a dire le nombre de commit effectués sur le projet entre chaque modification d'une ligne, mais une mesure temporelle nous semblait plus juste puisque les commits peuvent être séparés de plusieurs mois sur certains projets, rendant les mesures trop peu consistantes.

## V. Analyse des résultats

Dans le cadre de cette étude, nous n'avons pas pu effectuer l'analyse sur un grand nombre de dépôts Github. Cette dernière a été limitée par le temps d'analyse qui pouvait être très important. En effet, il nous a fallu plus de 36 heures pour analyser chaque version de 8 dépôts Github. Ainsi, il n'était pas possible de réaliser dans le temps alloué une analyse sur plus d'une centaine de dépôts comme nous l'avons initialement souhaité.

**Est-ce que le code d'un contributeur minoritaire vient baisser la qualité logicielle du projet collaboratif ?**

Après avoir obtenu nos résultats, nous arrivons aux données suivantes :

|  | **Bug** | \*\*\*\* | **Vulnérabilités** | \*\*\*\* | **Code Smell** |  | **Duplications** |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  | Minoritaires | Majoritaires | Minoritaires | Majoritaires | Minoritaires | Majoritaires | Minoritaires | Majoritaires |
| OpenFeign/feign | 43 | 53 | 9 | 16 | 2462 | 3007 | 4739 | 5079 |
| aaugustin/websockets | 0 | 13 | 0 | 0 | 0 | 70 | 0 | 82 |
| burnash/gspread | 4 | 2 | 0 | 0 | 29 | 40 | 0 | 1 |
| jgraph/drawio | 2 | 237 | 0 | 2034 | 0 | 9286 | 0 | 105367 |
| mkorpela/pabot | 14 | 48 | 0 | 1 | 4 | 27 | 0 | 2 |
| reactiveops/pentagon | 0 | 11 | 3 | 6 | 42 | 48 | 2522 | 1949 |
| recipy/recipy | 7 | 47 | 1 | 21 | 42 | 294 | 36 | 300 |
| spring-cloud/spring-cloud-consul | 21 | 62 | 30 | 79 | 1192 | 3319 | 2342 | 4926 |

**A. Bug**

![](../.gitbook/assets/git04.png)

Pour ce qui est des bugs, nous pouvons constater sur les 8 projets analysés, 75% contiennent un ratio bug/ligne amplement supérieur pour les commits des contributeurs minoritaires par rapport aux majoritaires.

**B. Vulnérabilités**

![](../.gitbook/assets/git03.png)

Au niveau des vulnérabilités introduites, on reste sur un constat plus négligé, avec une répartition égale entre les groupes de contributeurs. Cependant, on note une différence très nette, dans un sens ou dans l'autre entre nos groupes de contributeurs sur cette métrique.

**C. Code smell**

![](../.gitbook/assets/git02.png)

Pour ce qui est du code smell par ligne de code, on repasse sur des contributions négatives effectuées en premier lieu \(75%\) par les contributeurs minoritaires. Il est intéressant de constater que la différence entre les 2 groupes est très forte si le ratio est dominé par les contributeurs minoritaires alors qu'à l'inverse, elle est presque négligeable si ce sont les contributeurs majoritaires qui apportent le moins bon ratio.

Cela met en avant une rigueur moins forte des contributeurs minoritaires sur ce type de projet ou du moins, des pratiques moins bonnes dans le code \(celles prônées par SonarQube\). On peut assimiler cela à une négligence de la part des contributeurs minoritaires qui ne prennent pas le temps de rendre un code de meilleure qualité, ou bien à une meilleure d'expérience des contributeurs majoritaires ayant connaissance des bonnes pratiques \(beaucoup de contributions dans un projet...\).

**D. Code dupliqué**

![](../.gitbook/assets/git01.png)

Enfin, cette dernière métrique apporte des constatations très similaires avec pour un projet près d'une ligne sur 4 qui est dupliquée par les contributeurs minoritaires.

Ces observations apportent une dimension supplémentaire sur notre analyse et mettent en avant notamment des contributeurs minoritaires enclins à dupliquer du code déjà présent dans le projet. Cela met en lumière une possible maîtrise du projet moins importante par rapport aux contributeurs majoritaires, ce qui reste cohérent avec ce profil.

Si l'on prend du recul sur cette analyse, on constate globalement que les projets ont de fortes disparités entre les symptômes. On peut y voir de grandes différences dans le profil des projets étudiés, avec des distinctions entre les exigences de chaque projet vis-à-vis des contributions extérieures. Seul le 1er présente une proportion équivalente entre toutes les métriques mise en avant. On constate également que le 2e projet marque une très bonne cohérence avec une forte qualité des métriques. En se penchant sur ce projet, on constate qu’un seul contributeur majoritaire le gère, et nous pouvons donc penser que ses exigences pour les contributions ne changent pas d'une proposition à une autre. Nous pouvons soulever ici un nouveau point d'analyse en nous basant sur le profil hiérarchique des projets, en prenant en compte le nombre de contributeurs dans chaque groupe afin de pousser encore plus loin notre réflexion.

#### Critiques

Nous présentons ici les limites de notre étude et comment les contourner dans de futures recherches. Tout d'abord, notre jeu de données est très limité, en partie dû à des contraintes techniques et de temps. Avec seulement huit projets, il est dangereux de tirer des conclusions plus générales sur une relation entre propriété et qualité de code. Par ailleurs, notre étude se retrouve amputée d'une recherche plus approfondie sur l'impact de la propriété en fonction du langage de programmation utilisé.

Ensuite, l'utilisation de SonarQube peut être critiquée. Bien que les métriques de SonarQube soient globalement reconnues par l'industrie, celles-ci ne sont pas universelles et il n'est pas rare que des projets suivent des lignes de conduite qui divergent des normes de SonarQube. De ce fait, les contributions apportées à ce type de projets peuvent être détectées comme impactant négativement la qualité du code par SonarQube alors qu'elles suivent rigoureusement les contraintes de qualité imposées par l'équipe de développement.

Enfin, il est important de remarquer qu'un des aspects portant sur la pérennité du code des contritubuteurs minoritaires n'a pas été abordé faute de temps. Le travail sur la pérennité du code aurait pu, entre autres, enlever le doute sur la validité de SonarQube pour notre étude. On peut en effet supposer que si un code est jugé de mauvaise qualité par les autres développeurs, alors celui-ci sera modifié, remplacé ou supprimé. Ainsi, la pérennité nous donne un indice fiable de la qualité de code. Cet indice manque cependant de précision sur les parties du code impacté par ces changements de qualité. C'est pourquoi nous envisagions de coupler l'analyse de la pérennité avec l'analyse de SonarQube afin d'obtenir une estimation de la qualité du code à la fois fiable et précise.

## VI. Conclusion

Grâce à nos analyses, nous pouvons conclure qu'il y a bien un lien entre la qualité du code et le profil du contributeur associé à ce code. De manière générale, un contributeur minoritaire va proposer du code de moins bonne qualité que ses homologues majoritaires.

Il reste important de cependant nuancer nos résultats étant donnés la très faible quantité de projets analysés. En effet, le temps a joué contre nous avec des analyses très longues portant sur l'intégralité des commits, ce qui nous a pénalisés dans l'extraction des données. On peut cependant mettre en avant un problème que nous avons constaté avec nos résultats par rapport aux profils des contributeurs sur les projets open-sources.

Ainsi, il est intéressant par la suite de se pencher sur des données plus conséquentes et de creuser dans cette voie afin de constater ou non nos observations de façon plus générale. Nous pouvons également mettre en avant un aspect temporel dans les futures analyses avec la pérennité d'un code selon les profils. On peut également s'intéresser aux profils des projets en eux-mêmes en regardant par exemple leur hiérarchie afin de voir les impacts de cette dernière sur la qualité des différents profils des contributeurs.

## VII. Outils

Nos outils et datasets sont entièrement disponibles à [cette adresse](https://github.com/FlorianLehmann/rimel).

Pour reproduire cette expérimentation, il faut suivre les explications de chaque README inclus dans le repository GitHub. Celui-ci contient 4 dossiers qui correspondent aux 4 étapes de notre expérience :

* La récupération des données initiales dans le dossier "kaggle"
* La selection des différents repositories dans le dossier "select\_repositories"
* Le clonage du contenu de chaque repositories dans le dossier "clone\_repositories" 
* Et enfin, l'analyse avec sonarqube dans le dossier "sonarqube\_analysis".

## VIII. References

1. Christian Bird , Nachiappan Nagappan , Brendan Murphy , Harald Gall , Premkumar Devanbu, Don't touch my code!: examining the effects of ownership on software quality, Proceedings of the 19th ACM SIGSOFT symposium and the 13th European conference on Foundations of software engineering, September 05-09, 2011, Szeged, Hungary
2. Matthieu Foucault , Jean-Rémy Falleri , Xavier Blanc, Code ownership in open-source software, Proceedings of the 18th International Conference on Evaluation and Assessment in Software Engineering, May 13-14, 2014, London, England, United Kingdom
3. N. Nagappan, B. Murphy, and V. Basili. The influence of organizational structure on software quality: an empirical case study. In Proc. of the 30th international conference on Software engineering, 2008.
4. C. Bird, N. Nagappan, P. Devanbu, H. Gall, and B. Murphy. Does distributed development affect software quality? an empirical case study of windows vista. In Proc. of the International Conference on Software Engineering, 2009
5. A Wheeler, David. \(2001\). Why Open Source Software/Free Software \(OSS/FS\)? Look at the Numbers!.

![](../.gitbook/assets/logo_uns%20%285%29.png) UCA : University Côte d'Azur \(french Riviera University\)

