---
layout: default
title:  Les bonnes pratiques lors de l'utilisation de Terraform
date:   2022-01-10 18:25:00 +0100
---

**_janvier 2022_**

## Auteurs

Nous sommes cinq étudiants en dernière année à Polytech Nice Sophia, en mineure Architectures Logicielles :

* João Brilhante &#60;joao.brilhante@etu.unice.fr&#62;
* Enzo Briziarelli &#60;enzo.briziarelli@etu.unice.fr&#62;
* Charly Ducrocq &#60;charly.ducrocq@etu.unice.fr&#62;
* Quentin Larose &#60;quentin.larose@etu.unice.fr&#62;
* Ludovic Marti &#60;ludovic.marti@etu.unice.fr&#62;

## I. Présentation

Les technologies cloud sont en plein essor et sont utilisées dans de nombreux projets réputés. Permettre un déploiement automatique et configurable de ces environnements est indispensable pour ces projets. En ce sens, des technologies, permettant une automatisation du déploiement de ces systèmes sur des infrastructures cloud et d’en configurer plusieurs aspects (ex. cloud provider, …), ont été développées. C’est le cas de [Terraform](https://www.terraform.io/). 
Cependant, n’importe quelle outils demande de respecter de bonne pratiques. Connaître et répandre les bonnes pratiques de l'utilisation de ce genre d’outil permet une meilleure maintenance de ces applications et de corriger de nombreux problèmes, qu’ils s'agissent de problèmes de lisibilité, de sécurité, de performance ou autre. 

## II. Problématique

Cependant, ce n’est pas parce qu’on connait une bonne pratique qu’elle est correctement appliquée. Pour autant qu’on sache, aussi incroyable soit les outils comme Terraform, il pourrait être mal utilisé en pratique. Mais comment vérifie-t-on que si c’est le cas ?  Pour certifier qu’un projet est de bonne qualité, il faut s’assurer qu’il respecte les bonnes pratiques autant que possible. Il faut donc les mesurer. Mais comment les mesurer ? Est-t-il ne serait-ce que possible de les mesurer ?

Ainsi, lors de ce projet nous chercherons à répondre à toute ces question que nous reformulons sous la forme suivante :

**Quelles bonnes pratiques de Terraform sont vérifiables ?**

1. Comment définir une bonne pratique en Terraform ?
2. Quelles pratiques sont encouragées en Terraform ?
3. Est-ce que Terraform est toujours bien utilisé en pratique ?

## III. Hypothèses, Collecte d’informations & Expériences

Notre projet se concentre sur la notion de bonne pratique. Avant toute chose, il nous faut définir ce qu’est une bonne pratique.  Cette définition n’est pas une définition unique, mais c’est à partir de cette définition que va évoluer le projet par la suite. 
Nous définirons une définition en vérifiant les point suivant :
- Une bonne pratique est justifiée.
- Une bonne pratique est non essentielle au fonctionnement de Terraform.
- Une bonne pratique est valide pour les versions récentes de Terraform.
- Une bonne pratique est appuyée par des exemples (optionnel).
- Une bonne pratique est mesurable (optionnel).

A partir de cette définition, nous pouvons rechercher et lister des bonnes pratiques en renseignant une description, une source, une explication, un potentiel exemple et une estimation de mesurabilité. Enfin, nous regroupons ces pratiques à l’aide de tags qui permettent de cerner certaines catégories de pratique (sécurité, lisibilité, …).
Vous trouverez l’ensemble des pratiques répertoriées dans ce tableau.

Maintenant que nous avons des pratiques, il nous faut estimer si elles sont bien utilisées en pratique.

Pour ce faire il nous faut des **projet de référence à analyser**, qui puissent représenter “la pratique” énoncée dans la question. Ensuite, il nous faut des **outils pour analyser des pratique dans un projet*. Enfin, nous pourrons lancer une analyse des dits projet afin d’estimer si telle ou telle bonne pratique est correctement employée.

#### Projets de référence

Avec ces projets, nous devons représenter une norme. Il ne faut pas que ce soit des projets réalisés par un développeur solitaire en autodidacte ou par une classe d'étudiants. 
En ce sens, nous définirons que la sélection des projets sera influencée par la **taille** du projet, sa **popularité**, la taille et l’expérience de l’**équipe** de développement, la régularité des **mises à jour** et la densité de la partie consacrée à **Terraform**.

En ce sens, nous avons sélectionné une petite quantité de code de confiance récolté à la main pour plus de certitude. 

| Nom | Taille | Popularité | Création | Régularité | Contributeurs |
| :-------- | :------: | :---------------: | :-----------: | :-------------: | :-------------------: |
| [StubbornJava](https://github.com/StubbornJava/StubbornJava) | 1214 ko | 218 stars | Dec. 2016| 361 commits | 4 |
| [Atlantis](https://github.com/runatlantis/atlantis) | 43 567 ko | 4.4k stars | Mai 2017| 2014 commits | 206 |
| [Docker Android](https://github.com/budtmo/docker-android) | 242 262 ko | 4.2k stars | Dec. 2016 | 541 commits | 38 |
| [DetectionLab](https://github.com/clong/DetectionLab) | 198 188 ko | 3.3k stars | Dec. 2017 | 541 commits | 6 |
| [StreamAlert](https://github.com/airbnb/streamalert) | 44 411 ko | 2.7k stars | Jan. 2017 | 1900 commits | 30 |

Le problème ici est la quantité de projets. Cependant, il est difficile de rechercher des projets de confiance avec [GitHub](https://github.com/). Peu de filtres, détection de code Terraform difficile et limite d’utilisation, le service a beaucoup de limites empêchant une bonne sélection de projets pertinents.
Il nous faut donc de meilleures méthodes de recherche. Pour cela nous pourrions utiliser des technologies comme [CodeDJ](https://2021.ecoop.org/details/ecoop-2021-ecoop-research-papers/7/CodeDJ-Reproducible-Queries-over-Large-Scale-Software-Repositories) qui offrent un DSL permettant des recherches plus pertinentes.

Cependant, les recherches concernant l’utilisation de CodeDJ n’ont à ce jour pas porté leur fruit, leur base de données étant privée, et reste à l’état de piste de recherche.

#### Outils d’analyse

Pour savoir si Terraform est bien utilisé en pratique, il faut mesurer cela grâce à des outils. De ce fait, nous sélectionnons des outils en procédant par recherches avec Github. Notre recherche  se fait en fonction de plusieurs critères : leur popularité, s’ils sont Open Source ou non et enfin la gratuité de l’outil. 
Au terme de cette recherche nous retenons 6 outils : TFLint, Checkov, TFSec, Terrascan, Regula, Snyk IAC, tous traitant de domaines différents (analyse syntaxique, sécurité, etc.). Cependant, il nous paraît important de vérifier le nombre de règles que vérifient ces outils et le nombre de technologies auxquelles ils s’appliquent : ainsi nous voyons que Checkov est le projet le plus flexible avec un total de 1600 règles traitées et une couverture large des technologies. TFSec est lui aussi assez flexible. En revanche, en appliquant de nouveaux filtres,  nous constatons qu’un bon nombre de projets ont une complexité non négligeable en raison du langage qu’ils utilisent (Rego, Go) et il est donc complexe d’écrire une règle personnalisée dans ces langages. Après quelques essais, nous nous apercevons que mettre en place une règle en Checkov, même avec un langage facile à prendre en main comme Python, reste compliqué. Au mieux, nous finirons par créer une règle qui utiliserait les règle déjà définie par le logiciel et ce n’est pas notre objectif. C’est pourquoi une piste d’exploration peut être d’essayer de mettre en place une règle avec TFLint malgré qu’il faille l’écrire en Go. 

Par la suite, nous allons nous concentrer sur l’utilisation de Checkov.

#### Analyse

Nous pouvons maintenant analyser les projets avec Checkov :

| Projet | Taux de succès |
| :--------- | :-----------------------: |
|  StubbornJava	| 13% 🟥 |
|  Atlantis		| 50% 🟧 |
|  Docker		| 76% 🟨 |
|  DetectionLab	| 48% 🟧 | 
|  StreamAlert		| 94% 🟩 |

Le problème ici c’est que ça nous ne nous donne aucune preuve que cette analyse reflète une bonne ou mauvaise utilisation de bonne pratique. En effet, rien nous dit que les vérifications effectuées par Checkov vérifient une pratique qui respecte notre définition d’une bonne pratique.

La question est : Comment pouvons nous utiliser checkov pour prouver une bonne pratique ?
Dans un premier temps, nous allons sélectionner un petit échantillon de bonne pratique simple afin de vérifier que l’analyse est possible avant de l’étendre au reste de la sélection.

Ainsi, nous nous concentrerons sur les pratique, accès sécurisé, suivante :
- Ne jamais écrire des identifiants en clair dans un fichier de configuration de Terraform.
- Ne jamais partager le fichier .tfstate sur le repository.
- Ne jamais divulguer de secrets en clair dans les sorties d'une configuration Terraform.

Il faut maintenant nous demander comment les vérifier dans un projet.

Pour que l’utilisation de chekov soit pertinent pour vérifier ces pratiques, il nous faut sélectionner des règle définie par la technologie qui sont associable à nos bonne pratique. Ainsi, nous effecturons les vérification suivante : CKV_AWS_41, CKV_AWS_45, CKV_AWS_46, CKV_AWS_58, CKV_AWS_149	, CKV_AZURE_41, CKV_BCW_1, CKV_GIT_4, CKV_LIN_1, 
CKV_OCI_1, CKV_OPENSTACK_1, CKV_PAN_1. Toutes ont été sélectionnées manuellement pour vérifier qu'elles sont bien associées à nos bonnes pratiques.

Ainsi, nous pouvons utiliser la commande suivante sur les projets :
``` 
checkov -d <directory> --compact --framework terraform --check CKV_AWS_41,CKV_AWS_45,CKV_AWS_46,CKV_AWS_58,CKV_AWS_149,CKV_AZURE_41,CKV_BCW_1,CKV_GIT_4,CKV_LIN_1,CKV_OCI_1,CKV_OPENSTACK_1,CKV_PAN_1
```
 
De cette manière, nous obtenons les résultat suivant :

| Projet | Taux de succès |
| :--------- | :-----------------------: |
|  StubbornJava	| 100% 🟩|
|  Atlantis		| 100% 🟩|
|  Docker		| 100% 🟩|
|  DetectionLab	| 100% 🟩|
|  StreamAlert		| 100% 🟩|

Nous observons un total respect des règles en question. De par ce résultat, nous pouvons affirmer que les bonnes pratiques sélectionnées sont en effet bien respectées. Cependant, cette affirmation est fondée sur peu de projets. Afin de confirmer cette affirmation, il est nécessaire de trouver plus de projets pertinents à analyser.

A défaut de pouvoir utiliser CodeDJ, nous retournons sur une recherche GitHub manuelle. Cette fois-ci en recherchant avec des mots clés que l’on peut retrouver dans des fichiers Terraform. Cependant, bien que cela nous permet de connaître des projets ayant des fichiers Terraform, cela ne nous permet pas de faire des filtre sur ces projets. Afin de combler ce manque, nous mettons en place une méthodologie de recherche et des script permettant leur exécution. 

#### Recherche de projet - Méthodologie

Afin d’effectuer une recherche des projets GitHub contenant au moins un fichier Terraform, nous avons tenté une première approche à base de requêtes à l'API de GitHub:

- `extension:tf language:hcl` (2 096 750 fichiers trouvés)
=> Trop de fichiers par projet, le parcours est par conséquent trop long
- `terraform extension:tf` (505 379 fichiers trouvés)
=> Trop de fichiers par projet, le parcours est par conséquent trop long 
=> Certains fichiers invalides indiquent des résultats peu fiables
- terraform extension:tf language:hcl 		(504 803 fichiers trouvés)
=> Trop de fichiers par projet, le parcours est par conséquent trop long 
- `terraform extension:tf` (126 222 fichiers trouvés)
=> Trop de fichiers par projet, le parcours est par conséquent trop long
=> Certains fichiers invalides indiquent des résultats peu fiables
- `required_providers extension:tf language:hcl` (126 219 fichiers trouvés) 
=> Requête OK ✅
Cette dernière requête fonctionne car l’argument `required_providers` n'est déclaré qu'une seule fois par projet Terraform (déclaration des dépendances).
Évidemment, il est toujours possible de détecter plusieurs fichiers répondant à ces critères dans un projet. Cependant, cela semble être grandement limité au vu du nombre de fichiers trouvés lors de chaque requête. 
Malheureusement, l'API de GitHub impose de nombreuses limites d'utilisation. Elle limite notamment le nombre de requêtes de recherche à 30 requêtes/min, le nombre de résultats par page à 100, ainsi que le nombre de résultats accessibles par requête à 1000 (même lorsque l’on utilise la pagination).
Il est donc nécessaire de trouver une stratégie afin de récupérer le maximum de fichiers (et donc de projets associés). Pour cela, nous avons utilisé l'argument size dans la requête qui permet d'indiquer la taille des fichiers (en octets) que nous souhaitons trouver.

Première stratégie : Envoyer une requête par taille de 0 à 203 000 octets, cette grande taille n’étant pas arbitraire mais correspond au plus grand fichier identifié.
Exemple : `required_providers extension:tf language:hcl size:1`,
`required_providers extension:tf language:hcl size:2`, …

Le parcours trop long et fastidieux (203 000 requêtes potentielles).
En effet, nous avons calculé qu'il faudrait 4.7 jours de calcul pour 203 000 / 30 = 6767 secondes = 112 heures = 4.7 jours

Deuxième stratégie : Envoyer une requête par intervalle de taille entre 0 et 203 000 octets, avec un pas de 50 octets.
Exemple : `required_providers extension:tf language:hcl size:0..49`,
`required_providers extension:tf language:hcl size:50..99`, ...

Cette stratégie de requête nous semble déjà plus appropriée, limitant le nombre de requêtes possibles.
Néanmoins, elle nécessite d'identifier un intervalle limité, à la fois calculable en temps et efficient, afin de récupérer le maximum de projets.

Nous allons donc essayer d’affiner cet intervalle, ci-dessous un graphe résultant d’un pas plus petit que précédemment: 10 octets.

Nous pouvons aisément remarquer une zone plus dense que les autres de 60 à 3780 octets. Cet intervalle regroupe 95% des fichiers Terraform trouvés par GitHub. Pour cette raison, nous allons nous concentrer sur cet intervalle avec un pas de 5 octets pour récupérer le maximum de fichiers Terraform (et donc de projets à analyser) en un minimum de temps. Une fois l’analyse terminée, nous constatons que nous avons identifié un total de 92 repositories GitHub contenant au moins un fichier Terraform. 

## V. Résultats et Conclusion

Une fois que nous avons récupéré la liste des repositories, nous utilisons un second script pour analyser chaque repository avec Checkov :

| Nombre de projet | Taux de succès moyen | Projets ayant 100% de succès |
| :--------------------------- | :-----------------------------------: | :----------------------------------------------: |
| 92                            |	88,04%                          | 	80,43%                             |

A première vue, Terraform semble être bien utilisé en pratique. Du moins, cette affirmation est basée sur les bonnes pratiques que nous avons nous mêmes sélectionnées. Afin de généraliser cette affirmation, il serait nécessaire de recommencer une procédure identique, adapté pour l’intégralité des bonnes pratiques, c.a.d:
Sélection de l’outils le plus adapté (Checkov étant le plus adapté pour les pratiques lié à la sécurité par exemple)
Sélection des règles de l'outil associé aux bonnes pratiques retenues (voir en créer si nécessaire).
Lancement de l’outil avec les règles sélectionnées sur les projets précédemment sélectionnés.
Analyse des résultats.

Cela représente un travail long et fastidieux, surtout si l’on prend en compte que certaines bonnes pratiques sont plus accessoires que d’autres: facilité de maintenance auprès des développeurs contre défaut majeur ou faille de sécurité en cas de négligence.

Si l’on se restreint à notre sélection, nous pouvons affirmer au vu des résultats que Terraform est bien utilisé en pratique.

## VI. References

1. ref1
1. ref2
