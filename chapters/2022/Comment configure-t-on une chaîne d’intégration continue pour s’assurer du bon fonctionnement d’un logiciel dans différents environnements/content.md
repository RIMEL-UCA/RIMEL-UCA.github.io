# Sujet 4 : Extraire des informations sur les systèmes de build (variabilité des builds)

**_janvier 2022_**

## Authors

Nous sommes cinq étudiants en dernière année à Polytech, Nice-Sophia specialisés en Architecture Logicielle :

* Valentin Roccelli <valentin.roccelli@etu.unice.fr>
* Rachid El Adlani <rachid.el-adlani@etu.unice.fr>
* Abdelouhab Belkhiri <abdelouhab.belkhiri@etu.unice.fr>
* Armand Fargeon <armand.fargeon@etu.unice.fr>
* Mohamed Fertala <mohamed.fertala@etu.unice.fr>

## I. Research context /Project

Les projets informatiques prennent de plus en plus d’ampleur au fil des années. Avec cette prise d’ampleur vient généralement une augmentation de la complexité du projet. Cette complexité se retrouve dans le code du projet mais également dans sa maintenance.\
En effet, il faut pouvoir builder et tester de telles applications. Ainsi, afin d’économiser du temps de travail (et de l’argent, évidemment) des outils d’intégration continue ont été développés. Ces outils sont, désormais, très développés et personnalisables, pour s’adapter au mieux aux besoins de leurs utilisateurs.
Le projet a pour but d’explorer et d’étudier la configurabilité de ces outils. Ainsi, nous pourrons fournir, aux personnes curieuses de savoir comment certaines entreprises s’assurent que leur(s) logiciel(s) fonctionne(nt) correctement sur plusieurs plateformes, un document répondant à ces interrogations.

## II. Observations/General question

__Notre question :__ 

Comment configure-t-on une chaîne d’intégration continue pour s’assurer du bon fonctionnement d’un logiciel dans différents environnements ?

__Pourquoi cette question est intéressante à nos yeux :__

Peu importe le niveau d’abstraction que l’on a par rapport à la machine, les applications destinées au grand public ont toujours besoin d’être testées sur plusieurs plateformes afin de garantir une utilisation optimale pour un très grand nombre d’utilisateurs. Pour n’importe quel type de CPU, d’OS ou de version d’un compilateur, l’application doit être buildée et testée. Ces actions doivent être faites de manière automatique afin de ne pas perdre de temps à chaque nouvelle version de l’application, à chaque nouveau merge, à chaque nouveau push.	
Ainsi, nous voulons comprendre comment les sociétés créant de telles applications configurent leur chaîne d’intégration continue pour les builder et les tester sur toutes les plateformes qu’ils ciblent.

__Pour nous aider à essayer de répondre à cette question, nous avons défini 3 sous-questions :__

* Quels éléments doivent être configurables côté pipeline et côté outil d’intégration continue ?
* Comment garantir le bon fonctionnement de l’outil d’intégration continue au fil de l’évolution d’un projet ?
* Pourquoi configurer une nouvelle pipeline pour un nouvel OS ou une nouvelle architecture ?

## III. Information gathering

Nous avons eu 4 grandes sources d’information pendant la durée de ce projet.
1. Les articles ou documents utiles à votre projet
Nous avons 3 grandes catégories de sources d’information dans cette catégorie :
    1. Les articles trouvables en ligne.\
    Ces premiers étaient notre point d’entrée sur le sujet, il s’agissait des premières pistes que nous découvrions au début de notre projet. Ils contenaient des informations mais étaient souvent des tutoriels pour des situations précises. Ils montraient un aspect de la configurabilité mais pas la configurabilité dans son ensemble.
    2. La documentation des chaînes d’intégration continue.\
    Nous avons ensuite orienté nos recherches vers les documentations des différentes chaînes d'intégration continue que nous connaissions et que nous avions découvertes lors de la première étape de nos recherches.\
    Ces documentations contenaient, pour la plupart, les informations que nous recherchions (et même plus encore).\
    Cependant, en raison de la complexité des chaînes d'intégration continue, ces documentations étaient très riches et denses en informations. Nous nous sommes donc penchés, en parallèle, sur d’autres pistes plus claires et/ou visibles.
    3. Les interfaces graphiques des chaînes  d'intégrations continue de projets open-source.\
    Comme dit juste au-dessus, nous cherchions également des informations plus visibles, claires et faciles à extraire (et donc à utiliser).\
    Nous avons donc trouvé les interfaces graphiques de chaînes d’intégration continue de projets open-source. Ces interfaces nous ont permis de voir beaucoup d’informations, dont celles relatives à la  variabilité, facilement.
2. Les outils\
Dans cette catégorie, nous avons utilisés les dépôts GitHub de projets implémentants des chaînes de CI.\
De plus, nous avons cherché des informations directement dans le code source de gros projets open source et des projets que nous avions déjà pu rencontrer lors des autres étapes de recherche.\
Cette source d’information avait comme avantage de nous permettre de voir une/des implémentation(s) concrète(s) de la configurabilité des chaînes de CI. De plus, cela nous a permis de regarder les différents problèmes qui ont pu amener à des modifications de la chaîne d’intégration continue.

## IV. Hypothesis & Experiences

Afin de répondre aux questions posées précédemment, nous allons énoncer 3 hypothèses correspondants à ces dernières :
> A. Les pipelines et les outils d’intégration continue se configurent différemment.

>B. Les outils d’intégration continue fournissent des fonctionnalités pour minimiser les conséquences négatives d’un nombre élevé de _jobs_.

>C. L’ajout de nouvelles pipelines permet de garantir un plus grand nombre d’utilisateurs potentiels.

Pour vérifier la validité de ces hypothèses, nous avons cherché des informations sur plusieurs projets, et nous allons détailler nos recherches et le résultat de ces dernières.

### **Vérification des hypothèses**
#### **A. Les pipelines et les outils d’intégration continue se configurent différemment.**

Pour tester et vérifier cette hypothèse, il suffisait de regarder du côté du code source (configuration de la pipeline au travers d’un fichier _.yml_ par exemple) et du côté de l’outil d’intégration continue (définition au travers d’une interface graphique ou d’une CLI).\
Nous prendrons ici comme exemple Cassandra et Maven, deux gros projets de la fondation Apache qui utilisent Jenkins. L’avantage de ces projets est qu’ils sont open source et que nous avons accès à l’interface graphique de l’outil Jenkins.

Dans les deux projets, nous voyons que les _Jenkinsfiles_ définissent les différentes _steps_ que doivent faire les _agents_. Dans ces fichiers sont également définis sur quelle(s) plateforme(s) ou avec quel(s) framework(s) ces _steps_ doivent être exécutées.\
Il est également possible de définir d’autres paramètres fournis par la chaîne d'intégration continue et qui peuvent se révéler utiles lors de l’exécution.\
Par exemple : la définition d’un _timeout_.\
Cependant, aucune définition des _agents_ n’est faite dans ce fichier ou dans n’importe quel autre fichier du dépôt. Seul un identifiant permettant de savoir à quel type de _worker_ assigner le travail apparaît.

Mais alors où sont définis ces _agents_ ? Nous pouvons les voir “de l’autre côté”, sur l’interface graphique de ces projets. En effet, là-bas, nous pouvons voir une cinquantaine de _workers_ alloués au projet Cassandra et une dizaine alloués au projet Maven.

En explorant la documentation de Jenkins, nous avons découvert l’architecture _master/agent_. C’est une architecture distribuée qui permet de répondre à une montée en charge mais également de définir différents environnements.\
Le _master_ est le serveur principal de Jenkins, qui se charge de la planification et de la répartition des _jobs_, de monitorer les _agents_, d'enregistrer les résultats des _builds_ et autres.\
Les _agents_ sont les _workers_, ce sont des exécutables qui écoutent les directives du _master_.

Mais pourquoi existe-t-il une cinquantaine de  _workers_ pour le projet Cassandra ? Nous allons répondre à cette question en testant et prouvant notre deuxième hypothèse.

#### **B. Les outils d’intégration continue fournissent des fonctionnalités pour minimiser les conséquences négatives d’un nombre élevé de _jobs_.**

S’il existe un aussi grand nombre de _workers_ pour le projet Cassandra, c’est que ce projet est massif et qu’il est devenu nécessaire de maintenir stable plusieurs versions de ce dernier. S’il y avait trop peu de _workers_, les jobs finiraient par attendre dans une queue de plus en plus longue et la vérification du bon fonctionnement de l’application finirait par être trop délayé pour être utile. Ainsi, avoir un grand nombre de workers permet de s’assurer d’en avoir assez dans le cas où toutes les versions du projet ont besoin d’être buildées et testées en même temps.

Mais répondre à cette sous-question et vérifier cette hypothèse ne passe pas seulement par le fait d’avoir un grand nombre de _workers_. Nous avons donc continué de chercher dans les projets et dans la documentation de Jenkins.

Ainsi, nous avons trouvé, dans les recommandations de Jenkins, qu’il fallait éviter de demander aux _masters_ d’effectuer un quelconque _job_ afin qu’ils puissent, à temps plein, s’occuper de l’assignation des _jobs_ aux _workers_.

Si cette recommandation peut paraître logique, d'autres recommandations comme le fait de configurer le niveau de logs des _workers_ ou programmer la suppression de l’historique des _jobs_ en fonction de leurs fréquences ne sont peut-être pas des réflexes que les utilisateurs ont.
De plus, ces recommandations, en plus d’être de bons conseils pour une utilisation optimale de l’outil, permettent de voir l’étendu des fonctionnalités rendues disponibles par les outils d’intégration continue.

Toujours avec l’exemple de Jenkins et des projets de la fondation Apache, nous avons eu la possibilité de voir qu’il est possible de configurer Jenkins pour un ensemble de projets. Il s’agit de la stratégie par *organisme*. En effet, en plus de Maven et de Cassandra, HBase, Hadoop et Beam sont également disponibles dans le même [organisme](https://jenkins-ccos.apache.org/).\
De plus, il est possible d’avoir plusieurs “versions” d’un même organisme. Cela permet d’avoir un environnement pour les QA et un pour les développeurs par exemple et est potentiellement intéressant pour les grands projets où il y a de nombreuses équipes.

Après une exploration de ce regroupement de projets, nous avons pu voir que les machines (_nodes_ dans le vocabulaire Jenkins) sont en fait partagées entre les différents projets. Ceci est fait afin de pouvoir optimiser le nombre total de _nodes_ (puisqu’une machine effectuant un travail coûte de l’argent).

Pour conclure sur cette partie, l’outil Jenkins met, à la disposition de l’utilisateur, un éventail de fonctionnalités afin de minimiser les conséquences négatives d’un grand nombre de _jobs_.\
Nous reviendrons sur ces résultats dans [la partie dédiée à celà](#v-result-analysis-and-conclusion).

#### **C. L’ajout de nouvelles pipelines permet de garantir un plus grand nombre d’utilisateurs potentiels.**

Pour vérifier cette hypothèse, nous avons regardé les chaînes d’intégration continue qui ont des _jobs_ qui buildent et testent des applications sur plusieurs plateformes et/ou frameworks.

Nos recherches nous ont d’abord amené à l’interface graphique de la chaîne d’intégration continue de Spring Boot.\
 Sur cette dernière, nous avons vu qu’ils avaient des jobs pour plusieurs versions du jdk, mais aussi des jobs spécifiques pour Windows.

Nous avons alors cherché, dans leur dépôt GitHub, une raison de l’existence de tous ces _jobs_. Les recherches les plus fructueuses furent sur la raison de l'existence du _job_ testant l’application sur Windows.

En effet, l’existence de ce job est né d’une issue GitHub reportant un [bug](https://github.com/spring-projects/spring-boot/issues/1145) de l’application n’ayant lieu QUE sous le système d’exploitation Windows.\
Il est intéressant de noter que d’autres projets, tels que Conan (un outil de gestion de paquets C/C++) ont rencontré des problèmes avec Windows.

À partir de là, nous avons cherchés les [_issues_ signalant des erreurs n’ayant lieu que sous Windows](https://github.com/search?q=%22fail+on+windows%22&type=Issues), et nous avons trouvés plus de 130,000 _issues_.\
Pour mettre ce chiffre en perspective, [la même recherche avec Linux](https://github.com/search?q=%22fail+on+linux%22&type=Issues) retourne 62,000 _issues_.

Toujours avec le même outil de recherche, nous avons trouvé [30,000 _issues_](https://github.com/search?q=%22Add+Windows+CI%22&type=Issues) ayant comme titre `Add Windows CI`, alors qu’il n’y en a que [6,000](https://github.com/search?q=%22Add+Linux+CI%22&type=Issues) avec le titre `Add Linux CI`.

Ainsi, nous pouvons en déduire que l’ajout de nouveaux jobs peut répondre aux besoins des utilisateurs des applications, qu’ils aient été anticipés ou non par les développeurs.\
_(situation anticipée = nouvelle version avec nouvelle version du framework_\
_situation non-anticipée = changement ne fonctionnant pas sous Windows)_

### **Les outils utilisés**
Comme mentionné précédemment, nous avons utilisé l’API GitHub afin de rechercher des informations dans les dépôts publics permettant de confirmer (ou d’infirmer) nos hypothèses et nous permettant d’avancer dans nos recherches.

En plus de la recherche précédente, nous avons également utilisé cet outil afin de rechercher d’autres éléments relatifs aux chaînes d’intégration continue dans des projets open source. La force de cet outil est que nous avons pu préciser le langage utilisé, le sujet du gist,etc… Cela nous a permis d’avoir un moteur de recherche flexible sur les critères de recherche, donc un moteur de recherche puissant.

Tout au long de notre démarche, nous avons également utilisé GitHub pour avoir accès aux dépôts de projets open source.\
Nous avons sélectionné les projets open source qui nous paraissaient être en adéquation avec le sujet et qui nous permettraient, à terme, d’apporter des éléments de réponses à nos différentes interrogations.

Il y a également d’autres éléments fournis par la plateforme que nous avons pu utiliser dans nos recherches, tels que les branches, qui permettent d’identifier des fonctionnalités, les types de livraison ou encore les tâches.\
Nous nous sommes également servis du _backlog_ de certains projets. Cela s’est révélé crucial dans nos recherches puisque, grâce à ce dernier, nous avons pu identifier manuellement des _issues_ décrivant des problématiques qui nous intéressaient.

### **Justification de nos choix**

Enfin, nous avons principalement parlé des projets de la fondation Apache et de Jenkins car ces projets avaient une taille allant de grand (51 branches et 11,500 commits pour maven) à énorme (303 branches et 26,000 commits pour Hadoop).\
De plus, lors de l’exploration de l’interface graphique de leur Jenkins, nous nous sommes rendus compte de la complexité générale de l’organisation de leur outil d’intégration continue. Cela nous a permis de répondre à notre deuxième sous-question. 

Au fur et à mesure que nous explorions ces projets et leur outil d’intégration continue, nous nous sommes rendu compte que beaucoup des informations que nous trouvions étaient des éléments de réponse pour la première sous-question, d’où son utilisation en tant qu’exemple dans l’hypothèse A ci-dessus.

## V. Result Analysis and Conclusion

Nos résultats sont assez peu nombreux, en raison de la recherche majoritairement manuelle que nous avons effectuée.\
Pour les deux premières hypothèses (et donc les deux premières sous-questions) nous avons principalement utilisé les projets de la fondation Apache, l’interface graphique Jenkins de ces projets  et la documentation de l’outil Jenkins. Les résultats sont peu nombreux mais satisfaisants ! Néanmoins, nous sommes conscients que l’analyse détaillée d’autres projets aurait permis de donner du relief à nos résultats.

Pour la deuxième hypothèse, nous avons vu plusieurs fonctionnalités implémentées dans le les projets de la fondation Apache, mais une majorité au travers de la documentation. Trouver une implémentation de toutes ces fonctionnalités aurait pu nous donner un retour des potentielles complications que les développeurs auraient pu avoir avec ces fonctionnalités. Comme pour la première sous-question, plus de résultats auraient permis de donner du relief à nos résultats.

Cependant, nos recherches furent longues et méticuleuses, et nous manquions simplement de temps pour fournir le même travail sur un autre projet implémentant un autre outil d’intégration continue.

Ensuite, la troisième hypothèse et sa sous-question correspondante.\
En raison des quelques métriques que nous avons pu dégager, nous pouvons déduire qu’une plus grande partie des projets n’incluent pas de _pipeline_ Windows, et que les tests sur cette plateforme sont moins nombreux.

Il est important de souligner que, malgré l’utilisation d’un outil ou d’un framework qui devrait rendre le code indépendant de l’OS, il reste des différences colossales entre les différents OS, tels que l’arborescence de fichiers ou la gestion des permissions. Un autre problème que nous avons vu lors de nos recherches est un ancien problème avec WSL, où un _long double_ était encodé sur 64 bits, alors que sur Linux, il est encodé sur 80 bits.

Ce que cette sous-question nous a appris est qu’il est important de bien tester les applications que l’on développe, ou, au moins, de vérifier que les outils / bibliothèques / frameworks utilisés le sont.

Finalement, toutes ces questions et ces recherches nous ont permis d’en savoir plus sur les outils d’intégration continue et nous ont permis d’en savoir plus sur leur configurabilité.

Il reste encore beaucoup de sous-questions découlant de notre question principale qui restent sans réponses, mais nous sommes fiers de notre travail et de ce que nous avons produit.

## VI. Tools

Nous avons déjà mentionné l’utilisation de l’API GitHub de recherche lors des parties précédentes. Vous trouverez ici deux exemples de nos recherches.

https://api.github.com/search/repositories?q=topic:JenkinsFile+language:*
https://github.com/search?q=%22fail+on+windows%22&type=Issues

## VII. References

Détails de la plupart des sources (certaines n’ayant pas été notées sur le coup et n’ayant pas été retrouvées) utilisées lors de nos recherches :
1. Les articles trouvables en ligne
   * [Getting started with Jenkins Configuration as Code](https://www.eficode.com/blog/start-jenkins-config-as-code)
   * [Editors for Pipeline and Task yml's](https://concoursetutorial.com/miscellaneous/yaml-editors.html#editors-for-pipeline-and-task-ymls)
   * [Exploring ARM Templates: Azure Resource Manager Tutorial](https://www.varonis.com/blog/arm-template)
2. La documentation des chaînes d’intégration continue
   * [Apache’s Confluence](https://cwiki.apache.org/confluence/display/INFRA/Jenkins)
   * [Jenkins’ wiki](https://wiki.jenkins-ci.org/JENKINS/)
   * [Jenkins’ user documentation](https://www.jenkins.io/doc/)
   * [Gitlab Docs](https://docs.gitlab.com/ee/ci/)
   * [Microsoft’s Azure DevOps Docs](https://docs.microsoft.com/en-us/azure/devops/pipelines/?view=azure-devops)
   * [CloudBees’ Doc](https://docs.cloudbees.com/docs/admin-resources/latest/)
3. Les interfaces graphiques des chaînes de CI de projets open-source
   * [Apache’s buildbot](https://ci2.apache.org/#/)
   * [Apache’s Jenkins](https://jenkins-ccos.apache.org/)
   * [FirefoxCI](https://firefox-ci-tc.services.mozilla.com/)
   * [Spring’s Concourse](https://ci.spring.io/)
   * [Spring’s Gradle](https://ge.spring.io/)
4. Les dépôts GitHub de projets implémentants des chaînes de CI
   * [Spring Boot](https://github.com/spring-projects/spring-boot)
   * [Code coverage for Ruby](https://github.com/simplecov-ruby/simplecov)
   * [React Native](https://github.com/facebook/react-native)
   * [Spring Initializr](https://github.com/spring-io/initializr)
   * [Helm](https://github.com/helm/helm)
   * [Conan](https://github.com/conan-io/conan)
   * [VSCode](https://github.com/microsoft/vscode)
   * [Azure CLI](https://github.com/azure/azure-cli)
   * [Recherche GitHub sur les issues ouvertes en raison du manque d’un build Windows](https://github.com/search?q=%22fail+on+windows%22&type=Issues)
