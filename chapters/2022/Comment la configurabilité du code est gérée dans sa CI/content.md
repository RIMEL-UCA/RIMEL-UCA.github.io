 # Sujet 4 : Extraire des informations sur les systèmes de build (variabilité des builds)

**_janvier 2022_**

## Auteurs

Nous sommes cinq étudiants en dernière année à Polytech' Nice-Sophia specialisés en Architecture Logicielle :

* Barna Anthony &lt;anthony.barna@etu.univ-cotedazur.fr&gt;
* Burette Léo &lt;leo.burette@etu.univ-cotedazur.fr&gt;
* Defendini Lara &lt;lara.defendini@etu.univ-cotedazur.fr&gt;
* Savornin Guillaume &lt;guillaume.savornin@etu.univ-cotedazur.fr&gt;
* van der Tuijn Anton &lt;anton.van-der-tuijn@etu.univ-cotedazur.fr&gt;

## I. Contexte de recherche

__Contexte__
Dans un contexte où beaucoup de systèmes sont interconnectés sur des plateformes variées, il est primordial de s'assurer de l'intégrité de ces derniers. Il est difficilement imaginable que certains utilisateurs d'un navigateur possèdent une meilleure expérience d'utilisation que les autres, de la même manière que l'ensemble des OS doit être supportés. Enfin, beaucoup de systèmes existent désormais et le même site web doit pouvoir être accessible de manière réactive et ergonomique sur différentes tailles d'écran (ordinateur, téléphone, tablette etc...).

__Intérêt du sujet__
Pour gérer l'ensemble de ces points, il faut disposer d'une chaîne d'intégration et de déploiement continu capable de tester les différentes possibilités existantes (différences fonctionnelles), mais aussi de les déployer de manière automatique et potentiellement conditionnelle.

C'est donc la multitude de perspectives ouvertes par ce sujet qui mène à des cas d'études concrets sur des systèmes à la pointe de leur époque qui le rendent intéressant.


## II. Observations/General question

<!-- **Question générale :** "Comment les chaînes de CI peuvent être configurées pour gérer les différences fonctionnelles ?" -->

**Question générale : _Comment la configurabilité du code est-elle gérée dans sa CI ?_** _On se focalisera notamment sur les platformes matérielles (OS, distributions ...)_

### Décomposition du sujet en sous-questions

- Quelles sont les différentes configurations matérielles qu'une chaîne de CI doit gérer ?
- Quelles sont les différentes configurations logicielles qu'une chaîne de CI doit gérer ?
- La variabilité d'un projet est-elle gérée dans le code source ou alors existe-il une version par variable ?
- Les CI peuvent-elles gérer la variabilité automatiquement ? 
- Est ce qu'il est compliqué, une fois la CI en place, de rajouter de nouveaux facteurs de variabilités ?
- Comment mesurer l'adaptabilité de la CI aux différentes configurations du code ?
- Comment bien définir les KPIs permettant de mesurer et gérer ces différences ?
- Quel est le niveau d'automatisation fourni par les chaînes de CI pour la gestion de la variabilité ?

<!-- - Quels sont les différences fonctionnelles qu'une chaîne de CI doit gérer ?
- Quels sont les différentes méthodes utilisées à ce jour pour gérer ces différences fonctionnelles ?
- Comment bien définir les KPIs permettant de mesurer et gérer ces différences ?
- Quelle sont les méthodes qui offrent les meilleurs KPIs ?
- Est-ce que les CI peuvent gérer la variabilité automatiquement ?
- Quel est le niveau d'automatisation fourni par les chaînes de CI pour la gestion de la variabilité ?
- Est ce que les variables d'environement définis dans la CI permettent de déduire la configuration du système ? (Question 3 général)

Pour chaque implémentation différente de la variabilité:
- Quel est le niveau de difficulté pour étendre la variabilité d'une CI existante ?
 -->

### Démarche prévue
- Rechercher des repositories GitHub qui ont une CI publique (GitHub Actions, Travis ...) et analyser la gestion de la variabilité (configuration côté CI ou dans le code du projet ?). 
- Trouver des exemples de tests UX/UI pour voir comment est gérée la variabilité des tailles d'écran/résolutions.
- Comparer les tests réalisés par les développeurs et ceux disponibles dans des frameworks, et s'intéresser notamment à leur taux d'utilisation.
- Regarder si il existe des tests avec machine bridée.
- Comparer un projet supporté à la fois par Linux et Windows et chercher les points de variabilité dans la CI et dans le code. Analyser la duplication du code dans chaque cas.

## III. information gathering

Méthodes de mise en place de CI cross platform
- https://docs.broadcom.com/doc/devops-with-mainframe-driven-applications-implementing-cross-platform-cicd
- https://medium.com/bewizyu/comment-bien-choisir-sa-ci-cd-mobile-en-2019-d4957cda709


## VII. References

[1] J. van Gurp, J. Bosch, and M. Svahnberg. 2001. On the notion of variability in software product lines. In Proceedings Working IEEE/IFIP Conference on Software Architecture, IEEE Comput. Soc, Amsterdam, Netherlands, 45–54. DOI:https://doi.org/10.1109/WICSA.2001.948406
[2] https://treeherder.mozilla.org/jobs?repo=autoland
[3] https://firefox-ci-tc.services.mozilla.com/tasks/index

