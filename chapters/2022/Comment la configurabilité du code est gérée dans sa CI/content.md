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
Dans un contexte où beaucoup de systèmes sont interconnectés sur des plateformes variées, il est primordial de s'assurer de l'intégrité de ces derniers. Il est difficilement imaginable que certains utilisateurs d'un navigateur possèdent une meilleure expérience d'utilisation que les autres, de la même manière que l'ensemble des OS doit être supportés. 

__Intérêt du sujet__
Pour gérer l'ensemble de ces points, il faut disposer d'une chaîne d'intégration et de déploiement continu capable de tester les différentes possibilités existantes (différences fonctionnelles), mais aussi de les déployer de manière automatique et potentiellement conditionnelle.

C'est donc la multitude de perspectives ouvertes par ce sujet qui mène à des cas d'études concrets sur des systèmes à la pointe de leur époque qui le rendent intéressant.


## II. Observations/General question


**Question générale : _Comment la variabilité des différents OS est-elle gérée dans les CI ?_** 


### Décomposition du sujet en sous-questions
- Quel sont les OS et distributions les plus souvent supportées ?
- Comment une CI peut-elle s'assurer qu'un logiciel est conforme sur les différents systèmes d'exploitation qu'il doit supporter ?
- Quelle sont les meilleures méthodes d'intégration continue gérant plusieurs système d'exploitations ?
- Ces méthodes sont-elles flexibles pour rajouter par la suite d'autres facteurs de variabilité ?



## III. Information gathering

### Démarche prévue
- Rechercher des projets pertinents qui ont publié leur chaîne de CI/CD (GitHub Actions, Travis, CircleCI ...) et analyser la gestion de la variabilité (configuration côté CI ou dans le code du projet ?). 
- Analyser les différentes méthodes de gestion de multiples systèmes d'exploitation
- Regarder les différences de configuration dans les builds Linux/Windows/... des CI
- Regarder si il existe des tests avec machine bridée.
- Comparer un projet supporté à la fois par Linux et Windows et chercher les points de variabilité dans la CI et dans le code. Analyser la duplication du code dans chaque cas.

Méthodes de mise en place de CI cross platform
- https://docs.broadcom.com/doc/devops-with-mainframe-driven-applications-implementing-cross-platform-cicd
- https://medium.com/bewizyu/comment-bien-choisir-sa-ci-cd-mobile-en-2019-d4957cda709


## VII. References

[1] J. van Gurp, J. Bosch, and M. Svahnberg. 2001. On the notion of variability in software product lines. In Proceedings Working IEEE/IFIP Conference on Software Architecture, IEEE Comput. Soc, Amsterdam, Netherlands, 45–54. DOI:https://doi.org/10.1109/WICSA.2001.948406
[2] https://treeherder.mozilla.org/jobs?repo=autoland
[3] https://firefox-ci-tc.services.mozilla.com/tasks/index
