---
layout: default
title : Etude de comparaison des commits structurés
date:   2023-11
---

**_02 février 2024_**

## Authors

We are four students in M2 or in last year of Polytech’ Nice-Sophia specialized in Software Architecture :

* Hadil AYARI
* Nicolas GUIBLIN
* Chahan MOVSESSIAN
* Floriane PARIS

## I. Research context 

<sub>Préciser ici votre contexte et Pourquoi il est intéressant. **</sub>

* Main question: How and to what extent are structured commits used in open-source projects?
* Interest: Understand the impact of commit standards on the management of software projects (clarity, traceability, maintenance).
* Under questions:
    * How common are conventional commits and gitmojis used in open-source projects?
    * Do employees structure themselves naturally? Or are there commit conventions on each git-hub?
    * Do regular contributors make more structured commits?
    * Is there a collaboration between the number of contributors and commit conventions?


## II. General question

1. Commencez par formuler une question sur quelque chose que vous observez ou constatez ou encore une idée émergente. 
    
2. Préciser pourquoi cette question est intéressante de votre point de vue.

Attention pour répondre à cette question, vous devrez être capable d'émettre des hypothèses vérifiables, de quantifier vos réponses, ...

     :bulb: Cette première étape nécessite beaucoup de réflexion pour définir la bonne question qui permet de diriger la suite de vos travaux.
     
* Conventional commit is part of a standard, why? (pre-analysis observation, “urban legend of development”)
* None of us knew about conventional commit before

## III. Information gathering

Préciser vos zones de recherches en fonction de votre projet, les informations dont vous disposez, ... 

Voici quelques pistes : 

1. les articles ou documents utiles à votre projet 
2. les outils que vous souhaitez utiliser
3. les jeux de données/codes que vous allez utiliser, pourquoi ceux-ci, ...

     :bulb: Cette étape est fortement liée à la suivante. Vous ne pouvez émettre d'hypothèses à vérifier que si vous avez les informations. inversement, vous cherchez à recueillir des informations en fonction de vos hypothèses.

* Sources : Articles, documentation sur les conventions de commits (ex: Conventional Commits, Gitmoji).
* Outils : Analyse de repositories GitHub,  pydriller.
* Données : Collecte de données depuis des repositories open-source sur GitHub.
* Création d’un algorithme afin de déterminer le pourcentage de commits structurés dans un projet
    * Jupyter notebook where we graph the commit patterns among different projects
    * Etude à la main des commits de petits répertoires afin de s’assurer de la justesse de l’outil
* How we chose our data : 
    * Picked big open-source projects that are relatively well-known (stars, watchers, etc..)
    * Picked small projects that were able to be easily analyzed
    * Picked Well known companies that have multiple open-source projects
* [optional] Talk about the projects we’ve decided not to study like Linux and Rust. Reason : Projects are too consuming for our tool which is meant to be lightweight. Huge Commit number, won’t be able to test and debug our tools for them easily. (testing issue). 
For the initial phase, we’ve not studied projects with unclear structures like React (although we’ve decided to study them for a later question)
 
## IV. Hypothesis & Experiences

1. Il s'agit ici d'**énoncer sous forme d'hypothèses** ce que vous allez chercher à démontrer. Vous devez définir vos hypothèses de façon à pouvoir les _mesurer/vérifier facilement._ Bien sûr, votre hypothèse devrait être construite de manière à _vous aider à répondre à votre question initiale_. Explicitez ces différents points.
2. Vous **explicitez les expérimentations que vous allez mener** pour vérifier si vos hypothèses sont vraies ou fausses. Il y a forcément des choix, des limites, explicitez-les.

     :bulb: Structurez cette partie à votre convenance : 
     Par exemples : 
        Pour Hypothèse 1 => 
            Nous ferons les Expériences suivantes pour la démontrer
        Pour Hypothèse 2 => Expériences 
        
        ou Vous présentez l'ensemble des hypothèses puis vous expliquer comment les expériences prévues permettront de démontrer vos hypothèses.
* Hypothèses : Influence des commits structurés sur la maintenance et la compréhension des projets.
    * Hypothesis 1 : Projects would use the already community-driven norm of conventional commits
    * Hypothesis 2 : Top project contributors usually stick to the commits
    * Hypothesis 3 :  The bigger a project is, the more there is the need for commit conventions within that project.
    * Hypothesis 4 : Companies would use the same commit structure among their top projects to facilitate integration between group members
* Expériences : Analyse quantitative et qualitative des pratiques de commits dans divers projets.
    * Finding the large projects was relatively easy, finding projects with specific commit structures was tricky. Especially for Gitmojis, we struggled to find any project that used only gitmojis.
    * Struggled with meta projects, as mentioned previously, Meta’s conventions are not public and unclear. Which led us to avoid them at first, but then thought to revisit them. For this, we’ve had to study their commits by hand and get the structure from our own interpretations (for this, it is sure that the accuracy is much lower) 
    * Initially for hypothesis 4, we planned on testing all the projects of a single company with a single commit structure and see if they use the same conventions for all projects. However, it didn’t make sense to do this for apache, as they have explicitly stated their structure for each project. Instead we decided to test how much they use those conventions.


## V. Result Analysis and Conclusion

1. Présentation des résultats
2. Interprétation/Analyse des résultats en fonction de vos hypothèses
3. Construction d’une conclusion 

     :bulb:  Vos résultats et donc votre analyse sont nécessairement limités. Préciser bien ces limites : par exemple, jeux de données insuffisants, analyse réduite à quelques critères, dépendance aux projets analysés, ...

* Résultats : Présentation et interprétation des données collectées.
Graphs
    * Companies prefer using their own commit structure rather than conventional commits
    * Even when adapted, Companies usually don’t stick to using community driven commits like conventional-commits or gitmoji and usually end up swapping them.
    * The top contributors are usually more diligent with their commit structure, making sure to adhere to the conventions.
    * Companies with multiple projects (Meta & Apache) don’t use the same conventions for all their projects. It seems like each team personalizes their commits to each team’s requirements
    * Comparing Apache with Meta : Apache is comparatively much stricter and much more explicit about their conventions. Apache has no “ultimate” convention, it defines a convention for each project and they’re strict for each convention per project. This doesn’t seem to be the case with Meta. Much more unconventional or “unprofessional” commits found in Meta’s commits [insert figures for meta unconventional commits]

    * Gitmoji jamais utilisé (sauf répertoire gitmoji ou sur une courte période puis abandonné)
    * Conventional commit souvent utilisé MAIS dérivé (rajout de mots-clés…)
    * Commits souvent structurés de manière unique au répertoire
    * Utilisation de commits automatiques selon les projets (merge, pull requests, squash…) qui rend moins nécessaire la structuration des commits car ils gardent la discussion dans la PR des leaders du repo
* Limits : 
    * Reconnaissance des limites de l'étude (échantillonnage, biais possibles) 
    * Our tool was not perfect, in fact, our tool requires a lot of tweaking depending on the project, since every project had its own conventions.
limitations relating to automated commits vs human made commits. 
    * Uncertainties when it comes to non-public commit information (Meta’s documentation doesn’t specify commits structure and claims that all commits are squashed [meaning automated commits], however studying their commits, we can notice a few obviously non-robotic commits)
* Conclusion : Synthèse des découvertes et implications pour les pratiques de développement.
    * Results are inconclusive and are always anecdotal depending on the projects chosen
    * This topic has a lot of nuance (project structure, team organization, developer priorities, outside team communication etc..) and these things we just cannot interpret through just the commit set.


## VI. Tools \(facultatif\)

Précisez votre utilisation des outils ou les développements \(e.g. scripts\) réalisés pour atteindre vos objectifs. Ce chapitre doit viser à \(1\) pouvoir reproduire vos expérimentations, \(2\) partager/expliquer à d'autres l'usage des outils.

![Figure 1: Logo UCA, exemple, vous pouvez l'enlever](assets/images/logo_uca.png){:height="12px"}
Talk about how it works in detail : 
Our tool detects conventions by using regex for each convention message. This is simply because most commit conventions require having specific terms at the beginning of a commit.
Our tool also uses Natural Language Processing to detect some other convention patterns. in fact, many projects have explicitly stated that their convention is simply “have a verb at the beginning of the commit”, some have specified the verb to be in Present form, some have specified in past tense, but most have just specified that it just needs to a verb. 

The way most conventions are tweaked is by modifying the subsystems within the regex as most projects use a variation of that same format.

While we have done our studies on multiple projects, we have kept only the most pertinent results in the notebook provided.


## VI. References

[Debret 2020] Debret, J. (2020) La démarche scientifique : tout ce que vous devez savoir ! Available at: https://www.scribbr.fr/article-scientifique/demarche-scientifique/ (Accessed: 18 November 2022).

* Articles Études pertinentes sur la communication et la qualité des messages de commit. : 
    * Tian, Y., Zhang, Y., Stol, K. J., Jiang, L., & Liu, H. (2022, May). What makes a good commit message?. In Proceedings of the 44th International Conference on Software Engineering (pp. 2389-2401). (https://arxiv.org/pdf/2202.02974.pdf)
* Informations sur les commits structurés 
    * Conventional commits : https://www.conventionalcommits.org/en/v1.0.0/
    * Gitmoji : https://gitmoji.dev/
* Liste projects github open sources choisis pour les statistiques (parmi ceux qui ont le plus de commits/contributeurs)
    * Angular (https://github.com/angular/angular.js) 
    * Nodejs (https://github.com/nodejs/node)
    * fastapi (https://github.com/tiangolo/fastapi)
    * gitmoji (https://github.com/carloscuesta/gitmoji) 
    * React (https://github.com/facebook/react) 
    * flutter (https://github.com/flutter/flutter)
    * Linux (https://github.com/torvalds/linux) 
    * Rust (https://github.com/rust-lang/rust) 
    * Typescript (https://github.com/microsoft/TypeScript)
    * Django (https://github.com/django/django)
    * Nextjs (https://github.com/vercel/next.js) 
    * Cpython (https://github.com/python/cpython)
    * Prometheus (https://github.com/prometheus/prometheus) 
    * Pytorch (https://github.com/pytorch/pytorch) 
    * Git (https://github.com/git/git)

* Liste petits projets testés à la main pour entraîner l’outil : 
    * Request-promised (https://github.com/request/request-promise) -> 300 commits
    * Deployd (https://github.com/deployd/deployd) -> 1300 commits


