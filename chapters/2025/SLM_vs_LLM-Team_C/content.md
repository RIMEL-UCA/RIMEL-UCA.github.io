---
layout: default
title : SLM vs LLM - Team C
date:   2024-11
---

```
---
**Date de rendu finale : fin février**
- Respecter la structure pour que les chapitres soient bien indépendants
- Remarques :
    - La structure proposée est là pour vous aider, mais peut-être adaptée à votre projet
    - Les titres peuvent être modifiés pour être en adéquation avec votre étude.
    - Utiliser des références pour justifier votre argumentaire, vos choix, etc.
    - Vous avez le choix d'utiliser le français ou l'anglais.

    Dans l'article de Blog [Debret 2020], l'auteure donne les éléments principaux de la démarche d'une manière simple et très facile à lire, dans la partie [Quelles sont les étapes d’une bonne démarche scientifique ?](https://www.scribbr.fr/article-scientifique/demarche-scientifique/#:~:text=La%20d%C3%A9marche%20scientifique%20permet%20d,de%20nouvelles%20hypoth%C3%A8ses%20%C3%A0%20tester.)

---

**_février 2025_**
```

## Auteurs

Nous sommes quatres étudiants de Polytech Nice-Sophia en spécialité Sustainable Software Engineering :
* [Dorian Bouchard](https://github.com/dirianDB) dorian.bouchard@etu.unice.fr
* [Julien Didier](https://github.com/JulienDidier-PNS) julien.didier@etu.unice.fr
* [Yasmine Moussaoui](https://github.com/yas-mous) yasmine.moussaoui@etu.unice.fr
* [Ivan van der Tuijn](https://github.com/Ivan-vanderTuijn) ivan.van-der-tuijn@etu.unice.fr

## I. Projet et context de recherche

```
(Préciser ici votre contexte et Pourquoi il est intéressant.)
```


Les Large Language Models (LLMs) sont de plus en plus présents dans notre société, où ils se distinguent notamment par leur capacité à générer et synthétiser des textes. Ces modèles présentent toutefois certaines limites, à la fois en termes de résultats et d’accessibilité. En effet, la croissance exponentielle du nombre de paramètres qui composent ces modèles a entraîné une augmentation significative de leur taille et des ressources nécessaires pour leur entraînement et leur utilisation. Par ailleurs, leur tendance à “halluciner” des informations erronées rend leur usage problématique pour des utilisateurs non avertis.

Récemment, une nouvelle approche émerge avec le développement de Small Language Models (SLMs), des modèles plus petits et spécialisés, qui visent à surpasser les LLMs dans des cas d’usage spécifiques. Grâce à leur taille réduite, les SLMs peuvent fonctionner localement, offrant une alternative plus légère et accessible tout en répondant à des problématiques ciblées.

Ce chapitre explore donc le sujet des SLMs vs LLMs. Plus précisément, cette étude a pour objectif d'analyser les tendances d'utilisation des SLMs et d’identifier l’intérêt et les limitations des SLMs par rapport aux LLMs actuellement utilisés.

Nous avons trouvé ce sujet particulièrement captivant, car il est au cœur de nombreuses discussions actuelles. Les avis divergent largement, avec des retours tant positifs que négatifs sur ces systèmes. Ce qui nous a semblé justifier une exploration approfondie pour mieux comprendre cette évolution des modèles de langage et leurs implications.

Une des finalités qui nous motive est de determiner dans quels cas d'utilisation un SLM pourrait se révéler plus pertinent qu'un LLM, et inversement. Étant donné que cette technologie est encore relativement récente, notre approche reste exploratoire et ouverte à de nouvelles découvertes.


## II. Observations et question générale
```
1. Commencez par formuler une question sur quelque chose que vous observez ou constatez ou encore une idée émergente.

2. Préciser pourquoi cette question est intéressante de votre point de vue.

Attention pour répondre à cette question, vous devrez être capable d'émettre des hypothèses vérifiables, de quantifier vos réponses, ...
d
     :bulb: Cette première étape nécessite beaucoup de réflexion pour définir la bonne question qui permet de diriger la suite de vos travaux.

Observations/Question générale : formulation, intérêt, limites éventuelles.

```

À partir de nos premières observations et de nos connaissances initiales sur les modèles de langages, nous avons remarqué que la distinction entre l’usage d’un SLM et d’un LLM n’est pas clairement définie ni implicite.
Cette absence de clarté peut poser des défis pour comprendre leurs avantages respectifs, leurs limites, et leur pertinence dans divers contextes d’utilisation.

Pour mieux cerner ces aspects, nous avons formulé la question générale suivante :  
**Quels sont les enjeux et les cas d’usage des SLM et des LLM ?**

Cette question nous a semblé intéressante, car elle englobe different aspects :
* Analyse des bénéfices et contraintes des deux types de modèles.
* Identification de scénarios où un modèle est préférable à l’autre.
* Guide sur les critères à prendre en compte dans des projets nécessitant l’utilisation de modèles de langages.

Pour préciser et structurer notre réflexion, nous avons décliné la question générale en plusieurs sous-questions clés, permettant une exploration approfondie.

1. **Existe-t-il un seuil de paramètres permettant de différencier SLM et LLM ?**  
   Bien que les SLM et LLM partagent de nombreuses caractéristiques, leur principale différence réside dans leur taille. Déterminer un seuil clair à partir duquel un modèle est considéré comme un SLM ou un LLM est la premiere étape pour les différencier et mieux comprendre leurs spécificités.

2. **Un SLM peut-il être aussi performant, voire plus performant, qu’un LLM lorsqu’il est appliqué à une tâche spécifique ?**  
   Cette question vise à explorer si, dans des contextes spécialisés, un SLM peut surpasser un LLM en efficacité et en précision, grâce à sa spécialisation et à ses besoins en ressources moindres.

3. **Quels sont les cas d’usage spécifiques aux SLM ?**  
   Identifier les domaines d’application où les SLM apportent une réelle valeur ajoutée par rapport aux LLM permet de formuler des scénarios types et de comprendre leur pertinence dans des contextes spécifiques.

4. **Quelles sont les tendances d’utilisation SLM/LLM ?**  
   Étudier les tendances actuelles et passés permet de mieux comprendre l’émergence des SLM et leur adoption progressive, tout en émettant des hypothèses sur leur évolution future.

Répondre à ces questions implique de surmonter certaines limitations, notamment :
* L’évolution rapide du domaine de l’intelligence artificielle, qui peut rendre certaines observations rapidement obsolètes.
* // TODO Autres ?

## III. Collecte d'informations
```
Préciser vos zones de recherches en fonction de votre projet, les informations dont vous disposez, ...

Voici quelques pistes :

1. les articles ou documents utiles à votre projet
2. les outils que vous souhaitez utiliser
3. les jeux de données/codes que vous allez utiliser, pourquoi ceux-ci, ...

   :bulb: Cette étape est fortement liée à la suivante. Vous ne pouvez émettre d'hypothèses à vérifier que si vous avez les informations. inversement, vous cherchez à recueillir des informations en fonction de vos hypothèses.
```
### Articles
Pour la recherche d'information concernant les SLM, ainsi que les résultats de l'évaluation de leur performance faite par des benchmarks.  Nous avons consulté plusieurs articles scientifiques.
* [A Comprehensive Survey of Small Language Models in the Era of Large Language Models: Techniques, Enhancements, Applications, Collaboration with LLMs, and Trustworthiness](https://arxiv.org/abs/2411.03350)
* [Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186)
* [Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118)

// TODO : Ajouter les articles consultés pour les autres parties

### Ressources
Nous avons utilisé les ressources suivantes pour nos expériences :
* [Hugging Face](https://huggingface.co/)
* [Repository GitHub référençant les SLM et leur Technical Report](https://github.com/FairyFali/SLMs-Survey)

### Codes
Nous avons développé des codes permettant de récupérer, traiter et afficher les données provenant de l'API Hugging Face.
Cela afin de mener à bien nos expériences.
Les codes utilisés se trouvent dans le répertoire `chapters/2025/SLM_vs_LLM-Team_C/assets/codes` du répository github.

### Datasets
Les données utilisées pour nos expériences proviennent de l'API Hugging Face.

## IV. Hypothèses et expériences
```
1. Il s'agit ici d'**énoncer sous forme d'hypothèses** ce que vous allez chercher à démontrer. Vous devez définir vos hypothèses de façon à pouvoir les _mesurer/vérifier facilement._ Bien sûr, votre hypothèse devrait être construite de manière à _vous aider à répondre à votre question initiale_. Explicitez ces différents points.
2. Vous **explicitez les expérimentations que vous allez mener** pour vérifier si vos hypothèses sont vraies ou fausses. Il y a forcément des choix, des limites, explicitez-les.

   :bulb: Structurez cette partie à votre convenance :
   Par exemples :
   Pour Hypothèse 1 =>
   Nous ferons les Expériences suivantes pour la démontrer
   Pour Hypothèse 2 => Expériences


        ou Vous présentez l'ensemble des hypothèses puis vous expliquer comment les expériences prévues permettront de démontrer vos hypothèses.
```
### Hypothèses :

#### Hypothèse 0 :
Pour répondre à la sous-question "Existe-t-il un seuil de paramètres permettant de différencier SLM et LLM ?", nous avons initialement formulé l'hypothèse qu’il existe un seuil universel, reconnu et plus ou moins standardisé, pour différencier les SLM et les LLM. Cependant, nos recherches préliminaires ont montré que ce seuil n’est pas clairement défini. Nous postulons donc que ce seuil pourrait varier selon l’éditeur et son référentiel de modèles.

**Expérience** :  
Pour tester cette hypothèse, nous avons mené une série d’analyses visant à explorer les différences en terme de nombre de parametre entre SLM et LLM. Une analyse comparative des modèles publiés par les principaux acteurs de l’IA a été réalisée. Nous avons étudié les modèles développés par des entreprises telles qu’OpenAI, Microsoft, Google, QwenML, Mistral et d'autres, afin d’identifier leur taille (en nombre de paramètres) et la classification officielle qui leur est attribuée (SLM ou LLM), ainsi que la nature de la série à la quelle appartient le modèle.

Enfin, nous avons procédé à une synthèse des données récoltés. Cette démarche consistait à croiser les résultats obtenus pour dégager des tendances ou des seuils récurrents tout en identifiant les disparités potentielles liées au contexte d’application ou aux choix des éditeurs.

**Démarche :**
Notre démarche s’est appuyée sur la compilation des informations publiques disponibles concernant les modèles existants. Ces informations ont été collectées à partir d’articles, de publications scientifiques et de documentations officielles, mais aussi de l'API Hugging Face. Nous avons effectué une analyse quantitative des seuils identifiés, en mettant l’accent sur le nombre de paramètres et sur la classification adoptée par les éditeurs, qu’elle soit catégorisée comme "small", "large", ou autre. Une interprétation des résultats a suivi pour évaluer si un seuil commun pouvait émerger ou si ces seuils demeuraient spécifiques à chaque acteur.

**Points à expliciter :**
Les limites potentielles de cette analyse incluent l’incomplétude ou l’absence d’informations concernant certains modèles propriétaires, ce qui pourrait biaiser les conclusions. En termes de choix méthodologiques, nous avons décidé de concentrer initialement nos efforts sur les acteurs majeurs de l’IA, car ils façonnent largement les tendances actuelles du domaine. Nous avons également inclus des études académiques afin d’obtenir une perspective plus neutre et complète.

---

**Limites** :  
La cohérence des résultats de ces expériences est dépendante de l’évolutivité des seuils des paramètres de LLM dans le temps (ex : 2019 LLM : GPT-2 1.5B).

#### Hypothèse 1 :
**Question** : Un SLM peut être aussi performant, voire plus performant, qu’un LLM lorsqu’il est appliqué à une tâche spécifique?  
**Expériences** :
Nous allons mener des experiences afin de comparer les performances entre SLM et LLM sur différents benchmarks spécialisés dans des domaines plus ou moins vastes.
Cela nous permettra de déterminer si un SLM peut atteindre une performance similaire à celle d'un LLM dans des cas d’utilisation spécifiques.
1. Comparaison des performances d’un SLM et d’un LLM sur un benchmark de génération de code.
2. Comparaison des performances d’un SLM et d’un LLM sur un benchmark de complétion de code.
3. Comparaison des performances d’un SLM et d’un LLM sur un benchmark de génération de texte.
4. Comparaison des performances d’un SLM et d’un LLM sur un benchmark de text-to-speech.

#### Hypothèse 2 :
**Question** : Quels sont les cas d’usages spécifiques aux SLM ?  
**Expériences** :  
Nous allons mener des expériences afin de déterminer les cas d’usages spécifiques aux SLM.
Cela nous permettra d’identifier les domaines pour lesquels les SLM sont conçus. Sur la base de la façon dont ils sont présentés par leur éditeur.
1. A quel moment utiliser un LM, pour quel type de tâches ?
2. Quand est-ce qu’un SLM est plus performant qu’un LLM ?
3. Analyse de données sur HuggingFace

#### Hypothèse 3 :
**Question** : Quelles sont les tendances d’utilisation SLM/LLM ?
**Expériences** :  
Nous allons mener des expériences afin de déterminer s'il y a des tendances d'utilisation des SLM et LLM. Et connaitre la part d'utilisation des SLM par rapport aux LLM.
1. Analyse des tendances d’utilisation des modèles (SLM/LLM) disponibles sur Hugging Face.

## V. Result Analysis and Conclusion

### Hypothèse 2
#### 1. A quel moment utiliser un LM, pour quel type de tâche ?

```Les modèles de langage (Language Models, ou encore LM) ont connu une évolution fulgurante au cours des dernières décennies, marquant l'histoire dans le domaine de l'intelligence artificielle. À leurs débuts, les LM s’appuyaient sur des méthodes statistiques simples comme les modèles n-grams, reposant sur des entrées très limitées et nécessitant des règles explicites pour capturer pour espérer avoir un semblant de conversation avec une machine. Ces approches, bien que novatrices pour l’époque, étaient limitées par leur capacité à généraliser leur champ d'action et leur dépendance à des données soigneusement préparées.

Avec l'avènement des réseaux de neurones, l'introduction des modèles comme Word2Vec et GloVe a représenté un premier bond dans les LM, en offrant des représentations vectorielles denses du langage. Cependant, ce n'est qu'avec les architectures de type Transformer, introduites par Vaswani et al. en 2017, que les LM ont atteint un nouveau niveau de performance comme ceux que l'on connait aujourd'hui. Des modèles tels que BERT, GPT et leurs évolutions (GPT-3, GPT-4) ont ouvert la voie à une compréhension et une génération de texte d'une précision impressionnante, grâce à leur capacité à capturer des relations complexes à travers de vastes quantités de données.

https://fr.wikipedia.org/wiki/Word2vec
https://en.wikipedia.org/wiki/GloVe
https://fr.wikipedia.org/wiki/Transformeur


Dans ce contexte, deux catégories de modèles de langage émergent :

Les modèles de langage généralistes (LLM), conçus pour traiter une grande variété de tâches et capables de s'adapter à divers domaines grâce à leur entraînement sur des corpus massifs.
Les modèles de langage spécialisés (SLM), développés pour répondre à des besoins spécifiques, que ce soit dans des secteurs particuliers (santé, finance, etc.) ou pour des tâches ciblées nécessitant une précision accrue.
```

Il faut savoir que l'on connait les LM depuis plus longtemps que l'on ne le croit. Un LM n'est pas forcement obliger d'avoir une forme d'intelligence poussée. Les premiers LM qui ont vus le jour on commencer par des tâches très ciblées :
- Correction orthographique et suggestion de mots
- Reconnaissance vocale
- Traduction Automatique

Pour chaques langues, il fallait faire un nouveau LM. Pour chaque nouveau cas d'usage, il fallait en faire un nouveau.

https://www.smalsresearch.be/nlp-modeles-de-langue/
https://fr.wikipedia.org/wiki/Histoire_des_langages_de_programmation
https://www.mongodb.com/fr-fr/resources/basics/artificial-intelligence/large-language-models

Désormais, les LM sont pratiquement tous basés sur l'architecture Transformer que nous avons précédemment. Cette architecture à permis aux LM à monter en "intelligence".

Il est désormais possible d'avoir un LM qui puisse répondre à plus de choses en même temps. Cela est possible par, notamment :
- l'Augmentation des capacités des modèles. Au cours du temps, le nombre de paramètre n'a cesser d'augmenter, rendant la capacité des LM à comprendre mieux les informations qui lui sont données.

- l'Augmentation de la durée / du nombre de données lors de l'entrainement.
Avec le temps, les puissances de calculs ont permis d'entrainer plus vite les LM. Tout comme avec le temps, le nombre de data n'a cessé d'augmenter, ce qui permet aux LM de pouvoir s'entrainer sur des sources de données de plus en plus grandes.

Les cas d'utilisations de nos jours deviennent de plus en plus variés. Pour trouver un équilibre dans les utilisations, les LM ont été divisés en deux catégories (SLM et LLM).

Au final, on peut dire d'après les résultats précédents que les LM sont généralement utilisés pour de la génération de texte. Entre la traduction, la correction orthographique,... il semble que l'interface première entre les humains et ces LM soit le texte.

https://www.autolex.ai/post/guide-pratique-des-cas-dutilisation-des-grands-modeles-de-langage-llm?utm_source=chatgpt.com

#### 2. Quand est-ce qu’un SLM est plus performant qu’un LLM ?

Maintenant que nous avons vu qu'il existait deux types de LM (SLM / LLM), essayons d'étudier les différences entre eux, ce qui pourrait montrer pourquoi les cas d'usages peuvent autant varier.

Pour cette étude, nous ne testerons pas nous même les SLM/LLM. Nous essaierons de trouver des recherches sur ce sujet pour essayer de trouver justement des différences (ou non) à ce sujet.

Dans pratiquement tous les sujets que nous avons regarder, tous sont d'accord sur le fait que les SLM ont une capacité similaire aux LLM à répondre à un sujet bien spécifique sur un domaine (avec quelques fois, une plus grande précision dans les réponses). On le voit donc, un LLM apport une simplicité d'utilisation qui empiète sur sa qualité de réponse. Tandis qu'un SLM, bien qu'il ne soit pas aussi bon qu'un LLM sur tous les domaines, peut exceller dans un domaine en particulier, ce qui lui permet d'être plus petit (car moins de choses à apprendre, à analyser), ce qui lui permet donc d'être utilisé de manière local sur une machine. Les cas d'utilisations sont donc plus ciblés côté SLM que LLM. Les LLM de nos jours sont utilisés pour à peu près tous les domaines sans exception. Tandis que les SLM, eux, comme dit précédemment, doivent obligatoirement cibler un sujet en particulier. 

https://www.analyticsvidhya.com/blog/2024/11/slms-vs-llms/
https://ieeexplore.ieee.org/abstract/document/10590016?casa_token=LDxEMDqRddYAAAAA:5TLDKfJEleeNj-Fup-ZEOaq2d9yvUwgyPD3WeWBFhqAnn14-5-LrC2UUfFEfVOAslmSiTkehWZi1
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10590016


#### 3. Analyse de données sur HuggingFace

Pour vérifier tout ce que nous avons dit dans la partie précédente, nous allons utiliser l'outil HuggingFace qui permet à n'importe qui de déposer des LM mais aussi des données d'entrainement.

Nous avons choisit celui-ci car c'est le leader sur ce marché, c'est donc sur cette plateforme que nous serons les plus aptes à extraire des informations en grande quantitié pour essayer de confirmer ou non, si les LLM / SLM ont bien des cas d'usages différents et si oui, quels sont-ils.

Pour commencer, il faut savoir que sur HuggingFace, il n'y a pas de définition SLM/LLM. Tous les modèles sont des LM. Pour pouvoir donc différencier ces LM, nous avons appliquer la limite vu dans l'hypothèse précédente, qui est la limite des 7 Milliards de paramètres.

Avant d'analyser les résultats, nous tenons quand même ç mentionner que ces résultats ne représentent pas la réalité, nous avons essayer de faire au mieux mais plusieurs freins nous ont limités dans ces analyses :
- La période dans le temps est très courte. En effet, l'API de HuggingFace nous permet de récolter uniquement des données à l'instant ou nous les prenons. Par exemple, les statistiques sur les téléchargements sont valables uniquement sur les 30 derniers jours. Nous ne pouvons accéder a ces informations antérieures.

- Le manque d'informations dans les modèles. Bien que HuggingFace soit la plateforme de référence dans ce domaine, certains modèles manquent de données. Lors de l'execution de nos scripts, nous filtrons les modèles qui ont l'information sur leurs nombre de parmaètres ou non. Une grande partie des modèles présents sur la plateforme n'ont pas ce paramètre, ce qui entraine une inutilité du modèle pour nous.

Pour commencer notre analyse, nous nous baserons sur un jeu de donnée de 5000 modèles. Ces modèles ont été téléchargés par ordre décroissant de téléchargement.
Pour commencer, commencons par regarder quel est le ratio de SLM/LLM présent dans nos données.

![](./assets/results/models_repartition_5000.PNG)

On voit que les SLM sont plus présents de manière générale. Ce qui peut sembler logique car si les SLM sont aussi, voir plus puissant que les LLM mais pour des domaines bien spécifiques, pour un LLM bon sur tous les sujets, il faut créer un SLM pour chaques sujets.

Cherchons maintenant à voir quels types d’usages ont les utilisateurs avec les LM. Pour cela, les contributeurs aux LM peuvent attribuer à leur modèle un ou plusieurs “tag” qui permettent de mieux cibler le type de tâche du modèle. Nous pouvons nous baser là-dessus pour avoir une idée en fonction du type de modèle.

![](./assets/results/llm_most_tags_5000.PNG)

![](./assets/results/slms_most_tags.png)

On remarque que les SLM ont des tags bien plus précis que les LLM. Les LLM sont clairement axés sur la génération de texte, avec une application dominante dans le traitement de texte complexe ou créatif. Les SLM sont beaucoup plus polyvalents, couvrant des domaines tels que la vision par ordinateur, la reconnaissance audio, et les analyses spécialisées. Ils répondent à des besoins variés, avec une utilisation répartie sur des tâches plus techniques ou spécifiques. Ce qui confirme notre hyptothèse précédente via les benchmarkings.

Nous aurions apprécier trouver quelques informations supplémentaires comme les langages utilés pour développer le LM ou encore le type de dataset qui a été utilisé pour entrainer le modèle. Cela aurait pu permettre de savoir si un langage à permis de développer le dévoppement des LM par exemple, ou encore si un certain type de dataset est plus utilsé sur les SLM. Ce qui peut indiquer quels types de besoins les modèles doivent cerner le plus. Et voir, si en fonction des types de modèles, les SLM ou les LLM s’appuient sur des dataset communs ou pas dutout.


### Hypothèse 3
2. Quelles sont les tendances d’utilisation SLM/LLM ?
Nous nous sommes posés deux sous questions pour répondre à notre hypothèse :
   - Est-ce que l'utilisation des SLM/LLM dure dans le temps  ?
   - Est-ce que l'utilisation des SLM est en croissance dans certains domaines ?

Est-ce que l'utilisation des SLM/LLM dure dans le temps ?
L'api de ugginface nous permet d'obtenir pour les le nombre de téléchargement les 30 derniers jours ainsi que les tags ( les cas d'usages ) et la date de de cree

Nous avons donc décidé de regarder quelle etaie la part qu'occupaient les modèles crées pour quaque moi des télacdes 30 derniers jours. Nous avons donc pu réliser le graphique suivant :
 lien
Le graphique affiche donc la répartition des téléchargemebts en fonction de te de pour les slm et llm.
Nous nous attendions à une forte crsoissant de la courbe sur les derniers mois (ce qui signifie que les nouveaux modèles profitent d'un gain de popularité avec leur récent)
Ce n'est pourtant pas le cas, en effet mememe si on observe un forte croissance sur la fin, la pente de la courbe est pluto^t linéaire ce qui signigie que les modèles perdurent dans le temps.
Nous ne notons pas de différence majeure entre les slm et llm si ce n'est que les slm crée long occupent une partie très importante

Est-ce que l'utilisation des SLM est en croissance dans certains domaines ?
nous avons regardé pour tous les tags le pourcentage de slm/llm cré par mois
ensemble les
ee
Nous avons fait face à certains soucis, en effet certains tags sont sous représentés et ne permettent donc pas de confirmer nos hypothèses.
e




1. Présentation des résultats
2. Interprétation/Analyse des résultats en fonction de vos hypothèses
3. Construction d’une conclusion

   :bulb:  Vos résultats et donc votre analyse sont nécessairement limités. Préciser bien ces limites : par exemple, jeux de données insuffisants, analyse réduite à quelques critères, dépendance aux projets analysés, ...

## VI. Outils \(facultatif\)
```
Précisez votre utilisation des outils ou les développements \(e.g. scripts\) réalisés pour atteindre vos objectifs. Ce chapitre doit viser à \(1\) pouvoir reproduire vos expérimentations, \(2\) partager/expliquer à d'autres l'usage des outils.
```
![Figure 1: Logo UCA, exemple, vous pouvez l'enlever](assets/images/logo_uca.png)


## VI. References

[Debret 2020] Debret, J. (2020) La démarche scientifique : tout ce que vous devez savoir ! Available at: https://www.scribbr.fr/article-scientifique/demarche-scientifique/ (Accessed: 18 November 2022).


[LM History](https://en.wikipedia.org/wiki/Language_model)
