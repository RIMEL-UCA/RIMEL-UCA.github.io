---
layout: default
title : SLM vs LLM - Team C
date:   2024-11
---

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

## Authors

We are four students in last year of Polytech' Nice-Sophia specialized in Software Architecture :
* [Dorian Bouchard](https://github.com/dirianDB) dorian.bouchard@etu.unice.fr
* [Julien Didier](https://github.com/JulienDidier-PNS) julien.didier@etu.unice.fr
* [Yasmine Moussaoui](https://github.com/yas-mous) yasmine.moussaoui@etu.unice.fr
* [Ivan van der Tuijn](https://github.com/Ivan-vanderTuijn) ivan.van-der-tuijn@etu.unice.fr

## I. Research context /Project

(Préciser ici votre contexte et Pourquoi il est intéressant.)

Nous avons trouvé ce sujet particulièrement captivant, car il est au cœur de nombreuses discussions actuelles. Les avis divergent largement, avec des retours positifs et négatifs sur ces systèmes, ce qui nous a semblé justifier une exploration approfondie pour mieux comprendre l'évolution des modèles de langage.

En effet, il existe différents types de modèles de langage, et nous souhaitons examiner dans quels cas d'utilisation un SLM pourrait être pertinent, tout comme un LLM. Actuellement, cette technologie étant encore relativement récente, nous l'explorons de manière assez spontanée, car nous sommes en phase de découverte. Poser des questions pour approfondir notre compréhension de leur fonctionnement, de leurs atouts et de leurs limites nous permettra de clarifier le sujet et d'élaborer un guide initial sur les bonnes pratiques d'utilisation.

## II. Observations/General question

1. Commencez par formuler une question sur quelque chose que vous observez ou constatez ou encore une idée émergente.

2. Préciser pourquoi cette question est intéressante de votre point de vue.

Attention pour répondre à cette question, vous devrez être capable d'émettre des hypothèses vérifiables, de quantifier vos réponses, ...
d
     :bulb: Cette première étape nécessite beaucoup de réflexion pour définir la bonne question qui permet de diriger la suite de vos travaux.

### Question générale : Quels sont les enjeux, et les cas d’usage entre les SLM et LLM ?

## III. Information gathering

Préciser vos zones de recherches en fonction de votre projet, les informations dont vous disposez, ...

Voici quelques pistes :

1. les articles ou documents utiles à votre projet
2. les outils que vous souhaitez utiliser
3. les jeux de données/codes que vous allez utiliser, pourquoi ceux-ci, ...

   :bulb: Cette étape est fortement liée à la suivante. Vous ne pouvez émettre d'hypothèses à vérifier que si vous avez les informations. inversement, vous cherchez à recueillir des informations en fonction de vos hypothèses.

## IV. Hypothesis & Experiences

1. Il s'agit ici d'**énoncer sous forme d'hypothèses** ce que vous allez chercher à démontrer. Vous devez définir vos hypothèses de façon à pouvoir les _mesurer/vérifier facilement._ Bien sûr, votre hypothèse devrait être construite de manière à _vous aider à répondre à votre question initiale_. Explicitez ces différents points.
2. Vous **explicitez les expérimentations que vous allez mener** pour vérifier si vos hypothèses sont vraies ou fausses. Il y a forcément des choix, des limites, explicitez-les.

   :bulb: Structurez cette partie à votre convenance :
   Par exemples :
   Pour Hypothèse 1 =>
   Nous ferons les Expériences suivantes pour la démontrer
   Pour Hypothèse 2 => Expériences


        ou Vous présentez l'ensemble des hypothèses puis vous expliquer comment les expériences prévues permettront de démontrer vos hypothèses.

### Hypothèses :

#### Hypothèse 0 :  
**Question** : Existe-t-il un seuil de paramètres pour un SLM ?  
**Expériences** :  
Nous allons mener des expériences afin de déterminer si un seuil plus ou moins nette de nombre de paramètres existe pour un SLM. 
1. Analyse de la taille des modèles de langage et de leur présentation faite par les grands acteurs de l’IA (OpenAI, Microsoft, Google, etc.).
2. Analyse similaire auprès des laboratoires scientifiques et des chercheurs indépendants.

**Limites** :  
La cohérence des résultats de ces expériences est dépendante de l’évolutivité des seuils des paramètres de LLM dans le temps (ex : 2019 LLM : GPT-2 1.5B).

#### Hypothèse 1 :  
**Question** : Un SLM peut être aussi performant, voire plus performant, qu’un LLM lorsqu’il est appliqué à une tâche spécifique.  
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
1. Analyse des tendances des cas d’utilisation des modèles (SLM/LLM) disponibles sur Hugging Face.

#### Hypothèse 3 :  
**Question** : Quelles sont les tendances d’utilisation SLM/LLM ?
**Expériences** :  
Nous allons mener des expériences afin de déterminer s'il y a des tendances d'utilisation des SLM et LLM. Et connaitre la part d'utilisation des SLM par rapport aux LLM.
1. Analyse des tendances d’utilisation des modèles (SLM/LLM) disponibles sur Hugging Face.

## V. Result Analysis and Conclusion

1. Présentation des résultats
2. Interprétation/Analyse des résultats en fonction de vos hypothèses
3. Construction d’une conclusion

   :bulb:  Vos résultats et donc votre analyse sont nécessairement limités. Préciser bien ces limites : par exemple, jeux de données insuffisants, analyse réduite à quelques critères, dépendance aux projets analysés, ...

## VI. Tools \(facultatif\)

Précisez votre utilisation des outils ou les développements \(e.g. scripts\) réalisés pour atteindre vos objectifs. Ce chapitre doit viser à \(1\) pouvoir reproduire vos expérimentations, \(2\) partager/expliquer à d'autres l'usage des outils.

![Figure 1: Logo UCA, exemple, vous pouvez l'enlever](assets/images/logo_uca.png)


## VI. References

[Debret 2020] Debret, J. (2020) La démarche scientifique : tout ce que vous devez savoir ! Available at: https://www.scribbr.fr/article-scientifique/demarche-scientifique/ (Accessed: 18 November 2022).
