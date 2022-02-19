<html lang="en">
  <head>
    <title>Vis Network | Basic usage</title>

    <script
      type="text/javascript"
      src="./umd/vis-network.min.js"
    ></script>

    <style type="text/css">
      #mynetwork {
        width: 1200px;
        height: 800px;
        border: 1px solid lightgray;
      }
    </style>
  </head>
  <body>
    <p>Create a simple network with some nodes and edges.</p>

    <div id="mynetwork"></div>

    <script src="app.js"></script>
    
  </body>
</html>


---
layout: default
title : Extraire les préconditions des codes de RapidMiner
date:   2022-01-08 17:08:00 +0100
---
**_January 2022_**

## Authors

We are five students in last year of Polytech' Nice-Sophia specialized in Software Architecture :

* Tigran Nersissian &lt;tigran.nersissian@etu.univ-cotedazur.fr&gt;
* Yann Martin D&#039;Escrienne &lt;yann.martin&dash;d&#039;escrienne@etu.univ-cotedazur.fr&gt;
* Valentin Campello &lt;valentin.campello@etu.univ-cotedazur.fr&gt;
* Lucie Morant &lt;lucie.morant@etu.univ-cotedazur.fr&gt;
* Yohann Tognetii  &lt;yohann.tognetti@etu.univ-cotedazur.fr&gt;

## I. Research context /Project
Extraire les préconditions des opérateurs par analyse des codes de RapidMiner ou par IHM.

RapidMiner est un outil très avancé de construction de workflows de ML.
Il intègre notamment la possibilité de définir de nouveaux opérateurs (algorithmes) en précisant les préconditions sur ces algorithmes et l’impact sur les données en sortie. Cette information est utilisée pour aider l’utilisateur en l’empêchant de connecter des data sets et des opérateurs inadaptés. 

En partant des codes de RapidMiner, nous aurions aimé “sortir” cette connaissance pour l’étudier et la ré-injecter dans un environnement dédié à l’enseignement.

Voici une première visualisation des classes impliquées dans la vérification des préconditions.

Sauriez-vous extraire des codes de RapidMiner
les préconditions sur les opérators ? les associer à la hiérarchie de définition des opérateurs, sachant que nous avons déjà identifié que par surcharge, certaines “capabilities” n’en sont plus dans les sous classes.
Faire des stats sur l’utilisation des préconditions?
Analyser les impacts (comment les données sont modifiées).


![Figure 1: Logo UCA, exemple, vous pouvez l'enlever](../assets/model/UCAlogoQlarge.png){:height="25px" }


## II. Observations/General question

1. Commencez par formuler une question sur quelque chose que vous observez ou constatez ou encore une idée émergente. Attention pour répondre à cette question vous devrez être capable de quantifier vos réponses.
2. Préciser bien pourquoi cette question est intéressante de votre point de vue et éventuellement en quoi la question est plus générale que le contexte de votre projet \(ex: Choisir une libraire de code est un problème récurrent qui se pose très différemment cependant en fonction des objectifs\)

Cette première étape nécessite beaucoup de réflexion pour se définir la bonne question afin de poser les bonnes bases pour la suit.

## III. information gathering

Préciser vos zones de recherches en fonction de votre projet,

1. les articles ou documents utiles à votre projet
2. les outils
 
## IV. Hypothesis & Experiences

1. Il s'agit ici d'énoncer sous forme d' hypothèses ce que vous allez chercher à démontrer. Vous devez définir vos hypothèses de façon à pouvoir les _mesurer facilement._ Bien sûr, votre hypothèse devrait être construite de manière à v_ous aider à répondre à votre question initiale_.Explicitez ces différents points.
2. Test de l’hypothèse par l’expérimentation. 1. Vos tests d’expérimentations permettent de vérifier si vos hypothèses sont vraies ou fausses. 2. Il est possible que vous deviez répéter vos expérimentations pour vous assurer que les premiers résultats ne sont pas seulement un accident.
3. Explicitez bien les outils utilisés et comment.
4. Justifiez vos choix

## V. Result Analysis and Conclusion

1. Analyse des résultats & construction d’une conclusion : Une fois votre expérience terminée, vous récupérez vos mesures et vous les analysez pour voir si votre hypothèse tient la route. 

## VI. Tools \(facultatif\)

Précisez votre utilisation des outils ou les développements \(e.g. scripts\) réalisés pour atteindre vos objectifs. Ce chapitre doit viser à \(1\) pouvoir reproduire vos expérimentations, \(2\) partager/expliquer à d'autres l'usage des outils.

## VI. References

1. ref1
1. ref2
