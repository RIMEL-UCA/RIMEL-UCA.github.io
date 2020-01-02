# Les feature toggles créent-ils de la dette technique ?

## Auteurs <a id="auteurs"></a>

* Couvreur Alexis
* Matteo Lucas
* Junac Jérémy
* Melkonian François

## I. Contexte de recherche <a id="i-contexte-de-recherche"></a>

> _"Feature Toggles \(often also refered to as Feature Flags\) are a powerful technique, allowing teams to modify system behavior without changing code."  
> --_ Pete Hodgson, sur le site[ martinfowler.com](https://martinfowler.com/)

Comme expliqué par Pete Hodgson, le _feature toggling_ permet de changer le comportement d'une application sans changer son code source, et donc sans la recompiler.

Ce principe est très largement utilisé pour tout type d'application. Il peut être utilisé sur les logiciels pour que l'utilisateur puisse le régler à sa guise, comme par exemple un navigateur web, mais aussi dans les applications de type _Software as a Service_, notamment lors de l'ajout de nouvelle fonctionnalité, comme c'est le cas pour Gmail par exemple.

La quasi totalité des projets utlise un principe de _feature toggling_, la plupart du temps sans même le savoir, tout simplement en utilisant un bloc conditionnel. Il existe cependant des _frameworks_, qui permettent d'ajouter une plus-value sur ce système conditionnel. Ils fournissent par exemple une API ou interface graphique pour la modification des valeurs.

De plus, découle du _feature toggling_ d'autre technique, utilisée pour rendre disponible petit à petit les nouvelles fonctionnalités aux utilisateurs. On peut par exemple citer le _dark launch_.



## II. Observations et question générale <a id="ii-observations-et-question-generale"></a>

Dans ce contexte, nous pouvons demander ce qu'il advient du code nécessaire au _feature toggle._ En effet, qu'il soit réalisé à l'aide d'un if ou par le biais d'un framework spécialisé, il est tout de même nécessaire d'introduire du nouveau code pour le _feature flag_. Ce code, qui n'est pas lié au métier de l'application, devra être maintenu en même temps que le reste de l'application.

L'utilisation de ce genre de pratique a donc un impact sur la base de code, ce qui peut soulever des questions quant à sa rentabilité. En d'autres termes, est-ce que l'utilisation d'un _feature toggle_ a un impact sur la qualité de notre code, et peut-on la quantifier ? Nous sommes donc arrivés à la problématique suivante:

Nous sommes quatres étudiants en dernière d'école d'ingénieur à Polytech Nice Sophia, spécialisé en Architecture Logicielle:

**Question générale : Les features toggles créent-ils de la dette technique ?**

Ainsi, l'objectif pour nous sera de chercher une corrélation en l'évolution de la technique, et plus spécifiquement son augmentation, et l'ajout de _feature toggle,_ s'il en existe une_._

Afin de faciliter notre étude, nous avons restreint notre analyse au projet utilisant effectivement un framework de _feature toggling_ \(sachant que ceux-ci peuvent être "fait maison"\) et de laisser de côté les simples if. En effet, il serait beaucoup trop compliqué pour nous dans un grand nombre de projet de disinguer les if lié aux métiers de l'application et les if liés à un _feature flag_.

## III. Rassemblement d'informations

### III.1. Les projets à analyser

Afin de mener notre étude à bien, nous avons entamés la recherche d'un jeu de données. Pour prouver notre point, nous devions avoir des projets utilisant du _feature toggling_ de manière régulière, mais aussi avec une base de code conséquente, nécessaire pour mesurer l'évolution de la dette technique. Pour la base de projet, nous avons utilisé la plateforme [GitHub](https://github.com/), une source reconnue inépuisable de projet opensource.

Nous allons voir dans les paragraphes suivants que la recherche de ce jeu de données a été bien plus compliqué que nous l'anticipions, et a connu de nombreuses évolutions au cours du temps.

Considérant ces points, nous avons tout d'abord choisi le langage sur lequel nous voulions travailler, ce qui est aussi lié au framework de _feature toggle_ disponible dans le dit langage. Deux couples langage/framework nous a alors paru assez populaire pour garantir une bonne base de projet dans notre jeu de données: le framework "[Togglz](https://www.togglz.org/)" dans le langage Java, et le framework "[LaunchDarkly](https://launchdarkly.com/)" en JavaScript.

Il aurait été trop complexe d'étudier les deux frameworks et les deux langages en même temps, il a donc fallu trancher entre les deux couples. Après étude des projets utilisant les deux frameworks via l'API de dépendance GitHub, la base de projet utilisant Togglz nous paraissait plus intéressante à étudié que LaunchDarkly. En effet, les projets utilisant ce framework étaient globalement plus gros, _ie._ avaient plus de commits, plus de contributeurs et était plus "sérieux". De plus, Java étant plus structuré et plus restricitf, ce qui facilite l'analyse de la dette technique.

Sur la centaine de projet utilisant Togglz, nous avons ensuite filtré les projets qui nous semblait les plus intéressant pour notre étude. Nous sommes arrivé au jeu de données suivant:

* [Estatio](https://github.com/estatio/estatio) \(4,300 commits, 9 contributeurs, 170 stars\)
* [ORCID-Source](https://github.com/ORCID/ORCID-Source) \(20,800 commits, 29 contributeurs, 158 stars\)
* [lightblue-migrator](https://github.com/lightblue-platform/lightblue-migrator) \(1,100 commits, 11 contributeurs, 3 stars\)
* [isis-app-todoapp](https://github.com/isisaddons/isis-app-todoapp) \(155 commits, 5 contributeurs, 37 stars\)

Ce jeu de données rassemble des assez gros projets comme des projets plus petit, mais tous utilisant de manière assidu le _feature toggle_. Nous avons donc établi le protocole suivant pour analyser notre jeu de données, pour chaque commit: identifier sa dette technique et identifier s'il introduit un _feature toggle_.

Ce protocole nous permettrait donc d’identifier la dette introduite par un feature toggle, et de corréler son évolution. A ce point du projet, la "dette technique" est une métrique "magique", _ie._ on ne sait pas comment et quoi calculer mais on considère qu'on sait le faire. Cette question sera résolue dans la partie 4.

Cependant, il y avait un problème dans notre raisonnement. Après discussion avec [Xavier Blanc](https://fr.linkedin.com/in/xavier-blanc-3b9785a), enseignant-chercheur à l'Université de Bordeaux et co-fondateur de ProMyze, notre jeu de données manquait de "projets témoins". Ces projets aurait le même métier que les projets que nous analysions, et premettrait d'appuyer nos observation. En effet, une augmentation de la dette peut ne pas être forcément lié au _feature toggle_, mais tout simplement au métier de l'application.   
Ces projets témoins \(avec le même métier donc\) aurait donc la même évolution de dette technique.

Néanmoins, il aurait été beaucoup trop compliqué et fasitdieux pour nous de trouvé pour chaque projet sélectionné un projet témoin de la même envergure avec le même métier. S'en sont suivies d'autres discussion avec Xavier Blanc et notre professeur encadrant Philippe Collet. 

A ce point, nous avons décidé de radicalement changer notre jeu de données. Si nous nous concentrions sur un unique projet, nous aurions toutes les données dont nous avions besoin, et le témoin serait le projet lui-même, pour peut qu'il soit assez gros.

Pour le choix du projet, il nous fallait donc un projet conséquent, utilisant le feature toggle, avec assez de contributeurs et une assez grosse base de code pour avoir des résultats non-biaisés. Aynt un unique projet à analyser, nous pouvions nous libérer de la contrainte du langage et du framework.

Notre première idée a été le noyau Linux. En effet, c'est un projet avec plus de 800,000 commits, connu pour reposer énormément sur le feature toggle \(cf. La distribution [Gentoo Linux](https://fr.wikipedia.org/wiki/Gentoo_Linux), tirant pleinement avantage de ceux-ci\). Cependant, nous avons rapidement abandonné cette idée, le noyau Linux étant trop complexe et ayant déjà été largement étudié, ne nous voyions pas de plus-value à ajouter.

Après une longue recherche de projet opensource à analyser, nous sommes tombés d'accord sur [Chromium](https://github.com/chromium/chromium), qui avec plus de 750,000 commits et une communauté très active est le candidat parfait pour nos expérimentations.

Chromium est écrit en C++, et possède son propre framework de _feature toggle_. Leurs framework permet de supporter 2 type de _feature flag_ : Certaines fonctionnalités seront activées à la compilation, en fonction par exemple du type de plateforme \(mobile, desktop,...\) ou de son système d'exploitation, et d'autres fonctionnalités dites "runtime", qui permettent par exemple d'avoir des accès anticipé sur les nouvelles fonctionnalités. Les deux types de _feature flag_ seront analysés dans ce projet.

### III.2. Les outils utilisés

Dans les parties précédentes et pendant la recherche de notre projet, nous avons désigné la "dette technique" comme une métrique "magique", qui nous permettrait d'évaluer l'état d'un projet à un instant t.

Après discussion avec nos professeurs, il en est ressorti que "la dette technique" était beaucoup trop vague, et surtout que pas toutes les métriques étaient utiles pour ce que nous souhaitions mesurer. En effet, le nombre de ligne dans une méthode ou la couverture de tests ne sont pas forcement lié au _feature toggling_.

Ils nous a donc fallu affiner ce que nous entendions par "dette technique". Nous sommes revenu à la définition même de _feature toggle_ : A quoi sert un _feature toggle_ ? A cette question, nous répondons qu'un _feature toggle_ permet de changer le comportement d'un logiciel. En d'autres termes, a certains endroit du code, il y a 2 \(ou plus\) flows d’exécutions, en fonction de la valeur du _feature flag_. C'est a ce moment que c'est devenu clair: la métrique la plus importante est la complexité cyclomatique.

Il nous fallait donc un outil pour rechercher la complexité cyclomatique en C++ **performant**. En effet, avec des milliers de commits à analyser pour avoir des résultats pertinents, on ne pouvais pas se permettre de passer plus de quelques minutes sur chacun. C'est alors que nous avons découvert [Lizard](https://github.com/terryyin/lizard). Lizard permet de mesurer la complexité cyclomatique et le nombre de lignes de code dans de nombreux langages, on obtenant des rapports précis \(resultat global, par fichier et par méthode\). Il remplit même le critère du temps d'execution, en s'exécutant en quelques minutes sur une bonne machine dans le cas de Chromium.

Pour le reste des outils, à savoir les scripts pour avoir la liste des commits qui ajoute un feature flag, _checkout_ un commit précis, l'analyser, extraire les résultats, nous avons écrits des scripts "maison".

## IV. Hypothèses et expériences <a id="iv-hypotheses-et-experiences"></a>

Avec le raffinement de noter sujet expliqué dans la partie III, l'hypothèse que nous allons essayer de prouver est la suivante.

**Hypothèse: Dans le cas de Chromium, l'ajout d'un feature toggle entraine une augmentation de la complexité cyclomatique.**

Pour prouver cette hypothèse, nous avons calculé l'ajout de complexité d'un commit implémentant un feature toggle et d'un commit normal. Nous avons établit le protocole suivant:

1. Définir quels commits nous intéresse, c'est-a-dire différencier les commits touchant au feature toggle du reste
2. Trouver la liste des commits qui répondent à ces critères
3. Pour chacun des commits de la liste calculer son impact sur la complexité cyclomatique du projet
4. Comparer la complexité obtenue entre les commit avec et sans feature toggle

## V. Analyse des résultats et conclusion

### V.1. Séparation des commits avec et sans feature toggle

Tout d'abord, nous avons défini ce qu'est un commit implémentant un feature toggle, pour nous c'est un commit modifiant le fichier 'content/public/common/content\_features.h'. En effet, c'est ce fichier qui contient la liste des feature toggles de chromium, une modification de celui-ci implique donc qu’un feature toggle a été modifié. 

Ce choix est critiquable.  Pour être certain qu’un commit à un impact sur le feature toggle, il nous faudrait générer un AST à partir du fichier _content-feature.h_ et utiliser celui afin de savoir si la modification du fichier a impacté le feature toggling. Dans notre cas par exemple, le simple ajout d'un commentaire est considéré comme une modification du feature toggling. Cependant nous pensons que ce genre de commit n'est pas assez fréquent pour changer fondamentalement nos résultats.

La modification du feature toggling est probablement possible à travers d'autres fichiers que _content-feature.h_  que nous utilisons, mais utiliser ce fichier est la manière la plus fiable que nous avons trouvé pour définir un changement dans le feature toggling.

### V.2. Liste des commits à analyser

L'obtention de la liste des commits à analyser s'est faite en utilisant directement le dépôt git chromium avec la commande _git log._ Grâce à celle ci on a pu facilement obtenir la liste des hash des commits qui nous intéresse.

En raison du temps de calcul nécessaire à l'analyse de cette liste, nous avons limité le nombre de commit. Nous avons analysé 1000 commits qui ne touche pas au feature toggle et 421 commits ajoutant un feature toggle. Plus le nombre de commit est grand, plus nos résultat sont fiables.

### V.3. Analyse des commits

Pour chacun d'eux nous avons calculé la complexité cyclomatique qu’il apporte par rapport au commit précédent. La taille du dépôt a posé problème pour cette étape puisqu’il il représente 10 GO de donnée. Cela implique que la durée d’un checkout est de l’ordre de 10 secondes. Il n’est donc pas réalisable de manipuler ce dépôt directement depuis notre machine. De plus il n’est pas possible ,en terme de temps de calcule, d’analyser la totalité du projet pour chaque commit.

Afin de palier à ces deux problèmes, nous avons utilisé l’API de Github. Elle nous as permis d’obtenir pour chaque commit la liste des fichiers modifiés et de les télécharger.  Pour chaque commit, nous analysons donc uniquement les fichiers modifiés par ce commit. Cela nous permet de ne pas avoir à manipuler le dépôt en entier et de gagner beaucoup de temps. Le résultat finale de complexité cyclomatique du commit est obtenue en utilisant _lizard_ sur chaque fichier afin d'en obtenir la complexité, et de faire la somme du traitement de tous les fichiers.

### V.4. Différence de complexité entre les deux types de commit

Pour les 421 commits trouvés, nous avons utilisé cette méthode pour calculer la complexité du commit et du commit précédent et obtenir une différence de complexité. De la même façon, nous avons analysé les 1000 commits les plus récents du dépôts pour avoir la complexité moyenne apporté par des commits "normaux"

Finalement, on obtient pour chaque commit la complexité qu’il apporte par rapport au précédent, en séparant les commits en deux : impactant ou non le feature toggle.

<table>
  <thead>
    <tr>
      <th style="text-align:center"></th>
      <th style="text-align:center">Commit normaux</th>
      <th style="text-align:center">Commit ajoutant un FEATURE TOGGLE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center">Complexit&#xE9; totale ajout&#xE9;e</td>
      <td style="text-align:center">3158</td>
      <td style="text-align:center">687</td>
    </tr>
    <tr>
      <td style="text-align:center">Nombre de commit analys&#xE9;</td>
      <td style="text-align:center">1000</td>
      <td style="text-align:center">421</td>
    </tr>
    <tr>
      <td style="text-align:center">Complexit&#xE9; moyenne</td>
      <td style="text-align:center">3.16</td>
      <td style="text-align:center">1.63</td>
    </tr>
    <tr>
      <td style="text-align:center">1er quartile</td>
      <td style="text-align:center">-1</td>
      <td style="text-align:center">0</td>
    </tr>
    <tr>
      <td style="text-align:center">M&#xE9;diane</td>
      <td style="text-align:center">1</td>
      <td style="text-align:center">0</td>
    </tr>
    <tr>
      <td style="text-align:center">3&#xE8;me quartile</td>
      <td style="text-align:center">3</td>
      <td style="text-align:center">1</td>
    </tr>
    <tr>
      <td style="text-align:center">Nombre de commit augmentant la complexit&#xE9;</td>
      <td style="text-align:center">222 (22%)</td>
      <td style="text-align:center">259 (62%)</td>
    </tr>
    <tr>
      <td style="text-align:center">
        <p>Commit entra&#xEE;nant un gros changement</p>
        <p></p>
      </td>
      <td style="text-align:center">15 (1.5%)</td>
      <td style="text-align:center">20 (4.8%)</td>
    </tr>
  </tbody>
</table>_Figure 1: complexité moyenne des commits analysés._

Comme nous pouvons le voir sur la Figure 1, la complexité ajoutée par un commit de feature toggle est inférieure à celle des commits normaux. Cela va à l'encontre de notre intuition, et mérite qu'on s'attarde plus sur ce résultat. 

Déjà, en observant la répartition de la complexité, on s'aperçoit que les commits ajoutant une feature toggle complexifie plus souvent le projet que les commits normaux. Cependant, les commits ajoutant une feature toggle implique un gros changement de complexité. Cela est d'autant plus flagrant sur cette représentation de l'évolution de la complexité ci dessous. \(Fig.2\)

![Figure 2: &#xE9;volution de la complexit&#xE9; de Chromium en fonction des ajouts de feature toggle dans le temps](../.gitbook/assets/image%20%2811%29.png)



![Figure 3: &#xE9;volution de la complexit&#xE9; de Chromium sur 1000 commits se suivant](../.gitbook/assets/image%20%2820%29.png)

Pour comprendre l'évolution de cette complexité, nous nous sommes concentrés sur les commits qui impliquaient des gros changements. Dans les faits, ces grosses différences sont au lié au guide des contributions de Chromium: Lors de l'ajout d'une fonctionnalité ou lors d'un changement profond d'une existante, le contributeur doit créer un feature toggle avec la nouvelle fonctionnalité, vérifier son bon fonctionnement en laissant les deux versions cohabiter, pour ensuite supprimer l'ancienne version quand la stabilité du changement a fait ces preuves. Par exemple, le commit ci dessous \([https://github.com/chromium/chromium/commit/6beb0cec0f47 ](https://github.com/chromium/chromium/commit/6beb0cec0f47)\) diminue la complexité globale de chromium en supprimant une version dépassé d'un algorithme qui n'était plus utilisé. Ce commit diminue de 300 la complexité cyclomatique, mais la complexité induite par la nouvelle version a été intégrée en plusieurs fois et ne transparaît pas dans notre analyse.

![Figure 4: Exemple de commit qui supprime une ancienne fonctionnalit&#xE9;  ](../.gitbook/assets/image%20%2810%29.png)

### V.5 Conclusion de notre analyse

L'analyse de la différence de dette technique entre ces deux groupes de commits ne nous permet pas de trouver une réponse à la question initiale. En effet, les pratiques mises en place sur Chromium tirent pleinement avantage des _feature toggles_. La parallélisation du développement des nouvelles fonctionnalités ne nous permet pas de connaître réellement l'impact de l'utilisation des _feature toggles_. Cependant, cette analyse nous a permis de mieux comprendre l'impact des _feature toggles_ sur une méthode permettant de faire évoluer le logiciel Chromium tout en garantissant une version livrable et déployable à tout moment. 

## VI. Menaces à la validité

**Peu de commits analysés.** Une des remarques qui peut nous être faites est le "peu de commits analysés" par rapport à la base de 750,000. En effet, au vue du nombre de commits fait chaque jour, il se peut que certains de nos résultats soit biaisés. Cependant, pour des raisons de temps \(lié aux problèmes de recherche de notre projet\), nous n'avons pas pu approfondir notre études. Tout d'abord, il aurait fallu prendre plus de commits autour des commits modifiant les feature togglings, mais aussi analyser plus de commit "normaux" pour avoir une base plus solide. Idéalement, il aurait même fallu analyser tous les commits afin d'avoir des résultats les plus précis possibles. Cependant, nos résultats ouvrent tout de même la porte à de premières conclusions.

**Manque de métrique de "dette technique".** Une autre menaces à nos résultats et la simplification de "dette technique" à "complexité cyclomatique". Là encore, pour des raisons de temps lié à la recherche de notre sujet, nous avons fait ce "raccourci" pour obtenir des résultats certes un peu simpliste mais pertinent. On pourrait par la suite ajouter toute une liste de métrique intéressantes, et constater leur évolution en suivant le même protocole.

