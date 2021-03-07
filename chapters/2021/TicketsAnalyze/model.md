---
layout: default
title : Dans quelle mesure les tickets nous renseignent sur l’architecture et l’organisation du développement d’un système ?
date:   2021-01-03 22:00:00 +0100
---

---

> **Date de rendu finale : Mars 2021 au plus tard**
> - Respecter la structure pour que les chapitres soient bien indépendants
> - Remarques :
>>    - Les titres peuvent changer pour être en adéquation avec votre étude.
>>    - De même il est possible de modifier la structure, celle qui est proposée ici est là pour vous aider.
>>    - Utiliser des références pour justifier votre argumentaire, vos choix etc.

---

**_janvier 2021_**

## Authors

We are four students in last year of Polytech' Nice-Sophia specialized in Software Architecture :

* Rania Fekih &lt;rania.fekih@etu.unice.fr&gt;
* Anis Khalili &lt;anis.khalili@etu.unice.fr&gt;
* Gabriel Revelli &lt;gabriel.revelli@etu.unice.fr&gt;
* Robin Lambert &lt;robin.lambert@etu.unice.fr&gt;

## I. Contexte de recherche 
Comme chaque développeur, lorsqu’on est intégré à un projet qui est en cours de développement ou bien avancé, nous trouvons des difficultés à comprendre la structure de milliers de fichiers ainsi que le mode de fonctionnement au sein de l’équipe. Les tickets sont au centre de l’organisation de tout projet, c’est pourquoi nous allons étudier les tickets pour essayer de tirer des conclusions sur l’architecture d’un projet.
Cette étude pourrait nous permettre d’avoir une meilleure vision de structure des gros projets, nous avons pris comme exemple des projets open source où nous avons accès aux tickets pour trouver des informations sur la manière dont les projets sont découpés (modules, composants, microservices...).

___

## II. Problématiques : 
Dans toutes les entreprises la période d’adaptation d’un nouvel ingénieur ou d’un ingénieur qui change de projet est estimée entre 2 et 3 semaines. Qui est une perte de temps et d’argent. C’est par là que notre question générale a été inspirée “dans quelle mesure les tickets nous renseignent sur l’architecture et l’organisation du développement d’un système ?”. 
Pour répondre à cette problématique nous avons décomposé notre problème en 4 sous problème : 
### 1. Comment peut-on distinguer les composants les plus importants dans l’architecture à partir des tickets ?
La première étape pour découvrir la composition d’un système est déjà de connaître ses composants principaux. Nous avons commencé par rassembler les tickets par composants vu que nous avons considéré nécessaire d’avoir une visualisation générale sur les composants existants avant de commencer à rentrer plus dans les détails. C’est une étape qu’on considère nécessaire pour avoir une vision globale sur leur impact dans le système général. 
### 2.Une fois qu’on trouve les composants qui ont des tickets en commun, peut-on identifier un couplage ?
Une partie que nous considérons nécessaire dans le processus de compréhension et de découverte de l’architecture d’un système est le couplage entre composants. Analyser les tickets en communs entre composants peut éventuellement nous aider à poser des hypothèses sur le couplage et la relation entre les composants en question. 
### 3. Comment peut-on identifier les composants qui sont en évolution ?
Après avoir une idée générale sur l’architecture globale de l’application, c’est intéressant de voir l’évolution des composants dans les dernières années qui sera généré à partir du nombre de fonctionnalités ajoutées par composant.

### 4. Comment peut-on évaluer l’implémentation d’un composant ?
Enfin, il est intéressant de se focaliser sur un composant donné. Consulter les différents types de tickets existants et comparer les chiffres par rapport à la taille de ce composant. Lors de l’investigation nous avons trouvé que les types de tickets qui peuvent nous renseigner au plus sur la qualité d’un composant sont les tickets de type bug et les tickets de type improvement. 

___
##  III. Collecte d’informations
Au début du projet nous avons commencé par explorer le ticketing du projet mongodb hébergé sur Jira. Nous avons choisi ce projet parce qu'il est open-source et qu’il dispose d’une grande variété de tickets à analyser. Ce projet contient également 9 sous-projets qui ont tous leurs propres composants. 
Les outils utilisés sont :
La documentation de l’API Jira
Le projet Firefox sur bugzilla 
La thèse “Evolution structurelle dans les architecture logicielle à base de composants” publié sur HAM le site des archives ouvertes
Le projet mongoDb sur jira
Nous avons également créé deux programmes qui permettent d’analyser le gestionnaire de ticket Jira. Pour l’élaboration de ces programmes nous avons eu besoin de l’API de Jira pour récupérer des informations sur les tickets d’un projet et également de la librairie d3-node sur nodeJS pour afficher des graphiques utiles à l’analyse de ces tickets.
___
## IV. Hypothèses & Experiences


### 1.Comment peut-on distinguer les composants les plus importants dans l’architecture  ?

#### Hypothèse 

Notre première intuition c’est que les composants qui ont plus de ticket sont des composants de valeur importante dans un projet .

#### Expérimentation

Pour vérifier cette hypothèse nous avons expérimenté ceci à travers notre propre serveur d’analyse qu’on a développé en passant dans le coeur de la requête 
le nom de l’API et du projet sur lequel on va faire notre expérimentation:
```json
{
 "APIJira" : "https://jira.mongodb.org/rest/api/2/search",
 "ProjectName":"SERVER"
}
```

``` FILTER : APIJira+"?jql=" "project = "+ProjectName +" AND component is not EMPTY"; ```



### 2.Une fois qu’on trouve les composants qui ont des tickets en commun, peut-on identifier un couplage ?

#### Hypothèse n°1 
Nous avons l'intuition que si deux composants  partagent un nombre assez important de tickets, ils peuvent éventuellement être couplés
#### Expérimentation
Pour vérifier intuitivement notre hypothèse, nous avons utilisé l’API de jira pour récupérer les tickets qui sont partagés entre deux composants, et nous avons commencé par analyser à la main les “include” dans les fichiers du code source du projet associés au lien github. L’objectif était de vérifier à la main dans un premier temps s' il était possible d’identifier du couplage entre les composants en regardant le code.

``` FILTER : "project = "+ProjectName +" AND component="+ principalComponent+ " AND component in ("+componentsNames+")"; ```

Après la récupération des tickets,  nous avons récupéré les liens github associés et nous les avons passés à notre algorithme pour analyser et extraire les includes qui existent. 
Dans les informations d’un ticket nous avons une liste de composants liés à ce ticket. Nous allons donc essayer de prouver réellement que le fait de citer plusieurs composants dans un même ticket indiquent qu’ils sont couplés. 

Nous nous sommes rendus compte que nous ne savions pas à quel composant appartiennent les fichiers du code source (le code n’est pas organisé de façon à différencier clairement les composants). 

Nous avons trouvé une solution efficace pour résoudre ce problème: récupérer les liens Github en descriptions des tickets vers les fichiers du code source, et associer ces fichiers au composant du ticket.

Une fois que nous aurons établi un lien entre les chemins des fichiers du code source et les composants du projet, nous pourrons identifier, dans les imports, les composants associés à ces fichiers, et valider notre hypothèse de couplage. 

Nous ne pouvons évidemment utiliser cette technique que sur les tickets liés à un seul composant.

Ensuite nous récupérons tous les tickets qui référencent deux composants avec des liens Github présents. On cherche, grâce aux données recueillies à la phase précédente, à associer le fichier à un composant cité par le ticket, ensuite on vérifie dans ses imports qu’il est bien associé aux autres composants et si c’est le cas on peut parler d’un couplage. Cela permet de nous donner un bon indicateur de couplage ou non.

#### Hypothèse n°2

Si deux composants sont liés à plusieurs reprises à des tickets de type bugs avec priorité P1 ou P2, le couplage entre ces deux composants peut être nommé couplage fort.

#### Expérimentation

Pour vérifier cette hypothèse nous avons fait exactement la même démarche qu’auparavant. Mais malheureusement les includes ne sont pas suffisants pour définir un couplage fort. Donc nous avons décidé d’éliminer cette hypothèse. 

### 3.Après avoir une idée sur les différents composants, comment peut-on identifier les composants qui sont en évolution ?

#### Hypothèse n°1
Nous avons l’intuition que les composants qui ont le plus de tickets de type new feature sont les composants de système qui sont actuellement en état d’évolution. 

#### Expérimentation
Pour vérifier cette hypothèse, nous avons demandé à notre entourage et lu quelques articles pour comprendre au plus l’utilisation des du label new-feature. Tout le monde a été d’accord que c’est un label qui est utilisé à l’ajout d’une nouvelle fonctionnalité à un composant. 

### 4.Comment peut-on évaluer l’implémentation des composants ?

#### Hypothèse n°1

Notre première hypothèse c’est qu' un composant qui a plusieurs tickets de type bugs , la qualité de son code peut être considérée comme non optimale. 

#### Expérimentation

Pour vérifier cette hypothèse , nous allons comparer par année les composants qui ont plus de bugs vs les composants qui ont le plus de tickets de type improvement . Nous allons nous focaliser seulement sur les deux dernières années. Donc pour prouver cela la première étape était d’appliquer les filtres ci-dessous pour récupérer le nombre de tickets bug pour chaque composants du projet “server”.

``` FILTER  : APIJira+"?jql=""project = "+ProjectName +" AND issuetype = Bug AND component is not EMPTY AND created >= \"2019/01/01 00:00\" AND created < \"2019/12/31 23:59\""; ```

``` FILTER  : APIJira+"?jql=""project = "+ProjectName +" AND issuetype = Bug AND component is not EMPTY AND created >= \"2020/01/01 00:00\" AND created < \"2020/12/31 23:59\""; ```

Après, nous avons fait des expérimentations sur le même projet en modifiant les le type par improvement.

``` FILTER :  APIJira+"?jql="project = "+ProjectName +" AND issuetype = Improvement AND component is not EMPTY AND created >= \"2019/01/01 00:00\" AND created < \"2019/12/31 23:59\"" ```

``` FILTER : APIJira+"?jql="project = "+ProjectName +" AND issuetype = Improvement AND component is not EMPTY AND created >= \"2020/01/01 00:00\" AND created < \"2020/12/31 23:59\"" ```
___
## V. Result Analysis and Conclusion Tout le monde 

### Comment peut-on distinguer les composants les plus importants dans l’architecture ?

Pour répondre à l’hypothèse “le composant qui a le plus de tickets est le composant le plus important” nous avons commencé par appliquer le filtre qu’on a précisé auparavant au projet SERVER de mongodb et nous avons dégagé ce graph : 


