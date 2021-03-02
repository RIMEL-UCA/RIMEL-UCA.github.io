# Issue finder for Github

Le but de ce programme est de retrouver toutes les issues d'un projet à partir d'un ensemble de mots clés.

Ce programme va se charger de générer 3 fichiers csv nommés **all_tickets.csv**, **closed_tickets.csv**, **open_tickets.csv** contenant respectivement la liste de tous les tickets trouvés, la liste des tickets fermés et la liste des tickets ouverts.

Sur chacun de ces fichiers csv, on retrouve donc une liste de tickets avec diverses informations comme notamment le titre du ticket, l'id associé au ticket, l'information par un 1 ou un 0 si le ticket contient des logs, son nombre de commits et son nombre de commentaires.

Avoir ces informations sous format csv permet de les afficher via un outil tel que Excel et de pouvoir calculer des moyennes et en extraire des graphiques.

## Personnaliser le programme

Il y a différentes informations que vous pouvez modifier dans le programme.

Tout d'abord vous devez spécifier votre [token](https://github.com/settings/tokens) GitHub afin d'avoir une limite de requêtes par minute plus souple.

```javascript
// changez YOUR_GITHUB_TOKEN par votre token
const gh = new GitHub({
    token: 'YOUR_GITHUB_TOKEN'
});
```

Vous pouvez aussi personnaliser les mots clés de recherche :

```javascript
//propriétaire du répertoire git
const user = "angular";

//nom du repertoire
const repo = "angular";

//mots clés de recherche
const keyWords = "bug report";
```
*L'extraction des logs est adaptée selon les [spécifications](https://github.com/angular/angular/blob/master/CONTRIBUTING.md) des tickets de bug du projet Angular et risque de ne pas fonctionner avec un autre projet.*

## Exécuter le code

Au préalable vous devez avoir **Node.js** avec **NPM** d'installer sur votre machine.

```bash
# Permet d'installer les dépendances du projet
npm install

# Exécute le code --> prend un certain temps pour extraire tous les tickets
node index.js
```