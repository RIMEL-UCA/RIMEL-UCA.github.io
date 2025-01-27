# Récupération d'instructions

## Description

Ce projet contient un script pour récupérer des instructions à partir du dataset Alpaca en ligne et les stocker dans un fichier local `instructions.json`.

## Structure du projet

- `get_data.js`: Script permettant de récupérer les instructions depuis un fichier JSON en ligne et de les sauvegarder dans `instructions.json`.
- `instructions.json`: Fichier généré contenant les instructions récupérées.

## Utilisation

1. Exécutez le script `get_data.js` pour télécharger et traiter les données.
2. Les instructions seront sauvegardées dans le fichier `data/instructions.json`.

## Exemple

Pour exécuter le script, utilisez la commande suivante :

```sh
node get_data.js
```
