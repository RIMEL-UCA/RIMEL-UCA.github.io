# Exécution des Instructions et Collecte des Données Énergétiques

## Description

Le fichier `run.js` exécute des instructions sur différents modèles de langage hébergé en local et tournant sur GPT4All.
Il mesure le temps d'inférence, collecte les données énergétiques et enregistre les résultats dans des fichiers JSON.

## Fonctionnalités

Le programme `run.js` effectue les tâches suivantes :

1. Charge les instructions depuis un fichier JSON.
2. Envoie les instructions à différents modèles de langage via l'API GPT4All.
3. Mesure le temps d'inférence pour chaque instruction.
4. Collecte les données énergétiques toutes les 100 ms pendant l'inférence en utilisant l'API locale `powermetrics_wrapper`.
5. Enregistre les résultats, y compris les temps d'inférence et les données énergétiques, dans des fichiers JSON distincts pour chaque modèle.

## Installation

1. Assurez-vous d'avoir Node.js et npm installés sur votre machine.
2. Clonez ce dépôt et naviguez dans le répertoire du projet.
3. Installez les dépendances avec la commande suivante :

```sh
npm install
```

## Utilisation
Pour exécuter le script `run.js` et collecter les données, utilisez la commande suivante :

```sh
node run.js
```

Veillez à avoir le programme `powermetrics_wrapper` en cours d'exécution pour collecter les données énergétiques.

## Exemple de Réponse
Voici un exemple de structure de fichier JSON généré :

```json
[
  {
    "inferenceTime": 2366,
    "input": "Give three tips for staying healthy.",
    "inputLength": 36,
    "inputToken": 8,
    "output": "Here the model generates a response.",
    "outputLength": 1449,
    "outputToken": 270,
    "energyMetrics": [
      {
        "batteryPercentage": 100,
        "cpuPowerMw": 889.996,
        "gpuPowerMw": 16863.1,
        "combinedPowerMw": 17753.1,
        "timeElapsed": 0
      },
      {
        "batteryPercentage": 100,
        "cpuPowerMw": 1052.81,
        "gpuPowerMw": 11818.9,
        "combinedPowerMw": 12871.7,
        "timeElapsed": 100
      }
    ]
  }
]
```