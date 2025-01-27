# Serveur de Mesures Énergétiques avec `powermetrics` pour macOS

## Description

Ce programme est un serveur Express qui utilise la commande `powermetrics` de macOS pour collecter des données énergétiques sur la batterie, le CPU et le GPU. Les données sont ensuite exposées via une API REST.

## Fonctionnalités

Le programme `index.ts` effectue les tâches suivantes :

1. Lance la commande `powermetrics` avec un intervalle de mise à jour de 100 ms pour collecter des données énergétiques en format XML.
2. Parse les données XML pour extraire les informations suivantes :
   - Pourcentage de batterie (`batteryPercentage`)
   - Puissance CPU en milliwatts (`cpuPowerMw`)
   - Puissance GPU en milliwatts (`gpuPowerMw`)
   - Puissance combinée en milliwatts (`combinedPowerMw`)
3. Stocke les dernières mesures dans un buffer.
4. Expose une API REST à l'endpoint `/metrics` pour récupérer les dernières mesures énergétiques.

## Installation

1. Assurez-vous d'avoir Node.js et npm installés sur votre machine.
2. Clonez ce dépôt et naviguez dans le répertoire du projet.
3. Installez les dépendances avec la commande suivante :

```sh
npm install
```

## Utilisation
Pour lancer le serveur en mode développement, exécutez la commande suivante :

```sh
sudo npm dev
```

Le serveur démarrera et commencera à collecter des données énergétiques. Vous pouvez accéder aux mesures en envoyant une requête GET à l'endpoint suivant :

```
http://localhost:3033/metrics
```

Vous pouvez aussi build le projet en mode production et le lancer avec la commande suivante :

```sh
npm run build
npm run start
```

## Exemple de Réponse
Voici un exemple de réponse JSON de l'endpoint /metrics :

```json
{
    "batteryPercentage": 100,
    "cpuPowerMw": 1052.81,
    "gpuPowerMw": 11818.9,
    "combinedPowerMw": 12871.7
}
```
