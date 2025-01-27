# Génération de Graphiques de Performance et de Consommation Énergétique pour les SLM en local

## Description

Le script `graph.py` analyse les résultats des inférences effectuées par différents modèles de langage et génère des graphiques illustrant les performances et la consommation énergétique moyenne par inférence.

## Fonctionnalités

Le script `graph.py` effectue les tâches suivantes :

1. Charge les résultats des inférences depuis des fichiers JSON situés dans le répertoire `../infer/results/`.
2. Calcule les métriques moyennes suivantes pour chaque modèle :
   - Mots par seconde (Words per Second)
   - Tokens par seconde (Tokens per Second)
   - Réponses par seconde (Responses per Second)
   - Puissance moyenne GPU (Average GPU Power)
   - Énergie GPU moyenne par inférence (Average GPU Energy per Inference)
   - Temps d'inférence moyen (Average Inference Time)
3. Génère des graphiques en barres pour chaque métrique et les enregistre dans le répertoire `results`.

## Graphiques Générés

Les graphiques suivants sont générés par le script :

- `words_per_sec.png` : Mots par seconde moyen par inférence pour chaque modèle.
- `tokens_per_sec.png` : Tokens par seconde moyen par inférence pour chaque modèle.
- `responses_per_sec.png` : Réponses par seconde moyen par inférence pour chaque modèle.
- `average_power.png` : Puissance moyenne GPU par inférence pour chaque modèle.
- `average_energy.png` : Énergie moyenne GPU par inférence pour chaque modèle.
- `inference_time.png` : Temps d'inférence moyen par inférence pour chaque modèle.

## Utilisation

Pour exécuter le script `graph.py` et générer les graphiques, assurez-vous d'avoir les dépendances nécessaires installées (`matplotlib`, `numpy`) et exécutez la commande suivante :

```sh
python graph.py
```

Les graphiques seront enregistrés dans le répertoire results.

Vous devez bien sûr avoir les fichiers de résultats des inférences dans le répertoire `../infer/results/`. Pour cela veuillez lire le README.md du dossier `infer`.

## Prérequis
- Python 3.x
- Bibliothèques Python : matplotlib, numpy

Pour installer les bibliothèques nécessaires, utilisez la commande suivante :
```sh
pip install matplotlib numpy
```

ou

```sh
pip install -r requirements.txt
```