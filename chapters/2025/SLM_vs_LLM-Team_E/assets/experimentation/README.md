# Expérimentation de la Consommation Énergétique d'un SLM en Local

## 1. Introduction
Cette expérimentation a pour objectif de mesurer la consommation énergétique d'un **Small Language Model (SLM)** exécuté localement et de comparer ses performances avec celles établies dans l'étude *"[From Words to Watts: Benchmarking the Energy Costs of Large Language Model Inference](https://arxiv.org/pdf/2310.03003)" [1]*. Notre étude porte sur l'évaluation du coût énergétique et du temps d'inférence d'un SLM exécuté en local.

## 2. Méthodologie
L'expérimentation suit une démarche inspirée des travaux existants sur l'inférence des **Large Language Models (LLM)** dans le cloud [1].

### 2.1. Source de Données
- **Dataset :** **Alpaca**, conçu pour le fine-tuning de modèles de langage.
- **Nombre d’entrées :** 4096 instructions textuelles de type "question-réponse".
- **Instructions testées par modèle :** 256 (pour optimiser le temps d'exécution et l'utilisation des ressources).

### 2.2. Modèles Utilisés
L'expérimentation utilise **GPT4All**, avec les modèles suivants :
- **Llama 3.2 1B Instruct** (737 Mo, 2 Go de RAM, quantisation : q4_0, paramètres : 1B)
- **Llama 3.2 3B Instruct** (1.79 Go, 4 Go de RAM, quantisation : q4_0, paramètres : 3B)
- **Llama 3 8B Instruct** (4.34 Go, 8 Go de RAM, quantisation : q4_0, paramètres : 8B)

**Paramétrage des modèles :**
- Température : 0.7
- Taille maximale de sortie : 1024 tokens

### 2.3. Environnement et Configuration

#### Matériel
- **Machine :** MacBook Pro 16-inch 2021 (M1 Pro, 16 Go RAM, macOS Sequoia 15.2)

#### Logiciels
- **GPT4All** (version `3.7.0`), utilisant l'[API Metal](https://developer.apple.com/metal/) pour l'inférence.
- **`powermetrics`** pour mesurer la consommation énergétique via un utilitaire personnalisé (`powermetrics_wrapper`).
- **Node.js** (version `v20.18.0`) et **Python** (version `3.9.2`) pour l'orchestration et l'analyse des résultats.

Afin de garantir des mesures fiables, seuls les processus systèmes essentiels et ceux requis pour l'expérimentation sont actifs lors de l'exécution.

### 2.4. Mesures Effectuées
Les paramètres suivants sont analysés pour chaque modèle :
1. **Temps d’inférence**
2. **Taux de génération** (mots/s et tokens/s)
3. **Consommation énergétique** via `powermetrics_wrapper` :
   - Puissance GPU (W)
   - Énergie totale consommée (J)

### 2.5. Orchestration de l’Expérience
L’expérimentation repose sur plusieurs scripts JavaScript :
- **Script de préparation (`data`) :** Récupère les 4096 instructions du dataset Alpaca.
- **Script d’inférence (`infer`) :**
  - Charge les instructions en local.
  - Envoie les requêtes à l’API locale de GPT4All.
  - Mesure le temps de réponse et le taux de génération.
  - Collecte les données de consommation énergétique.
  - Stocke les résultats sous forme de fichiers JSON.

## 3. Résultats et Visualisation
Les résultats collectés sont analysés et visualisés sous forme de graphes via un script Python (`graph`) permettant de :
- Comparer **temps d’inférence vs consommation énergétique**.
- Analyser les performances en **mots/s et tokens/s**.
- Analyser la **puissance GPU** et l’**énergie totale consommée**.

## 4. Exécution de l'Expérimentation

### 4.1. Prérequis
Avant de démarrer, assurez-vous d’avoir :
- **GPT4All installé** avec les modèles mentionnés.
- **Node.js** installé pour exécuter les scripts JavaScript.
- **Accès à `powermetrics`** pour mesurer la consommation énergétique.

### 4.2. Lancement de l'Expérimentation
1. **Cloner le dépôt** sur votre machine.
2. **Récupérer les données Alpaca** (`data`).
3. **Démarrer l'API locale de `powermetrics`** (`powermetrics_wrapper`).
4. **Lancer GPT4All avec son API locale activée**.
5. **Exécuter `run.js` dans `infer`** pour démarrer l'expérimentation.
6. **Générer les graphiques** via le script Python (`graph`).

## 5. Résultats et Analyses
Les résultats et analyses comparatives avec l'étude [1] sont disponible dans le livre RIMEL à notre chapitre : [SLM vs LLM - Team E](https://rimel-uca.github.io/chapters/2025/SLM_vs_LLM-Team_E/content).

## 6. Perspectives et Améliorations
Pour aller plus loin, plusieurs pistes peuvent être explorées :
- Comparer des modèles d'autres familles.
- Tester diverses configurations (batch size, température).
- Augmenter le nombre d'instructions testées.
- Expérimenter avec d'autres frameworks (LM Studio, Transformers, etc.).
- Mesurer l'overhead de l'orchestration, de la collecte de données et des outils utilisés.
- Essayer avec un GPU externe pour évaluer les performances.

