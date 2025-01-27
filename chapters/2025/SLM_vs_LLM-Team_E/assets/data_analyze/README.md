# README

## Mise en place de l'environnement

Avant toutes choses, il est nécessaire d'installer les dépendances du projet. Pour cela, il suffit de lancer la commande suivante :

```bash
pip install -r requirements.txt
```

Aussi, il est nécessaire d'avoir les bonnes données pour les scripts.
Nous avons pu télécharger ces données depuis le repository [llm-perf-leaderboard](https://huggingface.co/spaces/optimum/llm-perf-leaderboard/tree/main). Malheuresment, ce repository a été mis à jour et ne permet plus de télécharger les données. Nous avons donc mis les données nécessaires dans le dossier `data`. Sinon, nous pouvons toujours utiliser ce repository pour accéder à l'API de HuggingFace mais les scripts utilisés ici devront être modifiés pour utiliser directement l'API.

## Description des données

Il y a quatre fichiers différents :

- [input-32vCPU.csv](data/input-32vCPU.csv)
- [input-A10.csv](data/input-A10.csv)
- [input-A100.csv](data/input-A100.csv)
- [input-T4.csv](data/input-T4.csv)

Ces fichiers contiennent différentes informations sur des modèles LLM (Large Language Models) et SLM (Small Language Models) qui ont été exécutés sur plusieurs unités de calculs différentes. Les colonnes incluent des informations telles que le nombre de paramètres, le coût énergétique, la mémoire utilisée, et d'autres métriques de performance.

- [leaderboard.csv](data/leaderboard.csv)

Ce fichier contient des informations provenant d'un autre classement de modèles LLM et SLM. Les colonnes incluent des informations telles que le nombre de paramètres, le coût en CO₂, le coût en dollars, et d'autres métriques de performance. Ces informations sont tirées de [open-llm-leaderboard](https://huggingface.co/datasets/open-llm-leaderboard/contents).

- [financier.csv](data/financier.csv)

Ce fichier contient des informations sur le coût en dollars pour 1K tokens en fonction du nombre de paramètres des modèles. Nous avons pu éxtraire ces informations de deux ressources différentes. La première, [arxiv](https://arxiv.org/abs/2312.14972), qui nous donne des informations sur les SLM et la deuxième, [llmpricecheck](https://llmpricecheck.com/), qui nous donne des informations sur les LLM.

## Description des scripts

Ce dépôt contient trois scripts Python : `co2.py`, `energy.py` et `finance.py`. Chacun de ces scripts lit des fichiers CSV en entrée, effectue des calculs et génère des graphiques pour visualiser les résultats.

### 1. `co2.py`

Ce script lit un fichier CSV contenant des informations sur les modèles de chat et leur coût en CO₂. Il filtre les données pour ne conserver que les modèles de chat, effectue une régression linéaire et génère un graphique montrant la relation entre le nombre de paramètres et le coût en CO₂.

#### Utilisation

```bash
python scripts/co2.py
```

### 2. `energy.py`

Ce script lit plusieurs fichiers CSV contenant des informations sur l'énergie consommée par million de tokens pour différents modèles. Il combine les données, calcule les moyennes et génère un graphique montrant la relation entre le nombre de paramètres et l'énergie consommée.

#### Utilisation

Pour exécuter le script, utilisez la commande suivante :

```bash
python scripts/energy.py
```

### 3. `finance.py`

Ce script lit un fichier CSV contenant des informations sur le coût en dollars pour 1K tokens en fonction du nombre de paramètres des modèles. Il effectue une régression linéaire et génère un graphique montrant la relation entre le nombre de paramètres et le coût.

#### Utilisation

Pour exécuter le script, utilisez la commande suivante :

```bash
python scripts/finance.py
```




