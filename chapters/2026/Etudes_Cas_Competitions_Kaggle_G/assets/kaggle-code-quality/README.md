# Kaggle Code Quality - RIMEL 2025-2026

## Question de recherche

**La qualite du code varie-t-elle selon le rang au leaderboard Kaggle ?**

Cette etude analyse si les notebooks des competiteurs du podium (top 1%) sont mieux codes que ceux du milieu de classement (percentile 40-50%).

## Structure du projet

```
kaggle-code-quality/
|-- book.md                     # Document principal complet
|-- README.md                   # Ce fichier
|-- QUICKSTART.md               # Guide de demarrage rapide
|-- prompt_agent.txt            # Prompt LLM pour l'evaluation
|-- requirements.txt            # Dependances Python
|-- scripts/
|   |-- validate_llm_evaluations.py   # Validation des evaluations LLM
|   |-- aggregate_evaluations.py      # Agregation et visualisation
|-- corpus/
|   |-- agent_manifest.csv      # Liste des 40 notebooks
|   |-- evaluations/            # JSON des evaluations par competition/strate
|-- data/
|   |-- results/                # CSV et graphiques generes
|   |-- validation_reports/     # Rapports de validation
|-- docs/
    |-- validation_methodology.md  # Documentation de la validation
```

## Methodologie

1. **Constitution du corpus** : 40 notebooks repartis sur 10 competitions et 3 strates
2. **Evaluation automatisee** : LLM (Claude) evalue 5 criteres de qualite (0-20 chacun)
3. **Validation Python** : Script de validation multi-niveaux avec score de confiance
4. **Agregation** : Statistiques et visualisations par strate

## Criteres d'evaluation

| Critere | Description |
|---------|-------------|
| A) Structure & Pipeline | Organisation logique du code |
| B) Modularite | Code factorise en fonctions/classes |
| C) Reproductibilite | Seeds fixes, CV documentee |
| D) Lisibilite | Documentation, nommage explicite |
| E) Hygiene | Proprete, gestion des erreurs |

## Resultats cles

- **39/39 evaluations valides** (score de confiance moyen : 99.7/100)
- **Pas de correlation significative** entre rang et qualite du code
- Les 2 meilleurs notebooks (95/100) sont dans la strate p40_50, pas top_1%

## Utilisation

```bash
# 1. Installer les dependances
pip install -r requirements.txt

# 2. Valider les evaluations LLM
python scripts/validate_llm_evaluations.py

# 3. Agreger et generer les visualisations
python scripts/aggregate_evaluations.py
```

## Auteur(s)

- RIMEL 2025-2026
