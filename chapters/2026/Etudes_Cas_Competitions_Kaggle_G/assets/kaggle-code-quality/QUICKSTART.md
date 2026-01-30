# Guide de Demarrage Rapide

## Pre-requis

- Python 3.8+
- pip

## Installation

```bash
cd kaggle-code-quality
pip install -r requirements.txt
```

## Etapes

### 1. Valider les evaluations LLM

```bash
python scripts/validate_llm_evaluations.py
```

Sorties :
- `data/validation_reports/validation_report.md`
- `data/validation_reports/validation_details.csv`

### 2. Agreger les resultats

```bash
python scripts/aggregate_evaluations.py
```

Sorties :
- `data/results/sq3_all_scores.csv`
- `data/results/sq3_summary_by_stratum.csv`
- `data/results/sq3_boxplot_by_stratum.png`
- `data/results/sq3_bar_mean_by_stratum.png`
- `data/results/sq3_heatmap_criteria_by_stratum.png`

### 3. Consulter les resultats

Ouvrir `book.md` pour le document complet avec methodologie, resultats et conclusion.

## Structure des donnees

Les evaluations sont dans `corpus/evaluations/<competition>/<strate>/<ref>.json`

Exemple de JSON :
```json
{
  "competition": "titanic",
  "stratum": "top_1%",
  "ref": "auteur/notebook",
  "scores_20": {
    "A_structure_pipeline": 20,
    "B_modularite": 15,
    "C_reproductibilite": 20,
    "D_lisibilite": 15,
    "E_hygiene": 15
  },
  "evidence": { ... },
  "summary": "..."
}
```

Le score total (85/100) est calcule automatiquement par le script Python.
