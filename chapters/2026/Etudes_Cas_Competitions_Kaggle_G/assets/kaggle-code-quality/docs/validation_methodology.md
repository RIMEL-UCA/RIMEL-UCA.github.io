# Methodologie de Validation des Evaluations LLM

## Contexte

Suite a une critique lors de la presentation ("Comment peut-on faire confiance aux scores generes par le LLM ?"), nous avons mis en place un pipeline de validation multi-niveaux.

## Principe de Separation des Responsabilites

### Couche Semantique (LLM)
- Analyse qualitative du code
- Identification des patterns architecturaux
- Attribution des 5 scores individuels (A-E)
- Generation de preuves textuelles

### Couche Calculatoire (Python)
- Calcul du score_total_100 = A + B + C + D + E
- Validation structurelle
- Validation des valeurs
- Detection d'anomalies
- Generation de metriques de confiance

## Architecture de Validation en 5 Couches

### Couche 1 : Validation Structurelle
- Tous les champs JSON requis presents
- Types de donnees corrects (dict, int)

### Couche 2 : Validation des Valeurs
- Scores dans {0, 5, 10, 15, 20}
- Pas de valeurs hors limites
- 5 criteres presents (A-E)

### Couche 3 : Calcul Automatique
- score_total_100 = A + B + C + D + E
- Detection d'erreurs arithmetiques du LLM

### Couche 4 : Detection d'Anomalies
- Tous scores identiques (evaluation non nuancee)
- Evaluation binaire (que 0 ou 20)
- Scores extremes (0/100 ou 100/100)
- Preuves manquantes ou vides

### Couche 5 : Score de Confiance
- Metrique agregee : 100 - (30 x erreurs) - (10 x warnings)
- Permet d'identifier les evaluations a revoir

## Score de Confiance

Pour chaque notebook, un score de confiance (0-100) est calcule :
- 100 = Aucune erreur ni warning
- -30 points par erreur critique
- -10 points par warning

Les evaluations avec un score < 50 doivent etre regenerees ou revues manuellement.

## Resultats de Validation

Sur 39 notebooks evalues :
- Evaluations valides : 39/39 (100%)
- Score de confiance moyen : 99.7/100
- Erreurs de calcul LLM : 0
- Warnings : 1

## Utilisation

```bash
python scripts/validate_llm_evaluations.py
```

Sorties :
- `data/validation_reports/validation_report.md`
- `data/validation_reports/validation_details.csv`
