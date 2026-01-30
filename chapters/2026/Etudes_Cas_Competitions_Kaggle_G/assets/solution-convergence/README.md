# Guide de Reproductibilité

## Structure du Projet

```
competitions/
  <COMPETITION_NAME>/
    solutions/                    # 10 solutions Kaggle (top 5 + 5 du top 100+)
    iteration-1/
      filled-structure/
        raw/                      # Solutions remplies par le LLM (topX.json)
        normalized/               # Solutions normalisées (topX.json)
        field-completion-ratio.txt
      justification/              # Justifications du LLM (topX-justification.json)
      improvement.json            # Propositions d'amélioration (si champs à 0%)
    iteration-N/                  # Itérations suivantes (même structure)
    final-iteration/
      structure-converted.json    # Schéma booléen final
      removed-field.txt           # Champs supprimés manuellement
      filled-structure/           # Solutions booléennes normalisées
initial-structure.json            # Schéma JSON initial, commun à toutes les itération-1
prompts/
  instructions_basic.md           # Prompt pour première itération
  instructions_enrich.md          # Prompt pour itérations suivantes
  instructions_improvements.md    # Prompt pour améliorer le schéma
scripts/
  normalize/normalize_json.py
  field-completion-ratio/field_completion_ratio.py
  completion-ratio/compare_completion.py
  list-to-boolean/generate_boolean_schema.py
  similarity/compare_solutions.py
```

## Outils Disponibles

- **normalize_json.py** : Normalise les extractions LLM selon le schéma
- **field_completion_ratio.py** : Analyse le taux de complétion par champ
- **compare_completion.py** : Compare deux itérations
- **generate_boolean_schema.py** : Convertit le schéma liste en schéma booléen
- **compare_solutions.py** : Calcule les scores de similarité et génère une heatmap

## Procédure de Reproduction

### Itération 1 : Extraction Initiale

**1. Extraire les caractéristiques avec le LLM**

Pour chaque solution (top1 à top5, puis 5 solutions du top 100+), prompter Claude Opus 4.5 :

```
Strictly follow this instruction file for the topX solution of the COMPETITION_NAME competition
```

Avec en contexte :
- `prompts/instructions_basic.md`
- `competitions/COMPETITION_NAME/iteration-1/structure.json`

Le LLM génère deux fichiers :
- `topX.json` → placer dans `iteration-1/filled-structure/raw/`
- `topX-justification.json` → placer dans `iteration-1/justification/`

**Vérifier** pour chaque solution le fichier `topX-justification.json` pour détecter les hallucinations ou erreurs d'extraction.

**2. Normaliser les solutions**

```powershell
python scripts/normalize/normalize_json.py `
  competitions/COMPETITION_NAME/iteration-1/structure.json `
  competitions/COMPETITION_NAME/iteration-1/filled-structure/raw `
  competitions/COMPETITION_NAME/iteration-1/filled-structure/normalized
```

**3. Analyser le taux de complétion**

```powershell
python scripts/field-completion-ratio/field_completion_ratio.py `
  competitions/COMPETITION_NAME/iteration-1/filled-structure/normalized
```

Le rapport dans `iteration-1/filled-structure/field-completion-ratio.txt` liste chaque champ avec son taux de complétion (0% à 100%).

### Branchement : Champs à 0% ?

####  Cas A : Champs à 0%  Améliorer le Schéma

**1. Générer un schéma amélioré**

Prompter Claude Opus 4.5 :

```
Strictly follow this instruction file
```

Avec en contexte :
- `prompts/instructions_improvements.md`
- `iteration-1/filled-structure/field-completion-ratio.txt`

Le LLM génère :
- `improvement.json` (explications des changements)
- `improved_structure.json` (nouveau schéma)

**2. Vérifier manuellement**

Valider que `improved_structure.json` :
- Remplace uniquement les champs à 0%
- Ne modifie/supprime aucun autre champ

Utiliser `improvement.json` pour confirmer la justification de chaque changement proposé.

**3. Nettoyer les solutions normalisées**

```powershell
python scripts/normalize/normalize_json.py `
  competitions/COMPETITION_NAME/iteration-1/improved_structure.json `
  competitions/COMPETITION_NAME/iteration-1/filled-structure/normalized `
  competitions/COMPETITION_NAME/iteration-2/filled-structure/normalized
```

Ce script supprime les champs à 0% des solutions existantes et ajoute les nouveaux champs vides. Créer `iteration-2/structure.json` en copiant `improved_structure.json`.

**4. Retour à l'Itération N+1**

Reprendre l'extraction (étape 1) avec :
- Prompt : `prompts/instructions_enrich.md`
- Schéma : `iteration-N/structure.json`

####  Cas B : Aucun Champ à 0%  Conversion Booléenne

**1. Générer le schéma booléen**

```powershell
python scripts/list-to-boolean/generate_boolean_schema.py `
  competitions/COMPETITION_NAME/iteration-N/filled-structure/normalized `
  competitions/COMPETITION_NAME/final-iteration/structure-converted.json
```

Ce script analyse toutes les valeurs extraites dans les listes de toutes les solutions de la dernière itération et génère un schéma où chaque valeur devient un champ booléen.

**2. Nettoyer les doublons**

Vérifier manuellement `structure-converted.json` pour les champs redondants :
- Exemple : `transformer_models.mlm_pretraining` et `pretraining_methods.mlm_pretraining`
- Documenter chaque suppression dans `final-iteration/removed-field.txt`

**3. Extraire avec le schéma booléen**

Reprendre l'extraction (étape 1) avec :
- Prompt : `prompts/instructions_basic.md` (ou `instructions_enrich.md`)
- Schéma : `final-iteration/structure-converted.json`
- Destination : `final-iteration/filled-structure/raw/`

Puis normaliser :

```powershell
python scripts/normalize/normalize_json.py `
  competitions/COMPETITION_NAME/final-iteration/structure-converted.json `
  competitions/COMPETITION_NAME/final-iteration/filled-structure/raw `
  competitions/COMPETITION_NAME/final-iteration/filled-structure/normalized
```

**4. Calculer les similarités**

```powershell
python scripts/similarity/compare_solutions.py `
  competitions/COMPETITION_NAME/final-iteration/filled-structure/normalized
```

Résultats générés :
- `similarity_heatmap.svg` : visualisation matricielle des scores de similarité
- `metrics_report.txt` : métriques intra-groupe (top 5 vs top 5) et inter-groupes (top 5 vs top 100+)