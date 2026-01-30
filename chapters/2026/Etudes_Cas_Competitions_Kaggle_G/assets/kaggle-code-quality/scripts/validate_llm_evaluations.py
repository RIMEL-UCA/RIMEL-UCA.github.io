#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation des evaluations LLM pour la qualite du code Kaggle.

Ce script implemente une architecture de validation en 5 couches :
1. Validation structurelle (champs JSON requis)
2. Validation des valeurs (scores dans {0, 5, 10, 15, 20})
3. Calcul automatique du score total (Python, pas LLM)
4. Detection d'anomalies (patterns suspects)
5. Score de confiance (metrique agregee)

Utilisation :
    python scripts/validate_llm_evaluations.py

Sorties :
    - data/validation_reports/validation_report.md
    - data/validation_reports/validation_details.csv
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import csv

# Configuration - chemins relatifs au dossier kaggle-code-quality
SCRIPT_DIR = Path(__file__).parent.parent  # Remonte de scripts/ vers kaggle-code-quality/
EVAL_DIR = SCRIPT_DIR / "corpus" / "evaluations"
REPORT_DIR = SCRIPT_DIR / "data" / "validation_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Criteres attendus
EXPECTED_CRITERIA = [
    "A_structure_pipeline",
    "B_modularite",
    "C_reproductibilite",
    "D_lisibilite",
    "E_hygiene"
]

# Valeurs de score valides
VALID_SCORES = {0, 5, 10, 15, 20}


@dataclass
class ValidationResult:
    """Resultat de validation pour un notebook."""
    json_path: str
    competition: str = ""
    stratum: str = ""
    ref: str = ""
    
    # Scores extraits
    scores: dict = field(default_factory=dict)
    computed_total: int = 0
    llm_total: Optional[int] = None
    
    # Validation
    is_valid: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    
    # Confiance
    trust_score: int = 100
    
    def add_error(self, message: str):
        """Ajoute une erreur critique."""
        self.errors.append(message)
        self.is_valid = False
        self.trust_score = max(0, self.trust_score - 30)
    
    def add_warning(self, message: str):
        """Ajoute un warning."""
        self.warnings.append(message)
        self.trust_score = max(0, self.trust_score - 10)
    
    def get_status(self) -> str:
        """Retourne le statut de validation."""
        if not self.is_valid:
            return "INVALIDE"
        elif self.warnings:
            return "WARNING"
        else:
            return "VALIDE"


def load_json(path: Path) -> Optional[dict]:
    """Charge un fichier JSON."""
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        return None


def validate_structure(data: dict, result: ValidationResult) -> None:
    """Couche 1 : Validation structurelle."""
    
    # Champs requis
    required_fields = ["competition", "stratum", "ref", "scores_20", "evidence"]
    for field in required_fields:
        if field not in data:
            result.add_error(f"Champ manquant: {field}")
    
    # Types
    if "scores_20" in data and not isinstance(data["scores_20"], dict):
        result.add_error("scores_20 doit etre un dictionnaire")
    
    if "evidence" in data and not isinstance(data["evidence"], dict):
        result.add_error("evidence doit etre un dictionnaire")
    
    # Extraction des metadonnees
    result.competition = data.get("competition", "")
    result.stratum = data.get("stratum", "")
    result.ref = data.get("ref", "")


def validate_scores(data: dict, result: ValidationResult) -> None:
    """Couche 2 : Validation des valeurs de scores."""
    
    scores_20 = data.get("scores_20", {})
    if not isinstance(scores_20, dict):
        return
    
    # Verifier les 5 criteres
    for criterion in EXPECTED_CRITERIA:
        if criterion not in scores_20:
            result.add_error(f"Critere manquant: {criterion}")
        else:
            score = scores_20[criterion]
            
            # Verifier le type
            if not isinstance(score, (int, float)):
                result.add_error(f"{criterion}: score non numerique ({score})")
                continue
            
            score = int(score)
            
            # Verifier la valeur
            if score not in VALID_SCORES:
                result.add_error(f"{criterion}: score invalide {score} (attendu: 0, 5, 10, 15, 20)")
            else:
                result.scores[criterion] = score
    
    # Criteres supplementaires (non attendus)
    for criterion in scores_20:
        if criterion not in EXPECTED_CRITERIA:
            result.add_warning(f"Critere inattendu: {criterion}")


def compute_total(result: ValidationResult) -> None:
    """Couche 3 : Calcul automatique du score total."""
    
    if len(result.scores) == 5:
        result.computed_total = sum(result.scores.values())
    else:
        result.computed_total = 0


def check_llm_total(data: dict, result: ValidationResult) -> None:
    """Verifie si le LLM a fourni un score total (ne devrait pas)."""
    
    if "score_total_100" in data:
        llm_total = data["score_total_100"]
        if isinstance(llm_total, (int, float)):
            result.llm_total = int(llm_total)
            
            # Comparer avec le calcul Python
            if result.computed_total != result.llm_total:
                result.add_warning(
                    f"Ecart LLM vs Python: LLM={result.llm_total}, Python={result.computed_total}"
                )


def detect_anomalies(data: dict, result: ValidationResult) -> None:
    """Couche 4 : Detection d'anomalies."""
    
    if len(result.scores) < 5:
        return
    
    scores_list = list(result.scores.values())
    
    # Anomalie 1 : Tous les scores identiques
    if len(set(scores_list)) == 1:
        result.add_warning(f"Tous les scores identiques ({scores_list[0]}) - evaluation peu nuancee")
    
    # Anomalie 2 : Evaluation binaire (que 0 et 20)
    unique_scores = set(scores_list)
    if unique_scores.issubset({0, 20}):
        result.add_warning("Evaluation binaire (que 0 et 20) - manque de nuance")
    
    # Anomalie 3 : Score total extreme
    if result.computed_total == 0:
        result.add_warning("Score total = 0 - verifier si intentionnel")
    elif result.computed_total == 100:
        result.add_warning("Score total = 100 - evaluation parfaite rare")
    
    # Anomalie 4 : Preuves manquantes
    evidence = data.get("evidence", {})
    if isinstance(evidence, dict):
        for criterion in EXPECTED_CRITERIA:
            if criterion not in evidence or not evidence[criterion]:
                result.add_warning(f"Preuve manquante pour {criterion}")


def validate_notebook_evaluation(json_path: Path) -> ValidationResult:
    """Valide une evaluation de notebook."""
    
    result = ValidationResult(json_path=str(json_path))
    
    # Charger le JSON
    data = load_json(json_path)
    if data is None:
        result.add_error("Impossible de parser le JSON")
        return result
    
    # Couche 1 : Structure
    validate_structure(data, result)
    
    # Couche 2 : Valeurs
    validate_scores(data, result)
    
    # Couche 3 : Calcul total
    compute_total(result)
    
    # Verifier si LLM a fourni un total
    check_llm_total(data, result)
    
    # Couche 4 : Anomalies
    detect_anomalies(data, result)
    
    return result


def generate_validation_report(results: list) -> None:
    """Genere le rapport de validation en Markdown."""
    
    valid_count = sum(1 for r in results if r.is_valid)
    warning_count = sum(1 for r in results if r.is_valid and r.warnings)
    invalid_count = sum(1 for r in results if not r.is_valid)
    
    avg_trust = sum(r.trust_score for r in results) / len(results) if results else 0
    
    lines = [
        "# Rapport de Validation des Evaluations LLM",
        "",
        f"Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Statistiques Globales",
        "",
        "| Metrique | Valeur |",
        "|----------|--------|",
        f"| Total evaluations | {len(results)} |",
        f"| Valides | {valid_count} |",
        f"| Avec warnings | {warning_count} |",
        f"| Invalides | {invalid_count} |",
        f"| Score de confiance moyen | {avg_trust:.1f}/100 |",
        "",
        "## Distribution des Scores de Confiance",
        "",
        "| Plage | Nombre |",
        "|-------|--------|",
    ]
    
    # Distribution des scores de confiance
    ranges = [(90, 100), (70, 89), (50, 69), (0, 49)]
    for low, high in ranges:
        count = sum(1 for r in results if low <= r.trust_score <= high)
        lines.append(f"| {low}-{high} | {count} |")
    
    lines.extend([
        "",
        "## Details par Evaluation",
        "",
        "| Competition | Strate | Ref | Score Total | Confiance | Statut |",
        "|-------------|--------|-----|-------------|-----------|--------|",
    ])
    
    # Trier par score de confiance
    for r in sorted(results, key=lambda x: x.trust_score):
        ref_short = r.ref[:30] + "..." if len(r.ref) > 30 else r.ref
        lines.append(
            f"| {r.competition} | {r.stratum} | {ref_short} | {r.computed_total} | {r.trust_score} | {r.get_status()} |"
        )
    
    # Ajouter les details des erreurs/warnings
    lines.extend([
        "",
        "## Erreurs et Warnings",
        "",
    ])
    
    for r in results:
        if r.errors or r.warnings:
            lines.append(f"### {r.ref}")
            lines.append("")
            if r.errors:
                lines.append("**Erreurs:**")
                for e in r.errors:
                    lines.append(f"- {e}")
            if r.warnings:
                lines.append("**Warnings:**")
                for w in r.warnings:
                    lines.append(f"- {w}")
            lines.append("")
    
    # Ecrire le rapport
    report_path = REPORT_DIR / "validation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Rapport sauvegarde: {report_path}")


def generate_csv_details(results: list) -> None:
    """Genere un CSV avec tous les details."""
    
    csv_path = REPORT_DIR / "validation_details.csv"
    
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        
        # En-tete
        header = [
            "json_path", "competition", "stratum", "ref",
            "A_structure_pipeline", "B_modularite", "C_reproductibilite",
            "D_lisibilite", "E_hygiene", "computed_total", "llm_total",
            "is_valid", "trust_score", "errors", "warnings"
        ]
        writer.writerow(header)
        
        # Donnees
        for r in results:
            row = [
                r.json_path,
                r.competition,
                r.stratum,
                r.ref,
                r.scores.get("A_structure_pipeline", ""),
                r.scores.get("B_modularite", ""),
                r.scores.get("C_reproductibilite", ""),
                r.scores.get("D_lisibilite", ""),
                r.scores.get("E_hygiene", ""),
                r.computed_total,
                r.llm_total if r.llm_total is not None else "",
                r.is_valid,
                r.trust_score,
                "; ".join(r.errors),
                "; ".join(r.warnings)
            ]
            writer.writerow(row)
    
    print(f"Details CSV sauvegardes: {csv_path}")


def main():
    """Point d'entree principal."""
    
    print("Validation des evaluations LLM...")
    print(f"Dossier d'evaluation: {EVAL_DIR}")
    
    if not EVAL_DIR.exists():
        print(f"ERREUR: {EVAL_DIR} n'existe pas")
        return
    
    # Trouver tous les JSON
    json_files = list(EVAL_DIR.rglob("*.json"))
    print(f"{len(json_files)} fichiers JSON trouves")
    
    if not json_files:
        print("Aucun fichier a valider")
        return
    
    # Valider chaque fichier
    results = []
    for i, json_path in enumerate(json_files, 1):
        print(f"=" * 70)
        ref_name = json_path.stem[:30]
        print(f"[{i}/{len(json_files)}] Validation: {ref_name}...")
        
        result = validate_notebook_evaluation(json_path)
        results.append(result)
    
    print("=" * 70)
    
    # Statistiques
    valid_count = sum(1 for r in results if r.is_valid)
    warning_count = sum(1 for r in results if r.is_valid and r.warnings)
    invalid_count = sum(1 for r in results if not r.is_valid)
    
    print(f"\nEvaluations valides: {valid_count}/{len(results)}")
    print(f"Avec warnings: {warning_count}")
    print(f"Invalides: {invalid_count}")
    
    # Generer les rapports
    print("\nGeneration du rapport de validation...")
    generate_validation_report(results)
    generate_csv_details(results)
    
    print("Validation terminee avec succes!")


if __name__ == "__main__":
    main()
