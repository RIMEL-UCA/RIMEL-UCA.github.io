#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Configuration - chemins relatifs au dossier kaggle-code-quality
SCRIPT_DIR = Path(__file__).parent.parent  # Remonte de scripts/ vers kaggle-code-quality/
EVAL_DIR = SCRIPT_DIR / "corpus" / "evaluations"
OUT_DIR = SCRIPT_DIR / "data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== Helpers =====
def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8", errors="replace"))

def find_score_total(obj: dict) -> float:
    """    
    Suite au feedback de présentation, le calcul du score total
    est maintenant toujours effectué par le script Python, pas par le LLM.
    
    """
    # Priorité 1: Calculer depuis scores_20 (format actuel)
    s20 = obj.get("scores_20", {})
    if isinstance(s20, dict) and s20:
        # Filtrer seulement les valeurs numériques valides
        scores = [pd.to_numeric(v, errors='coerce') for v in s20.values()]
        scores = [s for s in scores if pd.notna(s)]
        if scores:
            return float(sum(scores))
    
    # Fallback: ancien format "scores" sur 0/1/2 (si existant dans vieux JSON)
    s = obj.get("scores", {})
    if isinstance(s, dict) and s:
        scores = [pd.to_numeric(v, errors='coerce') for v in s.values()]
        scores = [s for s in scores if pd.notna(s)]
        if scores:
            return float(sum(scores))
    
    return float("nan")

def normalize_criteria(obj: dict) -> dict:
    # critères sur 20 si dispo
    s20 = obj.get("scores_20", {})
    if isinstance(s20, dict) and s20:
        # normaliser noms
        out = {
            "A_structure_pipeline": s20.get("A_structure_pipeline"),
            "B_modularite": s20.get("B_modularite"),
            "C_reproductibilite": s20.get("C_reproductibilite"),
            "D_lisibilite": s20.get("D_lisibilite"),
            "E_hygiene": s20.get("E_hygiene"),
        }
        return out

    # sinon 0/1/2 -> on garde quand même
    s = obj.get("scores", {})
    if isinstance(s, dict) and s:
        return {
            "A_structure_pipeline": s.get("structure_pipeline"),
            "B_modularite": s.get("modularite"),
            "C_reproductibilite": s.get("reproductibilite"),
            "D_lisibilite": s.get("lisibilite"),
            "E_hygiene": s.get("hygiene"),
        }

    return {
        "A_structure_pipeline": None,
        "B_modularite": None,
        "C_reproductibilite": None,
        "D_lisibilite": None,
        "E_hygiene": None,
    }

def main():
    if not EVAL_DIR.exists():
        raise SystemExit(f"Erreur: {EVAL_DIR} n'existe pas. Vérifie le chemin des JSON.")

    rows = []
    for p in EVAL_DIR.rglob("*.json"):
        obj = load_json(p)

        comp = obj.get("competition") or p.parts[-3] if len(p.parts) >= 3 else ""
        stratum = obj.get("stratum") or p.parts[-2] if len(p.parts) >= 2 else ""
        ref = obj.get("ref") or p.stem

        score_total = find_score_total(obj)
        crit = normalize_criteria(obj)

        rows.append({
            "competition": comp,
            "stratum": stratum,
            "ref": ref,
            "score_total_100": score_total,
            **crit,
            "json_path": str(p),
        })

    df = pd.DataFrame(rows)

    # Nettoyage
    df["score_total_100"] = pd.to_numeric(df["score_total_100"], errors="coerce")
    for c in ["A_structure_pipeline","B_modularite","C_reproductibilite","D_lisibilite","E_hygiene"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ===== Exports CSV =====
    df_path = OUT_DIR / "sq3_all_scores.csv"
    df.to_csv(df_path, index=False, encoding="utf-8-sig")

    by_stratum = (
        df.groupby("stratum")
          .agg(
              n=("score_total_100","count"),
              mean=("score_total_100","mean"),
              std=("score_total_100","std"),
              median=("score_total_100","median"),
              min=("score_total_100","min"),
              max=("score_total_100","max"),
          )
          .round(2)
          .reset_index()
    )
    by_stratum_path = OUT_DIR / "sq3_summary_by_stratum.csv"
    by_stratum.to_csv(by_stratum_path, index=False, encoding="utf-8-sig")

    by_comp_stratum = (
        df.groupby(["competition","stratum"])
          .agg(n=("score_total_100","count"), mean=("score_total_100","mean"))
          .round(2)
          .reset_index()
          .sort_values(["competition","stratum"])
    )
    by_comp_path = OUT_DIR / "sq3_summary_by_competition_stratum.csv"
    by_comp_stratum.to_csv(by_comp_path, index=False, encoding="utf-8-sig")

    # ===== Graph 1: Boxplot par strate =====
    order = ["top_1%","top_5%","top_10%","p40_50","40_50%","percent_40_50"]
    strata = [s for s in order if s in df["stratum"].unique()] + [s for s in df["stratum"].unique() if s not in order]

    data = [df[df["stratum"] == s]["score_total_100"].dropna().values for s in strata]
    plt.figure()
    plt.boxplot(data, labels=strata)
    plt.title("Qualité du code (score /100) par strate")
    plt.xlabel("Strate leaderboard")
    plt.ylabel("Score qualité code (/100)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sq3_boxplot_by_stratum.png", dpi=200)
    plt.close()

    # ===== Graph 2: Barplot des moyennes =====
    plt.figure()
    sub = by_stratum.copy()
    sub = sub.set_index("stratum").loc[strata].reset_index() if set(strata).issubset(set(sub["stratum"])) else sub
    plt.bar(sub["stratum"], sub["mean"])
    plt.title("Score moyen (/100) par strate")
    plt.xlabel("Strate leaderboard")
    plt.ylabel("Moyenne score qualité code (/100)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sq3_bar_mean_by_stratum.png", dpi=200)
    plt.close()

    # ===== Graph 3: Heatmap critères (moyennes) =====
    crit_cols = ["A_structure_pipeline","B_modularite","C_reproductibilite","D_lisibilite","E_hygiene"]
    crit_mean = df.groupby("stratum")[crit_cols].mean()
    crit_mean = crit_mean.loc[strata] if set(strata).issubset(set(crit_mean.index)) else crit_mean

    plt.figure()
    plt.imshow(crit_mean.values)
    plt.xticks(range(len(crit_cols)), crit_cols, rotation=30, ha="right")
    plt.yticks(range(len(crit_mean.index)), crit_mean.index)
    plt.colorbar()
    plt.title("Moyennes des critères (sur 20) par strate")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sq3_heatmap_criteria_by_stratum.png", dpi=200)
    plt.close()

    # ===== Top N notebooks =====
    top = df.sort_values("score_total_100", ascending=False).head(10)
    top_path = OUT_DIR / "sq3_top10_notebooks.csv"
    top.to_csv(top_path, index=False, encoding="utf-8-sig")

    print("OK outputs ->", OUT_DIR)
    print(" -", df_path.name)
    print(" -", by_stratum_path.name)
    print(" -", by_comp_path.name)
    print(" - sq3_boxplot_by_stratum.png")
    print(" - sq3_bar_mean_by_stratum.png")
    print(" - sq3_heatmap_criteria_by_stratum.png")
    print(" -", top_path.name)

if __name__ == "__main__":
    main()
