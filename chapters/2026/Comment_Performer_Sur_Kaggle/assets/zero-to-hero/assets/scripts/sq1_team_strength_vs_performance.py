import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout


PROFILE_USERNAME = "yuanzhezhou"


# ------------------------------------------------------------
# Logs
# ------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[LOG] {msg}", flush=True)


# ------------------------------------------------------------
# Scraping du rang global des coéquipiers
# ------------------------------------------------------------

RANK_RE = re.compile(r"(\d[\d,]*)\s*of\s*[\d,]+")


def parse_global_rank(text: str) -> Optional[int]:
    """
    Extrait le rang global à partir du texte complet de la page profil Kaggle.
    Cherche un motif du type '1 of 201,725'.
    """
    if not text:
        return None
    m = RANK_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except ValueError:
        return None


def fetch_teammate_ranks(usernames, state_path: str, cache_path: Path) -> Dict[str, Optional[int]]:
    """
    usernames : iterable de usernames Kaggle (strings)
    state_path : fichier kaggle_state.json (login)
    cache_path : JSON pour mémoriser les rangs déjà récupérés

    Retourne un dict {username: global_rank ou None}
    """
    usernames = sorted(set(usernames))

    # 1) Charger le cache existant si présent
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    else:
        cache = {}

    # usernames déjà en cache -> on ne les refait pas
    to_fetch = [u for u in usernames if u not in cache]
    log(f"[LOG] {len(usernames)} coéquipiers uniques, {len(to_fetch)} à scraper (reste en cache).")

    if not to_fetch:
        return cache

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(storage_state=state_path)
        page = context.new_page()

        for u in to_fetch:
            url = f"https://www.kaggle.com/{u}"
            log(f"[INFO] Profil -> {url}")
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            except PlaywrightTimeout:
                log(f"[WARN] Timeout lors du chargement de {url}")
                cache[u] = None
                continue

            page.wait_for_timeout(2000)

            try:
                full_text = page.inner_text("body")
            except Exception:
                log(f"[WARN] Impossible de lire le texte pour {u}")
                cache[u] = None
                continue

            rank = parse_global_rank(full_text)
            if rank is None:
                log(f"[WARN] Rang global non trouvé pour {u}")
            else:
                log(f"[DEBUG] {u}: global rank = {rank}")

            cache[u] = rank

        browser.close()

    # Sauvegarde du cache mis à jour
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    log(f"[OK] Cache des rangs coéquipiers mis à jour -> {cache_path.resolve()}")
    return cache


# ------------------------------------------------------------
# Construction du dataset "force d'équipe"
# ------------------------------------------------------------

def build_team_strength_df(
    competitions_csv: str,
    teammates_csv: str,
    state_path: str,
    cache_path: Path,
) -> pd.DataFrame:
    """
    - charge les compétitions du joueur
    - charge les coéquipiers par compétition
    - scrape le rang global de chaque coéquipier (avec cache)
    - calcule pour chaque compétition :
        * nb de coéquipiers
        * rang moyen des coéquipiers (team_strength)
    - fusionne avec perf_score etc.
    """
    # 1) compétitions
    df_comp = pd.read_csv(competitions_csv)
    required_comp_cols = {"competition_slug", "year", "perf_score"}
    if not required_comp_cols.issubset(df_comp.columns):
        raise SystemExit(f"Le CSV compétitions doit contenir les colonnes {required_comp_cols}")

    # 2) coéquipiers
    df_team = pd.read_csv(teammates_csv)
    required_team_cols = {"competition_slug", "year", "teammate"}
    if not required_team_cols.issubset(df_team.columns):
        raise SystemExit(f"Le CSV coéquipiers doit contenir les colonnes {required_team_cols}")

    # 3) récupérer / compléter les rangs globaux
    cache = fetch_teammate_ranks(df_team["teammate"].unique(), state_path, cache_path)

    df_team["global_rank"] = df_team["teammate"].map(cache)
    # On garde uniquement les coéquipiers pour lesquels on a un rang
    df_team_valid = df_team.dropna(subset=["global_rank"]).copy()
    df_team_valid["global_rank"] = df_team_valid["global_rank"].astype(int)

    # 4) agrégation par compétition
    agg = (
        df_team_valid
        .groupby("competition_slug")
        .agg(
            avg_teammate_rank=("global_rank", "mean"),
            min_teammate_rank=("global_rank", "min"),
            max_teammate_rank=("global_rank", "max"),
            num_teammates=("teammate", "nunique"),
        )
        .reset_index()
    )

    # 5) fusion avec les compétitions
    merged = df_comp.merge(agg, on="competition_slug", how="left")

    # Ajout d'un flag solo/team
    merged["has_team"] = merged["num_teammates"].fillna(0) > 0
    merged["team_size_incl_self"] = merged["num_teammates"].fillna(0) + 1

    return merged


# ------------------------------------------------------------
# Visualisations
# ------------------------------------------------------------

def plot_scatter_strength_vs_perf(df: pd.DataFrame, outdir: Path) -> None:
    """
    Scatter : force de l'équipe (rang moyen des coéquipiers) vs perf_score.
    On ne garde que les compétitions avec au moins 1 coéquipier.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    df_plot = df.dropna(subset=["avg_teammate_rank"]).copy()
    if df_plot.empty:
        log("[WARN] Aucun point avec coéquipier pour le scatter.")
        return

    plt.figure(figsize=(7, 5))
    plt.scatter(df_plot["avg_teammate_rank"], df_plot["perf_score"])
    plt.xscale("log")  # rang varie de 1 à 200k -> échelle log plus lisible

    plt.xlabel("Rang global moyen des coéquipiers (plus petit = plus fort)")
    plt.ylabel("Score de performance (0-100)")
    plt.title(f"Force moyenne de l'équipe vs performance ({PROFILE_USERNAME})")

    plt.tight_layout()
    out_path = outdir / "team_strength_vs_perf_scatter.png"
    plt.savefig(out_path)
    plt.close()
    log(f"[OK] Scatter sauvegardé -> {out_path}")


def plot_boxplot_strength_vs_perf(df: pd.DataFrame, outdir: Path) -> None:
    """
    Boxplot : performance en fonction de la force de l'équipe.
    On divise les compétitions avec coéquipiers en 3 groupes (fort/moyen/faible)
    selon le rang moyen des coéquipiers (quantiles).
    """
    outdir.mkdir(parents=True, exist_ok=True)

    df_team = df.dropna(subset=["avg_teammate_rank"]).copy()
    if df_team.empty:
        log("[WARN] Pas assez de données équipe pour le boxplot.")
        return

    # Plus le rang est petit, plus le teammate est fort
    df_team["strength_group"] = pd.qcut(
        df_team["avg_teammate_rank"],
        q=3,
        labels=["Équipe forte", "Équipe moyenne", "Équipe plus faible"],
    )

    data = [
        df_team[df_team["strength_group"] == label]["perf_score"].values
        for label in ["Équipe forte", "Équipe moyenne", "Équipe plus faible"]
    ]

    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=["Équipe forte", "Équipe moyenne", "Équipe plus faible"])
    plt.ylabel("Score de performance (0-100)")
    plt.title(f"Performance selon la force moyenne de l'équipe ({PROFILE_USERNAME})")

    plt.tight_layout()
    out_path = outdir / "team_strength_vs_perf_boxplot.png"
    plt.savefig(out_path)
    plt.close()
    log(f"[OK] Boxplot sauvegardé -> {out_path}")


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--competitions-csv",
        default="out/user_competitions_raw.csv",
        help="CSV compétitions (competition_slug, year, perf_score, ...)",
    )
    parser.add_argument(
        "--teammates-csv",
        default="out/user_teammates_raw.csv",
        help="CSV coéquipiers (competition_slug, year, teammate)",
    )
    parser.add_argument(
        "--state",
        default="kaggle_state.json",
        help="storage_state Playwright avec la session Kaggle",
    )
    parser.add_argument(
        "--cache",
        default="out/teammate_global_ranks.json",
        help="Fichier cache JSON pour les rangs globaux des coéquipiers",
    )
    parser.add_argument(
        "--outdir",
        default="out/figures_sq1",
        help="Dossier de sortie pour les figures",
    )
    args = parser.parse_args()

    cache_path = Path(args.cache)
    outdir = Path(args.outdir)

    # 1) Construire le DataFrame principal
    df_strength = build_team_strength_df(
        competitions_csv=args.competitions_csv,
        teammates_csv=args.teammates_csv,
        state_path=args.state,
        cache_path=cache_path,
    )

    # 2) Sauvegarde CSV
    out_csv = Path("out/user_team_strength.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_strength.to_csv(out_csv, index=False)
    log(f"[OK] Dataset force d'équipe -> {out_csv.resolve()}")

    # 3) Visualisations
    plot_scatter_strength_vs_perf(df_strength, outdir)
    plot_boxplot_strength_vs_perf(df_strength, outdir)

    print("[OK] Done. Visualisations force d'équipe vs performance générées.")


if __name__ == "__main__":
    main()
