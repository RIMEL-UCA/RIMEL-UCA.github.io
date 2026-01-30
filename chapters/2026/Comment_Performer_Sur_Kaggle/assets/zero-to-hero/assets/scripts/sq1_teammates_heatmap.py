import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

PROFILE_USERNAME = "yuanzhezhou"  # joueur étudié


# -------------------------------------------------------------------
# Logs
# -------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[LOG] {msg}", flush=True)


# -------------------------------------------------------------------
# Helpers pour filtrer les usernames valides
# -------------------------------------------------------------------

USERNAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]{0,50}$")

SYSTEM_SLUG_BLACKLIST = {
    "benchmarks",
    "login",
    "models",
    "datasets",
    "code",
    "kernels",
    "notebooks",
    "discussions",
    "discussion",
    "cookies",
    "learn",
    "search",
    "account",
    "c",
    "organizations",
    "game-arena",   # on bloque bien celui-là
}


def normalize_username(u: str) -> str:
    return u.strip().lstrip("@").strip("/")


def is_valid_teammate(name: str) -> bool:
    if not name:
        return False
    name = normalize_username(name)
    low = name.lower()
    if low == PROFILE_USERNAME.lower():
        # on ne garde pas le joueur lui-même
        return False
    if low in SYSTEM_SLUG_BLACKLIST:
        return False
    if USERNAME_RE.match(name) is None:
        return False
    return True


# -------------------------------------------------------------------
# Extraction des coéquipiers à partir de la page leaderboard filtrée
# -------------------------------------------------------------------

def extract_teammates_from_page(page) -> List[str]:
    """
    On ne cherche plus une row spécifique.
    On suppose qu'on est sur :
      /competitions/<slug>/leaderboard?search=yuanzhezhou
    donc la page est déjà filtrée sur son équipe.

    On prend tous les liens vers /account/<username> et /u/<username>,
    on nettoie, on blacklist, on enlève PROFILE_USERNAME.
    """
    teammates = set()

    # 1) /account/<username>
    for a in page.query_selector_all("a[href*='/account/']"):
        href = a.get_attribute("href") or ""
        m = re.search(r"/account/([^/?#]+)", href)
        if not m:
            continue
        candidate = normalize_username(m.group(1))
        if is_valid_teammate(candidate):
            teammates.add(candidate)

    # 2) /u/<username> ou /<username>...
    for a in page.query_selector_all("a[href]"):
        href = (a.get_attribute("href") or "").strip()
        if not href.startswith("/"):
            continue

        # liens clairement non-profils
        if href.startswith((
            "/competitions", "/datasets", "/code",
            "/models", "/organizations", "/learn",
            "/search", "/account", "/c", "/discussion",
            "/discussions", "/notebooks", "/kernels"
        )):
            continue

        candidate = href.split("?", 1)[0].split("#", 1)[0]
        candidate = candidate.strip("/")
        if candidate.startswith("u/"):
            candidate = candidate[2:]
        candidate = candidate.split("/", 1)[0]
        candidate = normalize_username(candidate)

        if is_valid_teammate(candidate):
            teammates.add(candidate)

    return sorted(teammates)


def scrape_teammates_for_user(csv_path: str, state_path: str,
                              headless: bool = True) -> pd.DataFrame:
    """
    Lit out/user_competitions_raw.csv (competition_slug, year, raw_text, ...)
    et pour chaque compétition :
      -> ouvre le leaderboard filtré avec ?search=PROFILE_USERNAME
      -> récupère tous les comptes Kaggle visibles sur la page
      -> les considère comme coéquipiers potentiels

    Retourne un DataFrame :
      competition_slug, year, teammate
    """
    df = pd.read_csv(csv_path)
    required_cols = {"competition_slug", "year"}
    if not required_cols.issubset(df.columns):
        raise SystemExit(
            f"Le CSV {csv_path} doit contenir les colonnes {required_cols}"
        )

    rows: List[Dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(storage_state=state_path)
        page = context.new_page()

        for slug, year in zip(df["competition_slug"], df["year"]):
            url = f"https://www.kaggle.com/competitions/{slug}/leaderboard?search={PROFILE_USERNAME}"
            log(f"[INFO] leaderboard -> {url}")

            try:
                page.goto(url, wait_until="networkidle", timeout=60_000)
            except PlaywrightTimeout:
                log("[WARN] Timeout sur le leaderboard, on continue quand même")
                continue

            # petite pause pour laisser les scripts React rendre la page
            page.wait_for_timeout(3000)

            teammates = extract_teammates_from_page(page)
            log(f"[DEBUG] {slug}: coéquipiers trouvés = {teammates}")

            for mate in teammates:
                rows.append(
                    {
                        "competition_slug": slug,
                        "year": int(year),
                        "teammate": mate,
                    }
                )

        browser.close()

    if not rows:
        log("[WARN] Aucun coéquipier trouvé (rows vide après scraping).")
        return pd.DataFrame(columns=["competition_slug", "year", "teammate"])

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Construction matrice & heatmap
# -------------------------------------------------------------------

def build_teammate_year_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    if "teammate" not in df.columns:
        log("[ERR] Pas de colonne 'teammate' dans le DF, impossible de construire la matrice.")
        return pd.DataFrame()

    df["teammate"] = df["teammate"].astype(str)
    df = df[df["teammate"].apply(is_valid_teammate)]

    if df.empty:
        return pd.DataFrame()

    pivot = (
        df.pivot_table(
            index="teammate",
            columns="year",
            values="competition_slug",
            aggfunc="count",
            fill_value=0,
        )
        .sort_index(axis=1)
        .sort_index(axis=0)
    )

    # Supprimer lignes / colonnes entièrement nulles (au cas où)
    pivot = pivot.loc[:, pivot.sum(axis=0) > 0]
    pivot = pivot.loc[pivot.sum(axis=1) > 0]

    log(f"[LOG] Matrice teammate/year shape = {pivot.shape}")
    return pivot


def plot_heatmap(pivot: pd.DataFrame, outdir: Path, top_n: int = 20) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    if pivot.empty:
        log("[ERR] Matrice vide, impossible de tracer la heatmap.")
        return

    pivot = pivot.copy()
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False)
    pivot_top = pivot.head(top_n).drop(columns=["total"])

    years = list(pivot_top.columns)
    mates = list(pivot_top.index)
    values = pivot_top.values

    plt.figure(figsize=(max(6, len(years) * 1.2), max(4, len(mates) * 0.4)))
    im = plt.imshow(values, aspect="auto")
    plt.colorbar(im, label="Nb de compétitions ensemble")

    plt.xticks(range(len(years)), years, rotation=45)
    plt.yticks(range(len(mates)), mates)

    plt.title(f"Top {top_n} coéquipiers par année ({PROFILE_USERNAME})")
    plt.xlabel("Année (approximative)")
    plt.ylabel("Coéquipier")

    plt.tight_layout()
    out_path = outdir / "teammates_heatmap_topN.png"
    plt.savefig(out_path)
    plt.close()
    log(f"[OK] Heatmap sauvegardée -> {out_path}")


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="out/user_competitions_raw.csv",
        help="CSV brut des compétitions scrappées pour l'utilisateur",
    )
    parser.add_argument(
        "--state",
        default="kaggle_state.json",
        help="storage_state Playwright avec la session Kaggle (02_login_save_state.py)",
    )
    parser.add_argument(
        "--outdir",
        default="out/figures_sq1",
        help="Dossier de sortie pour les figures",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Nombre de coéquipiers à afficher dans la heatmap (top N)",
    )
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Scrape coéquipiers -> DataFrame (competition_slug, year, teammate)
    df_teammates = scrape_teammates_for_user(
        csv_path=args.csv,
        state_path=args.state,
        headless=args.headless,
    )

    # 2) Sauvegarde CSV brut
    teammates_csv = Path("out/user_teammates_raw.csv")
    teammates_csv.parent.mkdir(parents=True, exist_ok=True)
    df_teammates.to_csv(teammates_csv, index=False)
    log(f"[OK] Coéquipiers scrappés -> {teammates_csv.resolve()}")

    # 3) Matrice teammate / year
    pivot = build_teammate_year_matrix(df_teammates)
    pivot_path = Path("out/user_teammates_matrix.csv")
    pivot.to_csv(pivot_path)
    log(f"[OK] Matrice teammate/year -> {pivot_path.resolve()}")

    # 4) Heatmap
    plot_heatmap(pivot, outdir, top_n=args.top_n)

    print("[OK] Done. Heatmap coéquipiers générée.")


if __name__ == "__main__":
    main()
