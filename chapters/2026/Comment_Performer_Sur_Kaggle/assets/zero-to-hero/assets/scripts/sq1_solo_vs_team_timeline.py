import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

PROFILE_NAME = "yuanzhezhou"  # juste pour les titres de graphes


def log(msg: str) -> None:
    print(f"[LOG] {msg}", flush=True)


# --------- helpers pour détecter la team --------- #

USERNAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_\-]{0,50}$")


def normalize_username(u: str) -> str:
    return u.strip().lstrip("@").strip("/")


def extract_team_size_from_row(row) -> int | None:
    """
    À partir d'un "row" Playwright (bloc de leaderboard),
    on récupère les liens de profil et on compte les membres.
    """
    if row is None:
        return None

    members = set()

    # 1) liens /account/<user>
    for a in row.query_selector_all("a[href*='/account/']"):
        href = a.get_attribute("href") or ""
        m = re.search(r"/account/([^/?#]+)", href)
        if m:
            members.add(normalize_username(m.group(1)))

    # 2) liens /u/<user> ou /<user>
    for a in row.query_selector_all("a[href]"):
        href = (a.get_attribute("href") or "").strip()
        if not href or not href.startswith("/"):
            continue
        if "/competitions/" in href:
            continue

        candidate = href.split("?", 1)[0].split("#", 1)[0]
        candidate = candidate.strip("/")
        if candidate.startswith("u/"):
            candidate = candidate[2:]
        candidate = candidate.split("/", 1)[0]
        candidate = normalize_username(candidate)

        if USERNAME_RE.match(candidate):
            members.add(candidate)

    if not members:
        # Au pire, on considère au moins le joueur lui-même
        return 1

    return len(members)


def get_team_size_on_leaderboard_by_rank(page, slug: str, rank: int) -> int | None:
    """
    Cherche sur le leaderboard la ligne dont le rang est 'rank'
    et renvoie la team_size.
    On copie la logique de 02_scrape_leaderboard_playwright :
    - on prend [role='row'], sinon tous les <div>
    - on garde ceux dont le texte commence par "<rank> "
    """
    # 1) rows avec role='row'
    rows = page.query_selector_all("[role='row']")
    if not rows:
        # 2) fallback : tous les div
        rows = page.query_selector_all("div")

    if not rows:
        log(f"[WARN] Aucun bloc (row/div) trouvé sur le leaderboard de {slug}")
        return None

    pattern = re.compile(rf"^\s*{rank}\s")

    for r in rows:
        try:
            txt = (r.inner_text() or "").strip()
        except Exception:
            continue

        if not txt:
            continue

        # on veut des blocs qui commencent par "rank "
        if not pattern.match(txt):
            continue

        size = extract_team_size_from_row(r)
        log(f"[DEBUG] {slug}: rank={rank}, team_size={size}")
        return size

    log(f"[WARN] Impossible de trouver la ligne avec rank={rank} sur {slug}")
    return None


# --------- scraping des tailles d'équipe --------- #


def get_team_sizes(csv_path: str, state_path: str, headless: bool = True) -> pd.DataFrame:
    """
    Lit le CSV user_competitions_raw.csv (doit contenir
    competition_slug, year, rank) et scrape la taille d'équipe.
    """
    df = pd.read_csv(csv_path)
    required_cols = {"competition_slug", "year", "rank"}
    if not required_cols.issubset(df.columns):
        raise SystemExit(
            f"Le CSV {csv_path} doit contenir les colonnes {required_cols}. "
            f"Colonnes actuelles: {set(df.columns)}"
        )

    df_valid = df.dropna(subset=["rank"]).copy()
    df_valid["rank"] = df_valid["rank"].astype(int)

    sizes: list[tuple[str, int, int | None]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(storage_state=state_path)
        page = context.new_page()

        for slug, year, rank in zip(
            df_valid["competition_slug"], df_valid["year"], df_valid["rank"]
        ):
            url = f"https://www.kaggle.com/competitions/{slug}/leaderboard"
            log(f"[INFO] leaderboard -> {url}")

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            except PlaywrightTimeout:
                log("[WARN] Timeout, on continue quand même")
            page.wait_for_timeout(2000)

            size = get_team_size_on_leaderboard_by_rank(page, slug, rank)
            sizes.append((slug, year, size))

        browser.close()

    return pd.DataFrame(sizes, columns=["competition_slug", "year", "team_size"])


# --------- visualisation --------- #


def plot_results(df: pd.DataFrame, outdir: str) -> None:
    """
    df : colonnes year, team_size
    """
    df_valid = df.dropna(subset=["team_size"]).copy()
    if df_valid.empty:
        log("[ERR] Aucune team_size valide trouvée -> impossible de tracer solo vs équipe.")
        return

    df_valid["team_size"] = df_valid["team_size"].astype(int)

    yearly = (
        df_valid.groupby("year", as_index=False)
        .agg(
            solo=("team_size", lambda s: (s == 1).sum()),
            team=("team_size", lambda s: (s > 1).sum()),
        )
        .sort_values("year")
    )

    years = yearly["year"].astype(str)

    # bar empilée
    plt.figure()
    plt.bar(years, yearly["solo"], label="Solo")
    plt.bar(years, yearly["team"], bottom=yearly["solo"], label="Équipe")
    plt.title(f"Évolution des participations solo vs équipe ({PROFILE_NAME})")
    plt.xlabel("Année (approximative)")
    plt.ylabel("Nombre de compétitions")
    plt.legend()
    plt.tight_layout()

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    out_counts = outdir_path / "solo_vs_team_counts.png"
    plt.savefig(out_counts)
    plt.close()
    log(f"[OK] Figure sauvegardée -> {out_counts}")

    # proportions
    yearly["total"] = yearly["solo"] + yearly["team"]
    yearly["solo_ratio"] = yearly["solo"] / yearly["total"]
    yearly["team_ratio"] = yearly["team"] / yearly["total"]

    plt.figure()
    plt.plot(years, yearly["solo_ratio"], marker="o", label="Solo")
    plt.plot(years, yearly["team_ratio"], marker="o", label="Équipe")
    plt.ylim(0, 1)
    plt.title(f"Proportion solo vs équipe ({PROFILE_NAME})")
    plt.xlabel("Année (approximative)")
    plt.ylabel("Proportion")
    plt.legend()
    plt.tight_layout()

    out_ratios = outdir_path / "solo_vs_team_ratios.png"
    plt.savefig(out_ratios)
    plt.close()
    log(f"[OK] Figure sauvegardée -> {out_ratios}")


# --------- main --------- #


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="out/user_competitions_raw.csv")
    parser.add_argument("--state", default="kaggle_state.json")
    parser.add_argument("--outdir", default="out/figures_sq1")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    df_sizes = get_team_sizes(args.csv, args.state, args.headless)
    log(df_sizes.head().to_string())

    plot_results(df_sizes, args.outdir)

    print("[OK] Done. Visualisations générées.")


if __name__ == "__main__":
    main()
