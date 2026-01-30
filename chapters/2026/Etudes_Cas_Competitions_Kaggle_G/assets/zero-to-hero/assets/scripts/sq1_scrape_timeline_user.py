import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

from datetime import datetime

PROFILE_URL = "https://www.kaggle.com/{username}"


def log(msg: str) -> None:
    print(f"[LOG] {msg}", flush=True)


# --------- Parsing utilitaires --------- #

REL_TIME_RE = re.compile(
    r"\b(a|\d+)\s+(year|years|month|months|day|days)\s+ago\b",
    flags=re.IGNORECASE,
)


def parse_relative_year(raw_text: str, current_year: int) -> int | None:
    """
    Cherche un motif du type '4 years ago', 'a year ago', '6 months ago', '14 days ago'
    et renvoie une année approximative.
    """
    if not isinstance(raw_text, str):
        return None

    m = REL_TIME_RE.search(raw_text)
    if not m:
        return None

    qty_raw = m.group(1).lower()
    unit = m.group(2).lower()

    if qty_raw == "a":
        qty = 1
    else:
        try:
            qty = int(qty_raw)
        except ValueError:
            return None

    # Heuristique simple :
    # - years ago -> year = current_year - qty
    # - months/days ago -> on reste sur current_year (approximation)
    if "year" in unit:
        return current_year - qty
    else:
        return current_year


def extract_rank_and_teams(raw_text: str) -> Tuple[int | None, int | None]:
    """
    Cherche un motif du type '3/1514' (rank/n_teams) dans le texte.
    """
    if not isinstance(raw_text, str):
        return None, None

    m = re.search(r"\b(\d+)\s*/\s*(\d+)\b", raw_text)
    if not m:
        return None, None
    try:
        rank = int(m.group(1))
        teams = int(m.group(2))
        return rank, teams
    except ValueError:
        return None, None


def score_from_rank(rank: int | None, teams: int | None) -> float:
    """
    Définit un score de performance 'maison' à partir du rang relatif.
    Plus le rang est bon, plus le score est élevé.
    """
    if rank is None or teams is None or teams <= 0:
        return 0.0

    pct = rank / teams  # 0.01 = top 1%

    if pct <= 0.01:
        return 100.0
    elif pct <= 0.05:
        return 70.0
    elif pct <= 0.10:
        return 40.0
    elif pct <= 0.25:
        return 15.0
    else:
        return 5.0


# --------- Scraping des cartes de compétitions --------- #


def extract_competition_cards(page) -> List[Dict]:
    """
    Heuristique :
    - on cherche des <div> qui contiennent au moins un lien '/competitions/<slug>'
    - pour chaque bloc, on récupère slug, nom, raw_text
    """
    cards: List[Dict] = []

    divs = page.query_selector_all("div")
    log(f"[DEBUG] Found {len(divs)} <div> on page")

    for d in divs:
        try:
            txt = (d.inner_text() or "").strip()
        except Exception:
            continue

        if not txt:
            continue

        # liens vers competitions dans ce bloc
        links = d.query_selector_all("a[href*='/competitions/']")
        if not links:
            continue

        slug = None
        comp_name = None

        for a in links:
            href = (a.get_attribute("href") or "").strip()
            m = re.search(r"/competitions/([^/?#]+)/?", href)
            if not m:
                continue
            slug = m.group(1)
            link_txt = (a.inner_text() or "").strip()
            if link_txt:
                comp_name = link_txt
                break

        if not slug:
            continue

        cards.append(
            {
                "competition_slug": slug,
                "competition_name": comp_name,
                "raw_text": txt,
            }
        )

    # dédup par slug
    dedup: Dict[str, Dict] = {}
    for c in cards:
        slug = c["competition_slug"]
        if slug not in dedup:
            dedup[slug] = c

    result = list(dedup.values())
    log(f"[DEBUG] extract_competition_cards -> {len(result)} unique cards")
    return result


def click_competitions_tab(page) -> None:
    """
    Essaie plusieurs façons de cliquer sur l'onglet 'Competitions'.
    """
    log("[LOG] Trying to click 'Competitions' tab")

    # 1) via role=tab
    try:
        page.get_by_role("tab", name=re.compile("Competitions", re.I)).click(timeout=5000)
        log("[LOG] Clicked 'Competitions' tab via role=tab")
        return
    except Exception:
        log("[DEBUG] Failed to click tab via get_by_role")

    # 2) via texte
    try:
        page.click("text=Competitions", timeout=5000)
        log("[LOG] Clicked 'Competitions' tab via text selector")
        return
    except Exception:
        log("[DEBUG] Failed to click tab via 'text=Competitions'")

    # 3) via locator
    try:
        page.locator("a:has-text('Competitions')").first.click(timeout=5000)
        log("[LOG] Clicked 'Competitions' tab via locator('a:has-text(...)')")
        return
    except Exception as e:
        log(f"[ERROR] Unable to click 'Competitions' tab: {e}")
        raise SystemExit("Impossible d'ouvrir l'onglet Competitions (sélecteurs à adapter).")


def accept_cookies_if_needed(page) -> None:
    """
    Tentative de clic sur un popup cookies / consentement.
    """
    for label in ["Accept", "I agree", "J'accepte", "OK, Got it."]:
        try:
            page.click(f"text={label}", timeout=1500)
            log(f"[LOG] Cookie popup accepted with '{label}'")
            page.wait_for_timeout(1000)
            return
        except Exception:
            continue
    log("[LOG] No cookie popup detected or none matched.")


def scrape_user_competitions(
    username: str,
    state_path: str,
    headless: bool = True,
    max_scrolls: int = 40,
    stall_limit: int = 5,
) -> List[Dict]:
    """
    Ouvre le profil utilisateur Kaggle, onglet Competitions, scroll et extrait les compétitions.
    Retourne une liste de dict {competition_slug, competition_name, raw_text}.
    """
    all_cards: Dict[str, Dict] = {}
    stall = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(storage_state=state_path)
        page = context.new_page()

        url = PROFILE_URL.format(username=username)
        log(f"[LOG] Opening profile: {url}")

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        except PlaywrightTimeout:
            log("[WARN] Timeout on profile page, continuing anyway")

        page.wait_for_timeout(2500)
        log(f"[LOG] Profile URL after load: {page.url}")
        log(f"[LOG] Title: {page.title()}")

        if "signin" in page.url or "login" in page.url:
            browser.close()
            raise SystemExit("[ERROR] Not authenticated. Regénère kaggle_state.json en te connectant à Kaggle.")

        accept_cookies_if_needed(page)

        # clic sur l'onglet "Competitions"
        click_competitions_tab(page)
        page.wait_for_timeout(3000)
        log("[LOG] Competitions tab should be visible now")

        for i in range(1, max_scrolls + 1):
            cards = extract_competition_cards(page)

            new_here = 0
            for c in cards:
                slug = c["competition_slug"]
                if slug not in all_cards:
                    all_cards[slug] = c
                    new_here += 1

            log(
                f"[LOG] scroll={i:02d} -> found={len(cards)} cards, "
                f"new={new_here}, total_unique={len(all_cards)}"
            )

            if new_here == 0:
                stall += 1
            else:
                stall = 0

            if stall >= stall_limit:
                log(f"[LOG] No new cards for {stall} scrolls -> stopping.")
                break

            # scroll pour charger plus de compet
            page.mouse.wheel(0, 3500)
            page.wait_for_timeout(1200)

        browser.close()

    return list(all_cards.values())


# --------- Timeline & visualisation --------- #


def build_yearly_timeline(rows: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not rows:
        raise SystemExit("[ERR] Aucune compétition scrappée (rows est vide).")

    current_year = datetime.utcnow().year

    enriched: List[Dict] = []
    for r in rows:
        raw = r.get("raw_text", "") or ""
        year = parse_relative_year(raw, current_year)
        rank, n_teams = extract_rank_and_teams(raw)
        score = score_from_rank(rank, n_teams)

        entry = dict(r)
        entry["year"] = year
        entry["rank"] = rank
        entry["num_teams"] = n_teams
        entry["perf_score"] = score
        enriched.append(entry)

    df = pd.DataFrame(enriched)

    # garder seulement les compet avec une année définie
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    if df.empty:
        raise SystemExit("[ERR] Impossible d'inférer une année pour les compétitions scrappées.")

    # agrégat par année
    yearly = (
        df.groupby("year", as_index=False)
        .agg(
            nb_competitions=("competition_slug", "count"),
            total_score=("perf_score", "sum"),
        )
        .sort_values("year")
    )

    return df, yearly


def plot_timeline(yearly: pd.DataFrame, out_dir: Path, username: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    years_str = yearly["year"].astype(str)

    # --- 1) Nombre de compétitions par an ---
    plt.figure(figsize=(7, 5))
    plt.bar(years_str, yearly["nb_competitions"])
    plt.title(f"Nombre de compétitions par an ({username})")
    plt.xlabel("Année (approximative, basée sur 'X years ago')")
    plt.ylabel("Nombre de compétitions")
    plt.tight_layout()
    out1 = out_dir / f"{username}_competitions_per_year.png"
    plt.savefig(out1, dpi=150)
    plt.close()
    log(f"[OK] Saved {out1}")

    # --- 2) Score de performance par an ---
    plt.figure(figsize=(7, 5))
    plt.plot(years_str, yearly["total_score"], marker="o")
    plt.title(f"Score de performance par an ({username})")
    plt.xlabel("Année (approximative, basée sur 'X years ago')")
    plt.ylabel("Score de performance (rang relatif)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    out2 = out_dir / f"{username}_performance_score_per_year.png"
    plt.savefig(out2, dpi=150)
    plt.close()
    log(f"[OK] Saved {out2}")


# --------- main --------- #


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--username",
        default="yuanzhezhou",
        help="Username Kaggle (par défaut: yuanzhezhou)",
    )
    parser.add_argument(
        "--state",
        default="kaggle_state.json",
        help="Fichier Playwright storage_state JSON (login Kaggle)",
    )
    parser.add_argument(
        "--out_raw",
        default="out/user_competitions_raw.csv",
        help="CSV brut des compétitions scrappées",
    )
    parser.add_argument(
        "--out_figdir",
        default="out/figures_sq1",
        help="Dossier de sortie pour les figures",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max_scrolls", type=int, default=40)
    parser.add_argument("--stall_limit", type=int, default=5)
    args = parser.parse_args()

    out_raw = Path(args.out_raw)
    out_raw.parent.mkdir(parents=True, exist_ok=True)

    log(f"=== SQ1 scrape+timeline for user '{args.username}' ===")

    rows = scrape_user_competitions(
        username=args.username,
        state_path=args.state,
        headless=args.headless,
        max_scrolls=args.max_scrolls,
        stall_limit=args.stall_limit,
    )

    log(f"[LOG] Total competitions scraped (unique): {len(rows)}")

    df_raw, yearly = build_yearly_timeline(rows)

    df_raw.to_csv(out_raw, index=False)
    log(f"[OK] Raw competitions CSV saved -> {out_raw.resolve()}")

    yearly_path = out_raw.with_name(out_raw.stem + "_per_year.csv")
    yearly.to_csv(yearly_path, index=False)
    log(f"[OK] Yearly aggregates saved -> {yearly_path.resolve()}")

    plot_timeline(yearly, Path(args.out_figdir), args.username)

    log("=== Done ===")


if __name__ == "__main__":
    main()
