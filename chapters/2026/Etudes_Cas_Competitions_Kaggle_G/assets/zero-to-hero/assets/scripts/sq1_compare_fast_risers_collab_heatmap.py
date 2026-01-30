# sq1_compare_fast_risers_collab_heatmap.py
import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# ----------------------------
# Config / regex
# ----------------------------

PROFILE_URL = "https://www.kaggle.com/{username}"
RANK_RE = re.compile(r"(\d[\d,]*)\s*of\s*[\d,]+", flags=re.IGNORECASE)

REL_TIME_RE = re.compile(
    r"\b(a|\d+)\s+(year|years|month|months|day|days)\s+ago\b",
    flags=re.IGNORECASE,
)

SYSTEM_PREFIXES = (
    "/competitions", "/datasets", "/code", "/models", "/organizations",
    "/learn", "/search", "/account", "/c", "/discussion", "/discussions",
    "/notebooks", "/kernels", "/login", "/signin"
)

USERNAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]{0,50}$")


def log(msg: str) -> None:
    print(f"[LOG] {msg}", flush=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Helpers parsing profile competitions
# (inspiré de sq1_scrape_timeline_user.py)
# ----------------------------

def parse_relative_year(raw_text: str, current_year: int) -> Optional[int]:
    if not isinstance(raw_text, str):
        return None
    m = REL_TIME_RE.search(raw_text)
    if not m:
        return None

    qty_raw = m.group(1).lower()
    unit = m.group(2).lower()
    qty = 1 if qty_raw == "a" else int(qty_raw)

    if "year" in unit:
        return current_year - qty
    return current_year


def extract_rank_and_teams(raw_text: str) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(raw_text, str):
        return None, None
    m = re.search(r"\b(\d+)\s*/\s*(\d+)\b", raw_text)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def score_from_rank(rank: Optional[int], teams: Optional[int]) -> float:
    if rank is None or teams is None or teams <= 0:
        return 0.0
    pct = rank / teams
    if pct <= 0.01:
        return 100.0
    if pct <= 0.05:
        return 70.0
    if pct <= 0.10:
        return 40.0
    if pct <= 0.25:
        return 15.0
    return 5.0


def accept_cookies_if_needed(page) -> None:
    for label in ["Accept", "I agree", "J'accepte", "OK, Got it.", "Tout accepter"]:
        try:
            page.click(f"text={label}", timeout=1500)
            page.wait_for_timeout(600)
            return
        except Exception:
            continue


def click_competitions_tab(page) -> None:
    # on tente plusieurs sélecteurs robustes
    for attempt in [
        lambda: page.get_by_role("tab", name=re.compile("Competitions", re.I)).click(timeout=5000),
        lambda: page.click("text=Competitions", timeout=5000),
        lambda: page.locator("a:has-text('Competitions')").first.click(timeout=5000),
    ]:
        try:
            attempt()
            return
        except Exception:
            continue
    raise RuntimeError("Impossible de cliquer l'onglet Competitions")


def extract_competition_cards(page) -> List[Dict]:
    """
    Heuristique simple: on récupère les div contenant un lien /competitions/<slug>
    et on garde inner_text pour parser l'année / rank/n_teams.
    """
    cards: List[Dict] = []
    divs = page.query_selector_all("div")
    for d in divs:
        try:
            txt = (d.inner_text() or "").strip()
        except Exception:
            continue
        if not txt:
            continue
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

        cards.append({"competition_slug": slug, "competition_name": comp_name, "raw_text": txt})

    # dedup slug
    dedup = {}
    for c in cards:
        dedup.setdefault(c["competition_slug"], c)
    return list(dedup.values())


def scrape_user_competitions(username: str, state_path: str, headless: bool, max_scrolls: int = 40) -> pd.DataFrame:
    current_year = datetime.utcnow().year
    rows = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(storage_state=state_path)
        page = context.new_page()

        url = PROFILE_URL.format(username=username)
        log(f"[INFO] Profil -> {url}")
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        except PlaywrightTimeout:
            log("[WARN] Timeout sur profil, on continue.")
        page.wait_for_timeout(2000)

        if "signin" in page.url or "login" in page.url:
            browser.close()
            raise RuntimeError("Not authenticated. Regénère kaggle_state.json")

        accept_cookies_if_needed(page)
        click_competitions_tab(page)
        page.wait_for_timeout(2000)

        all_cards = {}
        stall = 0
        for _ in range(max_scrolls):
            cards = extract_competition_cards(page)
            new = 0
            for c in cards:
                if c["competition_slug"] not in all_cards:
                    all_cards[c["competition_slug"]] = c
                    new += 1
            stall = stall + 1 if new == 0 else 0
            if stall >= 5:
                break
            page.mouse.wheel(0, 3500)
            page.wait_for_timeout(900)

        browser.close()

    for c in all_cards.values():
        raw = c.get("raw_text", "") or ""
        year = parse_relative_year(raw, current_year)
        rank, n_teams = extract_rank_and_teams(raw)
        perf = score_from_rank(rank, n_teams)
        rows.append({
            "username": username,
            "competition_slug": c["competition_slug"],
            "competition_name": c.get("competition_name"),
            "year": year,
            "rank": rank,
            "num_teams": n_teams,
            "perf_score": perf,
            "raw_text": raw,
        })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["year"])
    if not df.empty:
        df["year"] = df["year"].astype(int)
    return df


# ----------------------------
# Teammates scraping (leaderboard?search=username)
# (inspiré de sq1_teammates_heatmap.py)
# ----------------------------

def normalize_username(u: str) -> str:
    return u.strip().lstrip("@").strip("/")


def is_valid_teammate(profile_username: str, candidate: str) -> bool:
    if not candidate:
        return False
    candidate = normalize_username(candidate)
    if candidate.lower() == profile_username.lower():
        return False
    if USERNAME_RE.match(candidate) is None:
        return False
    return True


def extract_teammates_from_page(page, profile_username: str) -> List[str]:
    mates = set()

    # 1) /account/<username>
    for a in page.query_selector_all("a[href*='/account/']"):
        href = a.get_attribute("href") or ""
        m = re.search(r"/account/([^/?#]+)", href)
        if m:
            cand = normalize_username(m.group(1))
            if is_valid_teammate(profile_username, cand):
                mates.add(cand)

    # 2) autres liens /u/<user> ou /<user>
    for a in page.query_selector_all("a[href]"):
        href = (a.get_attribute("href") or "").strip()
        if not href.startswith("/"):
            continue
        if href.startswith(SYSTEM_PREFIXES):
            continue
        cand = href.split("?", 1)[0].split("#", 1)[0].strip("/")
        if cand.startswith("u/"):
            cand = cand[2:]
        cand = cand.split("/", 1)[0]
        cand = normalize_username(cand)
        if is_valid_teammate(profile_username, cand):
            mates.add(cand)

    return sorted(mates)


def scrape_teammates_for_competitions(df_comp: pd.DataFrame, profile_username: str, state_path: str, headless: bool) -> pd.DataFrame:
    """
    Pour chaque competition_slug:
      ouvre /leaderboard?search=<profile_username>
      extrait les usernames visibles -> coéquipiers
    """
    rows = []
    if df_comp.empty:
        return pd.DataFrame(columns=["username", "competition_slug", "year", "teammate"])

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(storage_state=state_path)
        page = context.new_page()

        for slug, year in zip(df_comp["competition_slug"], df_comp["year"]):
            url = f"https://www.kaggle.com/competitions/{slug}/leaderboard?search={profile_username}"
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            except PlaywrightTimeout:
                log(f"[WARN] Timeout leaderboard {slug}")
                continue

            page.wait_for_timeout(2500)
            mates = extract_teammates_from_page(page, profile_username)

            for m in mates:
                rows.append({
                    "username": profile_username,
                    "competition_slug": slug,
                    "year": int(year),
                    "teammate": m
                })

        browser.close()

    return pd.DataFrame(rows)


# ----------------------------
# Teammate global ranks (cache partagé)
# (inspiré de sq1_team_strength_vs_performance.py)
# ----------------------------

def parse_global_rank(text: str) -> Optional[int]:
    if not text:
        return None
    m = RANK_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except ValueError:
        return None


def fetch_teammate_ranks(usernames: List[str], state_path: str, cache_path: Path, headless: bool) -> Dict[str, Optional[int]]:
    usernames = sorted(set([u for u in usernames if u]))
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    else:
        cache = {}

    to_fetch = [u for u in usernames if u not in cache]
    log(f"[INFO] {len(usernames)} coéquipiers uniques, {len(to_fetch)} à scraper (reste cache).")

    if not to_fetch:
        return cache

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(storage_state=state_path)
        page = context.new_page()

        for u in to_fetch:
            url = PROFILE_URL.format(username=u)
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            except PlaywrightTimeout:
                cache[u] = None
                continue

            page.wait_for_timeout(1500)
            try:
                full_text = page.inner_text("body")
            except Exception:
                cache[u] = None
                continue

            cache[u] = parse_global_rank(full_text)

        browser.close()

    ensure_dir(cache_path.parent)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache


# ----------------------------
# Aggregation per year + collaboration index + heatmap
# ----------------------------

def safe_norm(series: pd.Series) -> pd.Series:
    """min-max sur série non vide, sinon 0"""
    if series.empty:
        return series
    s = series.astype(float)
    mn, mx = s.min(), s.max()
    if mx - mn < 1e-12:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - mn) / (mx - mn)


def build_yearly_features(
    df_comp: pd.DataFrame,
    df_team: pd.DataFrame,
    teammate_rank_cache: Dict[str, Optional[int]],
) -> pd.DataFrame:
    """
    Retourne df_yearly avec :
      year, nb_competitions, team_ratio, unique_teammates, avg_teammate_rank_mean, perf_score_total
    """
    if df_comp.empty:
        return pd.DataFrame()

    df = df_comp.copy()

    # perf score yearly
    perf = df.groupby("year", as_index=False).agg(
        nb_competitions=("competition_slug", "nunique"),
        perf_score_total=("perf_score", "sum"),
    )

    # team presence per competition
    if df_team is None or df_team.empty:
        out = perf.copy()
        out["team_ratio"] = 0.0
        out["unique_teammates"] = 0
        out["avg_teammate_rank_mean"] = float("nan")
        return out

    df_team2 = df_team.copy()
    df_team2["global_rank"] = df_team2["teammate"].map(teammate_rank_cache)
    df_team_valid = df_team2.dropna(subset=["global_rank"]).copy()
    if not df_team_valid.empty:
        df_team_valid["global_rank"] = df_team_valid["global_rank"].astype(int)

    # comps with >=1 teammate found (even if rank missing)
    comps_with_team = (
        df_team2.groupby(["year", "competition_slug"], as_index=False)
        .agg(num_teammates=("teammate", "nunique"))
    )
    comps_with_team["has_team"] = comps_with_team["num_teammates"] > 0
    team_year = comps_with_team.groupby("year", as_index=False).agg(
        comps_team=("has_team", "sum")
    )

    # unique teammates/year
    uniq_year = df_team2.groupby("year", as_index=False).agg(
        unique_teammates=("teammate", "nunique")
    )

    # avg teammate rank/year (force)
    if df_team_valid.empty:
        rank_year = pd.DataFrame({"year": perf["year"], "avg_teammate_rank_mean": float("nan")})
    else:
        rank_year = df_team_valid.groupby("year", as_index=False).agg(
            avg_teammate_rank_mean=("global_rank", "mean")
        )

    out = perf.merge(team_year, on="year", how="left").merge(uniq_year, on="year", how="left").merge(rank_year, on="year", how="left")
    out["comps_team"] = out["comps_team"].fillna(0).astype(int)
    out["unique_teammates"] = out["unique_teammates"].fillna(0).astype(int)

    out["team_ratio"] = out["comps_team"] / out["nb_competitions"].clip(lower=1)
    return out


def compute_collab_index(df_yearly: pd.DataFrame) -> pd.DataFrame:
    """
    Combine les métriques en un index 0..1.
    - team_ratio (déjà 0..1)
    - unique_teammates (log1p puis min-max)
    - avg_teammate_rank_mean : plus petit = plus fort -> on inverse après log
    - nb_competitions (log1p puis min-max)
    """
    if df_yearly.empty:
        return df_yearly

    d = df_yearly.copy()

    # Normalisations
    d["n_comp_norm"] = safe_norm(d["nb_competitions"].apply(lambda x: math.log1p(float(x))))
    d["uniq_norm"] = safe_norm(d["unique_teammates"].apply(lambda x: math.log1p(float(x))))

    # Force : on veut "plus fort = plus grand"
    # rank_mean petit = fort -> score = -log(rank_mean)
    def strength_score(x):
        if pd.isna(x) or x <= 0:
            return float("nan")
        return -math.log(float(x))

    strength_raw = d["avg_teammate_rank_mean"].apply(strength_score)
    # si tout nan -> 0
    if strength_raw.dropna().empty:
        d["strength_norm"] = 0.0
    else:
        # on remplit les nan par min pour éviter de punir trop violemment
        filled = strength_raw.fillna(strength_raw.min())
        d["strength_norm"] = safe_norm(filled)

    # team_ratio est déjà 0..1
    d["team_ratio_norm"] = d["team_ratio"].fillna(0.0).clip(0, 1)

    # Index final (pondération simple)
    d["collab_index"] = (
        0.40 * d["team_ratio_norm"] +
        0.25 * d["uniq_norm"] +
        0.20 * d["strength_norm"] +
        0.15 * d["n_comp_norm"]
    ).clip(0, 1)

    return d


def plot_users_years_heatmap(pivot: pd.DataFrame, outdir: Path) -> Path:
    ensure_dir(outdir)
    if pivot.empty:
        raise RuntimeError("Pivot vide, rien à tracer.")

    users = list(pivot.index)
    years = list(pivot.columns)
    values = pivot.values

    plt.figure(figsize=(max(7, len(years) * 1.0), max(4, len(users) * 0.5)))
    im = plt.imshow(values, aspect="auto")
    plt.colorbar(im, label="Collaboration Index (0->1)")

    plt.xticks(range(len(years)), years, rotation=45)
    plt.yticks(range(len(users)), users)

    plt.title("Fast risers - Collaboration Index par année (Users × Years)")
    plt.xlabel("Année (approx. Kaggle 'X years ago')")
    plt.ylabel("Utilisateur")

    plt.tight_layout()
    out_path = outdir / "fast_risers_collab_index_heatmap.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    log(f"[OK] Heatmap -> {out_path}")
    return out_path


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", nargs="+", required=True, help="Liste de usernames Kaggle (sans espaces).")
    parser.add_argument("--state", default="kaggle_state.json", help="storage_state Playwright (login Kaggle).")
    parser.add_argument("--outdir", default="out/figures_sq1", help="Dossier sortie.")
    parser.add_argument("--workdir", default="out/fast_risers", help="Dossier travail (CSVs intermédiaires).")
    parser.add_argument("--headless", action="store_true", help="Playwright headless.")
    parser.add_argument("--max_scrolls", type=int, default=40, help="Scrolls max sur l’onglet Competitions.")
    parser.add_argument("--cache_ranks", default="out/teammate_global_ranks.json", help="Cache global ranks teammates.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    workdir = Path(args.workdir)
    ensure_dir(workdir)

    all_yearly = []
    all_teammates = []
    all_teammate_names = set()

    # 1) scrape competitions + teammates for each user
    for u in args.users:
        if " " in u:
            log(f"[WARN] '{u}' contient un espace (pas un username Kaggle). SKIP. Donne le vrai username.")
            continue

        try:
            df_comp = scrape_user_competitions(u, args.state, headless=args.headless, max_scrolls=args.max_scrolls)
        except Exception as e:
            log(f"[WARN] Impossible de scraper competitions pour {u}: {e}")
            continue

        if df_comp.empty:
            log(f"[WARN] {u}: aucune compétition scrappée (ou année non inférée).")
            continue

        comp_csv = workdir / f"{u}_competitions_raw.csv"
        df_comp.to_csv(comp_csv, index=False)
        log(f"[OK] {u}: competitions -> {comp_csv}")

        # teammates per competition
        try:
            df_team = scrape_teammates_for_competitions(df_comp[["competition_slug", "year"]], u, args.state, headless=args.headless)
        except Exception as e:
            log(f"[WARN] Teammates scrape failed for {u}: {e}")
            df_team = pd.DataFrame(columns=["username", "competition_slug", "year", "teammate"])

        team_csv = workdir / f"{u}_teammates_raw.csv"
        df_team.to_csv(team_csv, index=False)
        log(f"[OK] {u}: teammates -> {team_csv}")

        all_teammates.append(df_team)
        all_teammate_names.update(df_team["teammate"].dropna().astype(str).tolist())

        # yearly features (rank cache filled later)
        all_yearly.append((u, df_comp, df_team))

    if not all_yearly:
        raise SystemExit("[ERR] Aucun utilisateur exploitable. Vérifie les usernames / login / state.")

    # 2) fetch global ranks of all teammates (shared cache)
    cache_path = Path(args.cache_ranks)
    teammate_rank_cache = fetch_teammate_ranks(list(all_teammate_names), args.state, cache_path, headless=args.headless)

    # 3) build yearly features + collab index per user
    rows_user_year = []
    for u, df_comp, df_team in all_yearly:
        y = build_yearly_features(df_comp, df_team, teammate_rank_cache)
        if y.empty:
            continue
        y = compute_collab_index(y)
        y["username"] = u
        rows_user_year.append(y)

        y.to_csv(workdir / f"{u}_yearly_features.csv", index=False)

    df_all = pd.concat(rows_user_year, ignore_index=True)

    # export global table
    df_all.to_csv(workdir / "fast_risers_yearly_features_all.csv", index=False)
    log(f"[OK] Table globale -> {workdir / 'fast_risers_yearly_features_all.csv'}")

    # pivot for heatmap: users x years
    pivot = df_all.pivot_table(index="username", columns="year", values="collab_index", aggfunc="mean").sort_index()
    pivot = pivot.sort_index(axis=1)

    heatmap_path = plot_users_years_heatmap(pivot, outdir)

    # also export pivot csv
    pivot.to_csv(workdir / "fast_risers_collab_index_pivot.csv")
    log(f"[OK] Pivot collab index -> {workdir / 'fast_risers_collab_index_pivot.csv'}")

    print(f"[OK] Done. Heatmap: {heatmap_path}")


if __name__ == "__main__":
    main()
