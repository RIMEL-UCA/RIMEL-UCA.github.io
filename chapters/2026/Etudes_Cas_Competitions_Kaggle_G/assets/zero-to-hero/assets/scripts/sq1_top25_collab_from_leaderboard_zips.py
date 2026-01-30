# sq1_top25_collab_from_leaderboard_zips.py
# Build collaboration metrics for a list of users by reading Kaggle leaderboard ZIPs already downloaded.
# Robust coverage: distinguish "missing ZIPs" vs "user not found inside ZIPs".

import argparse
import re
import zipfile
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


def log(msg: str) -> None:
    print(msg, flush=True)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def norm01(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - mn) / (mx - mn)


TEAM_COL_CANDIDATES = [
    "TeamMemberUserNames",
    "TeamMemberUsernames",
    "TeamMemberUserName",
    "TeamMembers",
    "TeamMemberNames",
]

def read_first_csv_from_zip(zip_path: Path) -> Optional[pd.DataFrame]:
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csvs:
                return None
            with z.open(csvs[0]) as f:
                return pd.read_csv(f)
    except Exception:
        return None

def get_team_col(df: pd.DataFrame) -> Optional[str]:
    for c in TEAM_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None

def extract_user_team_from_zip(zip_path: Path, username: str) -> Optional[List[str]]:
    df = read_first_csv_from_zip(zip_path)
    if df is None or df.empty:
        return None

    team_col = get_team_col(df)
    if team_col is None:
        return None

    username_l = username.lower()
    s = df[team_col].astype(str).str.lower()

    # Avoid pandas warning: use NON-capturing groups (?:...)
    pat = rf"(?:^|[^a-z0-9_-]){re.escape(username_l)}(?:[^a-z0-9_-]|$)"
    mask = s.str.contains(pat, regex=True, na=False)

    if not mask.any():
        return None

    cell = str(df.loc[mask, team_col].iloc[0])
    txt = cell.replace("|", ",").replace(";", ",").replace("\n", ",").replace("\t", ",")
    parts = [p.strip() for p in txt.split(",") if p.strip()]
    parts = [p for p in parts if p.lower() != username_l]
    return sorted(set(parts))

def team_signature(mates: List[str]) -> Tuple[str, ...]:
    return tuple(sorted(set(mates)))

def load_user_competitions_slugs(competitions_dir: Path, username: str, max_comp: int) -> List[str]:
    p = competitions_dir / f"{username}_competitions_raw.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    if "competition_slug" not in df.columns:
        return []
    slugs = df["competition_slug"].dropna().astype(str).tolist()
    return slugs[:max_comp]


def compute_metrics_for_user(username: str, competition_slugs: List[str], leaderboards_dir: Path) -> Dict[str, float]:
    total = len(competition_slugs)
    if total == 0:
        return {}

    zips_present = 0
    found_rows = 0

    team_flags = []
    team_sizes = []
    signatures = []
    unique_mates = set()

    for slug in competition_slugs:
        zip_path = leaderboards_dir / f"{slug}.zip"
        if not zip_path.exists():
            continue

        zips_present += 1
        mates = extract_user_team_from_zip(zip_path, username)
        if mates is None:
            continue

        found_rows += 1
        if len(mates) > 0:
            team_flags.append(1)
            team_sizes.append(1 + len(mates))
            signatures.append(team_signature(mates))
            unique_mates.update(mates)
        else:
            team_flags.append(0)
            team_sizes.append(1)

    # Coverage split (KEY FIX)
    zip_coverage = (zips_present / total) if total else 0.0
    row_coverage = (found_rows / zips_present) if zips_present else 0.0  # "matching quality" on available ZIPs

    team_comps = int(sum(team_flags))
    team_ratio = (team_comps / found_rows) if found_rows else 0.0

    avg_team_size = (sum([ts for ts, tf in zip(team_sizes, team_flags) if tf == 1]) / team_comps) if team_comps else 1.0
    unique_mates_count = len(unique_mates)
    rotation = (unique_mates_count / team_comps) if team_comps else 0.0

    if signatures:
        c = Counter(signatures)
        stability = c.most_common(1)[0][1] / len(signatures)
    else:
        stability = 0.0

    return {
        "username": username,
        "n_competitions": float(total),
        "zips_present": float(zips_present),
        "found_rows": float(found_rows),
        "zip_coverage": float(zip_coverage),
        "row_coverage": float(row_coverage),
        "team_ratio": float(team_ratio),
        "rotation": float(rotation),
        "stability": float(stability),
        "avg_team_size": float(avg_team_size),
        "unique_mates": float(unique_mates_count),
        "n_team_comps": float(team_comps),
    }


def plot_heatmap(df_metrics: pd.DataFrame, out_path: Path, highlight: List[str]) -> None:
    cols = ["zip_coverage", "row_coverage", "team_ratio", "rotation", "stability", "avg_team_size", "unique_mates"]

    norm = df_metrics.copy()
    for c in cols:
        norm[c] = norm01(norm[c])

    mat = norm.set_index("username")[cols]

    plt.figure(figsize=(16, max(6, int(0.35 * mat.shape[0]) + 3)))
    ax = plt.gca()
    im = ax.imshow(mat.values, aspect="auto")

    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index.tolist())
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(
        ["ZIP cov.", "Row cov.", "Team ratio", "Rotation", "Stability", "Avg team size", "Unique mates"],
        rotation=20, ha="right"
    )
    ax.set_title("Top 25 Kaggle - Collaboration profile (from leaderboard ZIPs)\n(normalized per metric)")

    if highlight:
        idx_map = {u: i for i, u in enumerate(mat.index.tolist())}
        for u in highlight:
            if u in idx_map:
                y = idx_map[u]
                ax.add_patch(plt.Rectangle((-0.5, y - 0.5), mat.shape[1], 1, fill=False, linewidth=2))

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Normalized (0->1)")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", nargs="+", required=True)
    ap.add_argument("--competitions_dir", default="out/top10")
    ap.add_argument("--leaderboards_dir", default="out/leaderboards")
    ap.add_argument("--max_competitions_per_user", type=int, default=60)
    ap.add_argument("--outdir", default="out/figures_sq1")
    ap.add_argument("--workdir", default="out/top10")
    ap.add_argument("--highlight", nargs="*", default=[])
    # NEW filters
    ap.add_argument("--min_zips_present", type=int, default=3, help="Keep only users with at least N leaderboard ZIPs available")
    ap.add_argument("--min_row_coverage", type=float, default=0.05, help="On available ZIPs, require user found at least this fraction")
    args = ap.parse_args()

    users = [u.strip() for u in args.users]
    competitions_dir = Path(args.competitions_dir)
    leaderboards_dir = Path(args.leaderboards_dir)
    outdir = Path(args.outdir)
    workdir = Path(args.workdir)

    ensure_dir(outdir)
    ensure_dir(workdir)

    rows = []
    for u in users:
        slugs = load_user_competitions_slugs(competitions_dir, u, args.max_competitions_per_user)
        if not slugs:
            log(f"[WARN] {u}: no competitions file or empty slugs -> skipped")
            continue
        m = compute_metrics_for_user(u, slugs, leaderboards_dir)
        if not m:
            log(f"[WARN] {u}: no metrics -> skipped")
            continue
        rows.append(m)

    if not rows:
        raise SystemExit("[ERR] No usable users. Check competitions CSV files and leaderboard ZIP names.")

    df = pd.DataFrame(rows)
    metrics_csv = workdir / "top25_collab_metrics_from_zips.csv"
    df.to_csv(metrics_csv, index=False)
    log(f"[OK] Metrics CSV -> {metrics_csv}")

    # Filter: avoid “all violet” that are just “no ZIP coverage”
    df_f = df[(df["zips_present"] >= args.min_zips_present) & (df["row_coverage"] >= args.min_row_coverage)].copy()
    removed = sorted(set(df["username"]) - set(df_f["username"]))
    log(f"[INFO] min_zips_present={args.min_zips_present} min_row_coverage={args.min_row_coverage:.2f} kept={df_f.shape[0]} removed={removed}")

    if df_f.empty:
        raise SystemExit("[ERR] After filters, no users left. Lower thresholds or download more leaderboard ZIPs.")

    df_f = df_f.sort_values(["zips_present", "row_coverage", "team_ratio"], ascending=False)

    heatmap_path = outdir / "top25_collab_metrics_heatmap_from_zips_filtered.png"
    plot_heatmap(df_f, heatmap_path, args.highlight)
    log(f"[OK] Heatmap -> {heatmap_path}")
    print(f"[OK] Done. Heatmap: {heatmap_path}")


if __name__ == "__main__":
    main()
