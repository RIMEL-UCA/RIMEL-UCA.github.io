import argparse
from pathlib import Path
from typing import List, Optional
import zipfile

import pandas as pd
import matplotlib.pyplot as plt


def log(msg: str) -> None:
    print(f"[LOG] {msg}", flush=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------
# Lecture des leaderboards locaux (.zip -> leaderboard.csv)
# --------------------------------------------------------

# On met tout en minuscule ici, et on fera un mapping insensible à la casse
POSSIBLE_RANK_COLS = ["rank"]
POSSIBLE_TEAMSIZE_COLS = [
    "teammembercount",
    "teammembercounts",
    "teamsize",
    "team_size",
    "teammembers",          # parfois noms concaténés
    "teammemberusernames",  # couvre 'TeamMemberUserNames'
]


def infer_team_size_from_string(col: pd.Series) -> pd.Series:
    """
    Quand on n'a pas TeamMemberCount mais une colonne texte
    type 'TeamMembers' ou 'TeamMemberUserNames', on essaie
    de compter la taille de l'équipe en splittant la chaîne.
    Heuristique simple : split sur ',' ou ';' ou '|' et
    on compte les morceaux non vides.
    """
    def _count(x):
        if pd.isna(x):
            return 1  # fallback: on considère solo
        s = str(x)
        # on remplace les séparateurs les plus courants par une virgule
        for sep in ["|", ";", "\n", "\t"]:
            s = s.replace(sep, ",")
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return max(1, len(parts))  # au moins 1

    return col.apply(_count)


def load_leaderboard_from_zip(zip_path: Path) -> Optional[pd.DataFrame]:
    """
    Ouvre un fichier .zip de Kaggle et essaie d'en extraire un DataFrame
    avec au moins : Rank, team_size, leaderboard_size, competition_slug.
    Retourne None si on ne peut pas.
    """
    slug = zip_path.stem  # ex: "optiver-trading-at-the-close"
    log(f"[INFO] Lecture du leaderboard local pour {slug} depuis {zip_path.name}")

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            # On cherche un fichier .csv (souvent 'leaderboard.csv')
            csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                log(f"[WARN] Aucun fichier CSV trouvé dans {zip_path.name}, on skip.")
                return None

            # On prend le premier CSV
            csv_name = csv_names[0]
            log(f"[INFO] -> CSV détecté dans l'archive : {csv_name}")
            with z.open(csv_name) as f:
                df = pd.read_csv(f)
    except Exception as e:
        log(f"[ERR] Impossible de lire {zip_path.name} : {e}")
        return None

    if df.empty:
        log(f"[WARN] CSV vide dans {zip_path.name}, on skip.")
        return None

    # Mapping insensible à la casse : lower -> nom réel
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    # 1) Rank
    rank_col = None
    for key in POSSIBLE_RANK_COLS:
        if key in lower_map:
            rank_col = lower_map[key]
            break

    if rank_col is None:
        # On essaye de reconstruire un rank si possible
        if "score" in lower_map:
            score_col = lower_map["score"]
            log(
                f"[WARN] Colonne 'Rank' absente dans {zip_path.name}, "
                f"on reconstruit un rang en triant par {score_col} (approximatif)."
            )
            # On suppose 'higher is better'. Si ce n'est pas le cas pour
            # une compète, l'ordre sera inversé mais ça reste utilisable
            df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
            df["Rank"] = df.index + 1
            rank_col = "Rank"
        else:
            log(f"[ERR] Impossible de trouver ou reconstruire 'Rank' dans {zip_path.name}, on skip.")
            return None

    # 2) team_size
    team_size_col = None
    for key in POSSIBLE_TEAMSIZE_COLS:
        if key in lower_map:
            team_size_col = lower_map[key]
            break

    if team_size_col is None:
        log(
            f"[WARN] Aucune colonne évidente pour la taille d'équipe dans {zip_path.name}. "
            f"On considère tout le monde comme solo (approx)."
        )
        df["team_size"] = 1
    else:
        # Si la colonne est déjà numérique : parfait
        if pd.api.types.is_numeric_dtype(df[team_size_col]):
            df["team_size"] = df[team_size_col].fillna(1).astype(int)
        else:
            # Sinon on essaie de compter les noms dans la chaine
            log(
                f"[INFO] Inférence de la taille d'équipe à partir de '{team_size_col}' "
                f"dans {zip_path.name}."
            )
            df["team_size"] = infer_team_size_from_string(df[team_size_col])

    # 3) filtre des lignes "valides"
    df = df[df[rank_col].notna()]
    df = df[df["team_size"].notna()]
    if df.empty:
        log(f"[WARN] Après filtrage, plus de lignes valides pour {zip_path.name}.")
        return None

    df["rank"] = df[rank_col].astype(int)
    df["leaderboard_size"] = len(df)
    df["competition_slug"] = slug

    return df[["competition_slug", "rank", "team_size", "leaderboard_size"]]


# --------------------------------------------------------
# Agrégation & déciles
# --------------------------------------------------------

def compute_deciles(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    À partir de tous les leaderboards concaténés,
    calcule pour chaque décile (0-9) la proportion solo / team.
    """
    if df_all.empty:
        return pd.DataFrame()

    df = df_all.copy()
    df["is_solo"] = df["team_size"] == 1
    df["is_team"] = df["team_size"] > 1

    # Décile = floor(10 * (rank-1) / leaderboard_size)
    df["decile"] = ((df["rank"] - 1) * 10 / df["leaderboard_size"]).astype(int)
    df["decile"] = df["decile"].clip(0, 9)

    grouped = df.groupby("decile").agg(
        total=("rank", "count"),
        solo=("is_solo", "sum"),
        team=("is_team", "sum"),
        num_competitions=("competition_slug", "nunique"),
    )
    grouped["solo_ratio"] = grouped["solo"] / grouped["total"]
    grouped["team_ratio"] = grouped["team"] / grouped["total"]

    grouped.reset_index(inplace=True)
    return grouped


def plot_deciles(grouped: pd.DataFrame, outdir: Path) -> None:
    if grouped.empty:
        log("[ERR] Pas de données pour les déciles, impossible de tracer.")
        return

    ensure_dir(outdir)

    grouped = grouped.sort_values("decile")
    labels = [f"{d*10}-{(d+1)*10}%" for d in grouped["decile"]]

    solo = grouped["solo_ratio"].values
    team = grouped["team_ratio"].values

    x = range(len(labels))

    plt.figure(figsize=(10, 6))
    plt.bar(x, solo, label="Solo")
    plt.bar(x, team, bottom=solo, label="Équipe")

    plt.xticks(list(x), labels, rotation=45)
    plt.ylabel("Proportion")
    plt.xlabel("Décile de classement global (0% = top du leaderboard)")
    plt.title("Répartition Solo vs Équipe par décile de leaderboard\n(agrégé sur plusieurs compétitions)")
    plt.legend()
    plt.tight_layout()

    out_path = outdir / "solo_vs_team_by_decile.png"
    plt.savefig(out_path)
    plt.close()
    log(f"[OK] Graphique solo vs équipe par décile -> {out_path}")


# --------------------------------------------------------
# main : parcours des .zip locaux
# --------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--leaderboards_dir",
        default="out/leaderboards",
        help="Dossier contenant les .zip téléchargés depuis Kaggle (bouton Download du leaderboard).",
    )
    parser.add_argument(
        "--outdir",
        default="out/figures_sq1",
        help="Dossier de sortie pour les figures et éventuels CSV agrégés.",
    )
    parser.add_argument(
        "--min_leaderboard_size",
        type=int,
        default=50,
        help="Taille minimale d'un leaderboard pour être pris en compte (par défaut 50 lignes).",
    )
    args = parser.parse_args()

    leaderboards_dir = Path(args.leaderboards_dir)
    outdir = Path(args.outdir)

    if not leaderboards_dir.exists():
        raise SystemExit(f"{leaderboards_dir} introuvable. Place les .zip de leaderboards dedans.")

    zip_files = sorted(leaderboards_dir.glob("*.zip"))
    if not zip_files:
        raise SystemExit(
            f"Aucun fichier .zip trouvé dans {leaderboards_dir}. "
            f"Télécharge quelques leaderboards depuis Kaggle (bouton Download)."
        )

    all_rows: List[pd.DataFrame] = []
    used_slugs: List[str] = []

    for zip_path in zip_files:
        df_lb = load_leaderboard_from_zip(zip_path)
        if df_lb is None or df_lb.empty:
            continue

        # Filtre par taille minimale
        size = df_lb["leaderboard_size"].iloc[0]
        if size < args.min_leaderboard_size:
            log(f"[INFO] Leaderboard {zip_path.name} trop petit ({size} lignes), on le skip.")
            continue

        all_rows.append(df_lb)
        used_slugs.append(df_lb["competition_slug"].iloc[0])

    if not all_rows:
        log("[ERR] Aucun leaderboard exploitable après filtrage.")
        return

    df_all = pd.concat(all_rows, ignore_index=True)
    log(f"[OK] {len(df_all)} lignes agrégées sur {len(used_slugs)} compétitions.")
    log(f"[INFO] Compétitions utilisées : {', '.join(used_slugs)}")

    grouped = compute_deciles(df_all)

    # On sauvegarde aussi la table agrégée pour debug / rapport
    ensure_dir(outdir)
    grouped.to_csv(outdir / "solo_vs_team_deciles_agg.csv", index=False)
    log(f"[OK] Table agrégée par décile -> {outdir / 'solo_vs_team_deciles_agg.csv'}")

    plot_deciles(grouped, outdir)

    print("[OK] Visualisation solo vs équipe par décile générée (offline, à partir des CSV locaux).")


if __name__ == "__main__":
    main()
