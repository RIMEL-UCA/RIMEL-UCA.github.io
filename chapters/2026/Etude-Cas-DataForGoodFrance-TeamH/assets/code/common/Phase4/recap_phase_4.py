import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# --- CONFIGURATION PATHS ---


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from utils.csv_helper import CsvHelper

BASE_DIR = Path(__file__).parent.parent.parent

REPO_PATHS = [
    "repositories_analysis/shiftdataportal/Phase4/results",
    "repositories_analysis/13_pollution_eau/Phase4/results",
    "repositories_analysis/13_reveler_inegalites_cinema/Phase4/results",
    "repositories_analysis/13_eclaireur_public/Phase4/results"
]

COMMITS_FILENAME = "commits_detailed.csv"
JOBS_FILENAME = "contributors_jobs_inferred.csv"

OUTPUT_IMG = BASE_DIR / "results" / "project_steps_jobs_bar_normalized.png"


# --- DATA LOADING ---

def load_all_commits_and_jobs(repo_paths):
    commits_dfs = []
    email_to_job = {}

    for repo_path in repo_paths:
        commits_file = BASE_DIR / repo_path / COMMITS_FILENAME
        jobs_file = BASE_DIR / repo_path / JOBS_FILENAME

        # --- Commits ---
        if not commits_file.exists():
            print(f"Fichier commits introuvable : {commits_file}")
            continue

        df_commits = CsvHelper.read_and_validate(
            commits_file,
            required_columns=['commit_date', 'author_email']
        )

        if df_commits is None or df_commits.empty:
            continue

        df_commits['repository'] = repo_path.split('/')[0]
        commits_dfs.append(df_commits)

        # --- Jobs ---
        if not jobs_file.exists():
            print(f"Fichier métiers introuvable : {jobs_file}")
            continue

        df_jobs = CsvHelper.read_and_validate(
            jobs_file,
            required_columns=['author_email']
        )

        if df_jobs is None or df_jobs.empty:
            continue

        df_jobs['author_email'] = (
            df_jobs['author_email']
            .astype(str)
            .str.lower()
            .str.strip()
        )

        for _, row in df_jobs.iterrows():
            email = row['author_email']
            raw_job = row.get('job', 'Unknown')
            job = CsvHelper.clean_cell_value(raw_job)
            email_to_job[email] = job

    if not commits_dfs:
        return pd.DataFrame(), {}

    return pd.concat(commits_dfs, ignore_index=True), email_to_job


# --- MAIN ---

def main():
    # --- Load data ---
    df_commits, email_to_job = load_all_commits_and_jobs(REPO_PATHS)

    if df_commits.empty:
        print("Aucun commit trouvé.")
        return

    # --- Normalize fields ---
    df_commits['commit_date'] = pd.to_datetime(df_commits['commit_date'], utc=True)
    df_commits['author_email'] = (
        df_commits['author_email']
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df_commits['job'] = (
        df_commits['author_email']
        .map(email_to_job)
        .fillna('Unknown')
    )

    # --- Temporal normalization (project -> 1 year) ---
    df_commits['project_start'] = (
        df_commits.groupby('repository')['commit_date']
        .transform('min')
    )

    df_commits['project_end'] = (
        df_commits.groupby('repository')['commit_date']
        .transform('max')
    )

    df_commits['project_duration_days'] = (
        (df_commits['project_end'] - df_commits['project_start'])
        .dt.days
        .clip(lower=1)
    )

    df_commits['relative_time'] = (
        (df_commits['commit_date'] - df_commits['project_start'])
        .dt.days
        / df_commits['project_duration_days']
    )

    # 12 months
    df_commits['project_month'] = (
        (df_commits['relative_time'] * 12)
        .astype(int)
        .clip(0, 11) + 1
    )

    # --- AGGREGATION PER PROJECT ---
    monthly_counts = (
        df_commits
        .groupby(['repository', 'project_month', 'job'])
        .size()
        .reset_index(name='count')
    )

    # --- AVERAGE ACROSS PROJECTS ---
    average_counts = (
        monthly_counts
        .groupby(['project_month', 'job'])['count']
        .mean()
        .reset_index()
    )

    if average_counts.empty:
        print("Pas assez de données agrégées.")
        return

    # --- DOMINANT JOB PER MONTH ---
    idx = average_counts.groupby('project_month')['count'].idxmax()
    majority_df = (
        average_counts
        .loc[idx]
        .sort_values('project_month')
    )

    majority_df['phase'] = (
        "Mois " + majority_df['project_month'].astype(str)
    )

    # --- VISUALIZATION ---
    plt.figure(figsize=(16, 9))

    sns.barplot(
        data=majority_df,
        x='phase',
        y='count',
        hue='job',
        dodge=False,
        palette='bright'
    )

    plt.title(
        "Évolution moyenne des contributions par métier pour tous les projets",
        fontsize=18
    )
    plt.ylabel("Nombre moyen de commits", fontsize=12)
    plt.xlabel("Phase du projet (année normalisée)", fontsize=12)

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.legend(title="Métier / Rôle", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_IMG, dpi=300)

    print(f"Graphique généré : {OUTPUT_IMG}")


if __name__ == "__main__":
    main()
