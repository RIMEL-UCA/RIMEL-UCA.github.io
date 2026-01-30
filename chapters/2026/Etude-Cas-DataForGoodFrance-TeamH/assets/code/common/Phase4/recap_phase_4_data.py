import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from utils.csv_helper import CsvHelper
# --- CONFIGURATION PATHS ---
BASE_DIR = Path(__file__).parent.parent.parent

REPO_PATHS = [
    "repositories_analysis/shiftdataportal/Phase4/results",
    "repositories_analysis/13_pollution_eau/Phase4/results",
    "repositories_analysis/13_reveler_inegalites_cinema/Phase4/results",
    "repositories_analysis/13_eclaireur_public/Phase4/results"
]

COMMITS_FILENAME = "commits_detailed.csv"
JOBS_FILENAME = "contributors_jobs_inferred.csv"

OUTPUT_IMG = BASE_DIR / "results" / "average_data_contributors_curve.png"


# --- DATA LOADING ---
def load_all_commits_and_jobs(repo_paths):
    commits_dfs = []
    email_to_job = {}

    for repo_path in repo_paths:
        commits_file = BASE_DIR / repo_path / COMMITS_FILENAME
        jobs_file = BASE_DIR / repo_path / JOBS_FILENAME

        if not commits_file.exists() or not jobs_file.exists():
            continue

        # --- Commits ---
        df_commits = CsvHelper.read_and_validate(
            commits_file,
            required_columns=['commit_date', 'author_email']
        )
        if df_commits is None or df_commits.empty:
            continue
        df_commits['repository'] = repo_path.split('/')[0]
        commits_dfs.append(df_commits)

        # --- Jobs ---
        df_jobs = CsvHelper.read_and_validate(
            jobs_file,
            required_columns=['author_email']
        )
        if df_jobs is None or df_jobs.empty:
            continue

        df_jobs['author_email'] = df_jobs['author_email'].astype(str).str.lower().str.strip()

        for _, row in df_jobs.iterrows():
            email = row['author_email']
            job = CsvHelper.clean_cell_value(row.get('job', 'Unknown'))
            email_to_job[email] = job

    if not commits_dfs:
        return pd.DataFrame(), {}

    return pd.concat(commits_dfs, ignore_index=True), email_to_job


# --- MAIN ---
def main():
    df_commits, email_to_job = load_all_commits_and_jobs(REPO_PATHS)
    if df_commits.empty:
        print("Aucun commit trouvé.")
        return

    # --- Normalize fields ---
    df_commits['commit_date'] = pd.to_datetime(df_commits['commit_date'], utc=True)
    df_commits['author_email'] = df_commits['author_email'].astype(str).str.lower().str.strip()
    df_commits['job'] = df_commits['author_email'].map(email_to_job).fillna('Unknown')

    # --- Keep only 'data' contributors ---
    df_commits = df_commits[df_commits['job'].str.contains('data', case=False)]
    if df_commits.empty:
        print("Aucun contributeur avec un métier contenant 'data'.")
        return

    # --- Temporal normalization ---
    df_commits['project_start'] = df_commits.groupby('repository')['commit_date'].transform('min')
    df_commits['project_end'] = df_commits.groupby('repository')['commit_date'].transform('max')
    df_commits['project_duration_days'] = ((df_commits['project_end'] - df_commits['project_start']).dt.days).clip(lower=1)
    df_commits['relative_time'] = ((df_commits['commit_date'] - df_commits['project_start']).dt.days / df_commits['project_duration_days'])
    df_commits['project_month'] = (df_commits['relative_time'] * 12).astype(int).clip(0, 11) + 1

    # --- Count unique contributors per month and project ---
    monthly_contributors = (
        df_commits.groupby(['repository', 'project_month'])['author_email']
        .nunique()
        .reset_index(name='unique_data_contributors')
    )

    # --- Average across projects ---
    average_contributors = (
        monthly_contributors.groupby('project_month')['unique_data_contributors']
        .mean()
        .reset_index()
    )

    # --- Fill missing months ---
    all_months = pd.DataFrame({'project_month': range(1, 13)})
    average_contributors = all_months.merge(average_contributors, on='project_month', how='left')
    average_contributors['unique_data_contributors'] = average_contributors['unique_data_contributors'].fillna(0)

    # --- VISUALIZATION ---
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=average_contributors,
        x='project_month',
        y='unique_data_contributors',
        color='blue',   # couleur fixe
        marker='o'      # points sur la courbe
    )

    plt.title("Nombre moyen de Data Scientist/Engineer par mois pour tous les projets", fontsize=16)
    plt.ylabel("Nombre moyen de contributeurs Data Scientist/Engineer", fontsize=12)
    plt.xlabel("Mois du projet (année normalisée)", fontsize=12)
    plt.xticks(ticks=range(1, 13))
    plt.grid(axis='y', linestyle='-.', alpha=0.3)
    plt.tight_layout()

    OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"Graphique généré : {OUTPUT_IMG}")



if __name__ == "__main__":
    main()
