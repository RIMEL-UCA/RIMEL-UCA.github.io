import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys

# --- CONFIGURATION IMPORTS ---
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from utils.csv_helper import CsvHelper

# --- CONFIGURATION PATHS ---
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"

INPUT_COMMITS = RESULTS_DIR / "commits_detailed.csv"
OUTPUT_IMG = RESULTS_DIR / "project_steps_jobs_bar.png"

def main(jobs_csv_path):
    df_commits = CsvHelper.read_and_validate(INPUT_COMMITS, required_columns=['commit_date', 'author_email'])
    
    if df_commits is None:
        return


    df_commits['commit_date'] = pd.to_datetime(df_commits['commit_date'], utc=True)
    df_commits['author_email'] = df_commits['author_email'].astype(str).str.lower().str.strip()

    email_to_job = {}
    target_jobs_file = Path(jobs_csv_path)
    
    df_jobs = CsvHelper.read_and_validate(target_jobs_file, required_columns=['author_email'])
    
    if df_jobs is not None:
        df_jobs['author_email'] = df_jobs['author_email'].astype(str).str.lower().str.strip()
        
        for _, row in df_jobs.iterrows():
            email = row['author_email']
            raw_job = row.get('job', 'Unknown')
            job = CsvHelper.clean_cell_value(raw_job)
            email_to_job[email] = job
    else:
        print("Impossible de lire le fichier métiers. Arrêt.")
        return

    df_commits['job'] = df_commits['author_email'].map(email_to_job).fillna('Unknown')

    monthly_counts = df_commits.groupby([pd.Grouper(key='commit_date', freq='ME'), 'job']).size().reset_index(name='count')
    
    if monthly_counts.empty:
        print("Pas assez de données.")
        return

    idx = monthly_counts.groupby('commit_date')['count'].idxmax()
    majority_df = monthly_counts.loc[idx].sort_values('commit_date')

    majority_df = majority_df[majority_df['count'] > 0]

    # ==============================================================================
    # VISUALISATION (BAR CHART)
    # ==============================================================================

    plt.figure(figsize=(16, 9))
    
    majority_df['month_str'] = majority_df['commit_date'].dt.strftime('%Y-%m')

    ax = sns.barplot(
        data=majority_df,
        x='month_str',
        y='count',
        hue='job',
        dodge=False, 
        palette='bright'
    )

    plt.title("Phases du Projet : Métier Dominant par Mois", fontsize=18)
    plt.ylabel("Volume d'activité du métier dominant (Commits)", fontsize=12)
    plt.xlabel("Chronologie", fontsize=12)

    plt.xticks(rotation=45, ha='right', fontsize=10)

    plt.legend(title="Métier / Rôle", bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"Graphique sauvegardé : {OUTPUT_IMG}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Affiche les étapes du projet basées sur le métier dominant.")
    parser.add_argument("jobs_file", help="Chemin OBLIGATOIRE vers le CSV des métiers (author_email, job)")
    
    args = parser.parse_args()
    
    main(args.jobs_file)