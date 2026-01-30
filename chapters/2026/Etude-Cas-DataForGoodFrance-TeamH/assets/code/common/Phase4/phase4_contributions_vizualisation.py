import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"

INPUT_COMMITS = RESULTS_DIR / "commits_detailed.csv"
OUTPUT_IMG = RESULTS_DIR / "contributions_timeline_jobs.png"

def clean_job_title(val):
    """Cleans up the job title (removes ?, nan, None)."""
    val = str(val).strip()
    if val.lower() in ['nan', 'none', '', '?', 'null']:
        return None
    return val.replace('?', '').strip()

def main(jobs_csv_path):
    if not INPUT_COMMITS.exists():
        print(f"Error: Commits file not found at {INPUT_COMMITS}")
        return

    df = pd.read_csv(INPUT_COMMITS)
    
    df['commit_date'] = pd.to_datetime(df['commit_date'], utc=True)

    email_to_label = {}
    
    if jobs_csv_path:
        path_obj = Path(jobs_csv_path)
        if path_obj.exists():
            print(f"Loading jobs from: {path_obj.name}")
            try:
                df_jobs = pd.read_csv(path_obj)
                
                if 'author_email' not in df_jobs.columns and len(df_jobs.columns) == 1:
                    df_jobs = pd.read_csv(path_obj, sep=';')

                if 'author_email' in df_jobs.columns:
                    for _, row in df_jobs.iterrows():
                        email = str(row['author_email']).strip()
                        name = str(row['author_name']).strip()
                        
                        raw_job = row.get('job', None)
                        job = clean_job_title(raw_job)
                        
                        if job:
                            label = f"{name} ({job})"
                        else:
                            label = name
                        
                        email_to_label[email] = label
                else:
                    print(f" Warning: Column 'author_email' not found in {jobs_csv_path}.")
                    print(f"Columns found: {df_jobs.columns.tolist()}")

            except Exception as e:
                print(f"Error reading jobs CSV: {e}")
        else:
            print(f"File not found: {jobs_csv_path}")
    else:
        print("No jobs file provided. Using default names.")


    df['display_name'] = df['author_email'].map(email_to_label).fillna(df['author_name'])



    
    # Group by Month
    monthly_activity = df.groupby([pd.Grouper(key='commit_date', freq='ME'), 'display_name']).size().unstack(fill_value=0)

    # Filter Top Contributors (Top 15 for better visibility)
    total_commits = monthly_activity.sum().sort_values(ascending=False)
    top_authors = total_commits.head(15).index.tolist()
    
    df_plot = monthly_activity[top_authors].copy()
    
    others = monthly_activity.columns.difference(top_authors)
    if not others.empty:
        df_plot['Others'] = monthly_activity[others].sum(axis=1)
    
    
    plt.figure(figsize=(16, 10))
    
    colors = sns.color_palette("tab20", len(df_plot.columns))
    
    plt.stackplot(df_plot.index, df_plot.T, labels=df_plot.columns, colors=colors, alpha=0.85)
    
    plt.title('Project Timeline: Contributors & Roles', fontsize=18)
    plt.ylabel('Commits / Month', fontsize=12)
    plt.xlabel('Timeline', fontsize=12)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Who is who?", fontsize=10)
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"Graph saved to: {OUTPUT_IMG}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Project Timeline with Job Titles.")
    parser.add_argument("jobs_file", nargs='?', help="Path to the CSV file containing author jobs (optional).")
    
    args = parser.parse_args()
    
    main(args.jobs_file)