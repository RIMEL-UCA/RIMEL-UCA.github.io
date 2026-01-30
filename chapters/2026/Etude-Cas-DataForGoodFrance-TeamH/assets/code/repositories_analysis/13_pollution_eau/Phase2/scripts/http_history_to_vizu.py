import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_CSV = "results/http_references_history.csv"
OUTPUT_IMAGE = "results/http_references_curve.png"

def main():
    if not Path(INPUT_CSV).exists():
        print(f"Error: File {INPUT_CSV} not found. Please run the extraction script first.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    df['date'] = pd.to_datetime(df['date'])

    commit_counts = df.groupby(['file', 'date', 'commit']).size().reset_index(name='count')

    daily_counts = commit_counts.groupby(['file', 'date'])['count'].max()

    history = daily_counts.unstack(level=0)

    all_dates = pd.date_range(start=history.index.min(), end=history.index.max(), freq='D')
    history = history.reindex(all_dates)
    history = history.ffill()
    history = history.fillna(0)
    total_references = history.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(total_references.index, total_references.values, label='Total HTTP References', color='#007acc', linewidth=2)
    
    plt.title("Evolution of HTTP References in Codebase", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Number of References", fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Format date on x-axis nicely
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    print(f"Curve generated: {OUTPUT_IMAGE}")
    plt.show()

if __name__ == "__main__":
    main()