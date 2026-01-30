import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

INPUT_CSV = Path("../results/enrichment_history.csv")
OUTPUT_IMAGE = Path("../results/enrichment_count_evolution.png")
START_DATE = "2025-01-01"


def plot_enrichment_evolution():
    if not INPUT_CSV.exists():
        print(f"Error: {INPUT_CSV} not found. Please run track_enrichment_history.py first.")
        return

    try:
        df = pd.read_csv(INPUT_CSV)
    except pd.errors.EmptyDataError:
        print("CSV is empty.")
        return

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'commit'])


    history = []
    current_count = 0

    for (commit_hash, commit_date), group in df.groupby(['commit', 'date'], sort=False):
        
        adds = len(group[group['change_type'] == 'ADD'])
        deletes = len(group[group['change_type'] == 'DELETE'])
        
        net_change = adds - deletes
        current_count += net_change
        
        history.append({'date': commit_date, 'count': current_count})

    df_history = pd.DataFrame(history)
    
    if df_history.empty:
        print("No history found.")
        return
    df_daily = df_history.set_index('date').resample('D').last().ffill()
    
    df_daily = df_daily[df_daily.index >= pd.to_datetime(START_DATE)]

    plt.figure(figsize=(12, 6))
    
    plt.plot(df_daily.index, df_daily['count'], 
             drawstyle='steps-post', linewidth=2.5, color='#2ca02c') # Green

    plt.fill_between(df_daily.index, df_daily['count'], step='post', color='#2ca02c', alpha=0.15)

    plt.title('Evolution of Enrichment Modules Count', fontsize=16, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Enrichment Files', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) 
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    plt.xticks(rotation=45)
    
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylim(bottom=0)

    plt.tight_layout()
    
    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_IMAGE, dpi=150)
    print(f"Plot saved to: {OUTPUT_IMAGE.resolve()}")
    plt.show()

if __name__ == "__main__":
    plot_enrichment_evolution()