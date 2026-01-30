import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

INPUT_CSV = Path("results/datasource_references_history.csv")
OUTPUT_IMAGE = Path("results/datasource_state_evolution.png")
START_DATE = "2025-01-01"

def plot_state_evolution():
    if not INPUT_CSV.exists():
        print(f"Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    df['date'] = pd.to_datetime(df['date'])
    
    df = df.sort_values(['date', 'commit'])

    file_states = {} 
    history = []
    
    for (commit_hash, commit_date), group in df.groupby(['commit', 'date'], sort=False):
        
        files_in_commit = group.groupby('file')['count'].sum()

        for filename, new_count in files_in_commit.items():
            file_states[filename] = new_count

        total_references = sum(file_states.values())
        history.append({'date': commit_date, 'total_refs': total_references})

    df_history = pd.DataFrame(history)
    
    if df_history.empty:
        print("No history found.")
        return
    df_2025 = df_history[df_history['date'] >= pd.to_datetime(START_DATE)].copy()
    
    if df_2025.empty:
        print(f"No data found after {START_DATE}.")
        return

    df_daily = df_2025.set_index('date').resample('D').last().ffill()

    plt.figure(figsize=(12, 6))

    plt.plot(df_daily.index, df_daily['total_refs'], 
             drawstyle='steps-post', linewidth=2.5, color='#d62728')

    plt.fill_between(df_daily.index, df_daily['total_refs'], step='post', color='#d62728', alpha=0.1)

    plt.title('Evolution of Data Source References (State of Codebase)', fontsize=16, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of References Present', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) 
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    plt.xticks(rotation=45)

    plt.ylim(bottom=0)

    plt.tight_layout()
    
    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_IMAGE, dpi=150)
    print(f"Plot saved to: {OUTPUT_IMAGE.resolve()}")
    plt.show()

if __name__ == "__main__":
    plot_state_evolution()