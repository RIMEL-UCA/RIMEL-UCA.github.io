import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

INPUT_CSV = Path("results/data_preparation_files_history.csv")
OUTPUT_IMAGE = Path("results/data_preparation_files_count_evolution.png")
START_DATE = "2023-01-01"  # Ajustez selon vos besoins


def plot_data_files_evolution():
    """Plot the evolution of data preparation files count over time."""
    
    if not INPUT_CSV.exists():
        print(f"❌ Error: {INPUT_CSV} not found.")
        print(f"   Please run track_data_files_evolution.py first.")
        return

    try:
        df = pd.read_csv(INPUT_CSV)
    except pd.errors.EmptyDataError:
        print("❌ CSV is empty.")
        return

    if df.empty:
        print("❌ No data in CSV.")
        return

    print(f"[+] Loaded {len(df)} changes from {INPUT_CSV}")
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'commit'])

    # Calculate cumulative file count
    history = []
    current_count = 0
    seen_files = set()

    for (commit_hash, commit_date), group in df.groupby(['commit', 'date'], sort=False):
        # Count adds and deletes
        for _, row in group.iterrows():
            file_path = row['file']
            change_type = row['change_type']
            
            if change_type == 'ADD':
                if file_path not in seen_files:
                    current_count += 1
                    seen_files.add(file_path)
            elif change_type == 'DELETE':
                if file_path in seen_files:
                    current_count -= 1
                    seen_files.discard(file_path)
            elif change_type == 'RENAME':
                # For rename, old file is implicitly removed, new is added
                # This is handled by PyDriller through old_path/new_path
                pass
        
        history.append({'date': commit_date, 'count': current_count})

    df_history = pd.DataFrame(history)
    
    if df_history.empty:
        print("❌ No history found.")
        return
    
    # Resample to daily and forward fill
    df_daily = df_history.set_index('date').resample('D').last().ffill()
    
    # Filter by start date
    df_daily = df_daily[df_daily.index >= pd.to_datetime(START_DATE)]
    
    if df_daily.empty:
        print(f"❌ No data after {START_DATE}")
        return

    print(f"[+] Plotting from {df_daily.index[0].date()} to {df_daily.index[-1].date()}")
    print(f"[+] Max count: {df_daily['count'].max():.0f} files")

    # Create the plot
    plt.figure(figsize=(14, 7))
    
    # Main line with steps
    plt.plot(df_daily.index, df_daily['count'], 
             drawstyle='steps-post', 
             linewidth=2.5, 
             color='#3498db',  # Blue
             label='Nombre de fichiers')

    # Fill area under the curve
    plt.fill_between(df_daily.index, df_daily['count'], 
                     step='post', 
                     color='#3498db', 
                     alpha=0.2)

    # Title and labels
    plt.title('Évolution du nombre de scripts de préparation de données\nThe Shift Data Portal',
              fontsize=16, 
              fontweight='bold',
              pad=20)
    plt.xlabel('Date', fontsize=13)
    plt.ylabel('Nombre de fichiers Python/Notebooks', fontsize=13)
    
    # Grid
    plt.grid(True, linestyle='--', alpha=0.4)

    # Configure X axis
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45, ha='right')
    
    # Configure Y axis (integers only)
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylim(bottom=0)
    
    # Add legend
    plt.legend(loc='upper left', fontsize=11)
    
    # Add statistics box
    stats_text = f"Max: {df_daily['count'].max():.0f} fichiers\n"
    stats_text += f"Final: {df_daily['count'].iloc[-1]:.0f} fichiers\n"
    stats_text += f"Période: {(df_daily.index[-1] - df_daily.index[0]).days} jours"
    
    plt.text(0.98, 0.02, stats_text,
             transform=ax.transAxes,
             fontsize=10,
             verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    
    # Save
    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_IMAGE, dpi=200, bbox_inches='tight')
    print(f"[✓] Plot saved to: {OUTPUT_IMAGE.resolve()}")
    
    # Show (might not work in WSL without X server)
    try:
        plt.show()
    except Exception as e:
        print(f"[i] Could not display plot: {e}")
        print(f"    But the file was saved successfully!")
    finally:
        plt.close()


def print_summary():
    """Print a summary of the evolution."""
    if not INPUT_CSV.exists():
        return
    
    df = pd.read_csv(INPUT_CSV)
    df['date'] = pd.to_datetime(df['date'])
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n📅 Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"📊 Total changes: {len(df)}")
    
    # Change types
    print(f"\n🔄 Change types:")
    for change_type, count in df['change_type'].value_counts().items():
        pct = 100 * count / len(df)
        print(f"   {change_type:10s}: {count:4d} ({pct:5.1f}%)")
    
    # Most active files
    print(f"\n📁 Most modified files (top 5):")
    for file, count in df['file'].value_counts().head(5).items():
        short_name = Path(file).name
        print(f"   {count:3d} changes - {short_name}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DATA PREPARATION FILES EVOLUTION PLOT")
    print("="*70 + "\n")
    
    plot_data_files_evolution()
    print_summary()