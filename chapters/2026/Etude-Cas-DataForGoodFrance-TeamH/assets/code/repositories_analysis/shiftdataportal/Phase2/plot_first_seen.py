import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

def load_and_prepare(sources_csv, modes_csv):
    """Charge et prépare les données."""
    df_sources = pd.read_csv(sources_csv)
    df_sources["label"] = df_sources["source"]
    df_sources["category"] = "Source"

    df_modes = pd.read_csv(modes_csv)
    df_modes["label"] = df_modes["access_mode"]
    df_modes["category"] = "Mode d'accès"

    df = pd.concat(
        [
            df_sources[["label", "first_commit_date", "category"]],
            df_modes[["label", "first_commit_date", "category"]],
        ],
        ignore_index=True
    )

    df["first_commit_date"] = pd.to_datetime(df["first_commit_date"])
    df = df.sort_values("first_commit_date")
    
    return df


def plot_timeline_gantt_style(df, output_path):
    """Visualisation style Gantt avec groupement par période."""
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Séparer sources et modes
    df_sources = df[df["category"] == "Source"].copy()
    df_modes = df[df["category"] == "Mode d'accès"].copy()
    
    # Créer des positions Y pour chaque élément
    y_pos = 0
    y_positions = {}
    y_labels = []
    y_ticks = []
    
    # D'abord les sources
    if not df_sources.empty:
        y_labels.append("═══ SOURCES ═══")
        y_ticks.append(y_pos)
        y_pos += 1
        
        for idx, row in df_sources.iterrows():
            y_positions[row["label"]] = y_pos
            y_labels.append(row["label"])
            y_ticks.append(y_pos)
            y_pos += 1
        
        y_pos += 1  # Espace entre les sections
    
    # Ensuite les modes d'accès
    if not df_modes.empty:
        y_labels.append("═══ MODES D'ACCÈS ═══")
        y_ticks.append(y_pos)
        y_pos += 1
        
        for idx, row in df_modes.iterrows():
            y_positions[row["label"]] = y_pos
            y_labels.append(row["label"])
            y_ticks.append(y_pos)
            y_pos += 1
    
    # Tracer les points
    colors = {"Source": "#3498db", "Mode d'accès": "#e74c3c"}
    
    for idx, row in df.iterrows():
        y = y_positions[row["label"]]
        color = colors[row["category"]]
        
        ax.scatter(row["first_commit_date"], y, 
                  s=120, c=color, alpha=0.7, 
                  edgecolors='black', linewidth=1, zorder=3)
        
        # Ligne horizontale depuis le début jusqu'au point
        ax.hlines(y, df["first_commit_date"].min(), row["first_commit_date"],
                 colors=color, alpha=0.2, linewidth=2, zorder=1)
    
    # Configuration des axes
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_ylim(-1, y_pos)
    
    # Axe X
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha='right')
    
    # Grille
    ax.grid(True, axis='x', linestyle='--', alpha=0.3, zorder=0)
    ax.axvline(df["first_commit_date"].min(), color='green', 
              linestyle='--', alpha=0.5, linewidth=1, label='Début du projet')
    
    # Labels et titre
    ax.set_xlabel("Date de première apparition", fontsize=12, fontweight='bold')
    ax.set_title("Chronologie d'introduction - The Shift Data Portal",
                fontsize=14, fontweight='bold', pad=20)
    
    # Légende
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', 
               markersize=10, label='Sources', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', 
               markersize=10, label='Modes d\'accès', markeredgecolor='black')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[✓] Timeline Gantt sauvegardée: {output_path}")
    plt.close()


def plot_timeline_separate(df, output_path):
    """Deux graphiques séparés pour sources et modes."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    df_sources = df[df["category"] == "Source"].sort_values("first_commit_date")
    df_modes = df[df["category"] == "Mode d'accès"].sort_values("first_commit_date")
    
    # Graphique 1: Sources
    if not df_sources.empty:
        y_pos = range(len(df_sources))
        ax1.scatter(df_sources["first_commit_date"], y_pos, 
                   s=150, c='#3498db', alpha=0.7, 
                   edgecolors='black', linewidth=1.5, zorder=3)
        
        for i, (idx, row) in enumerate(df_sources.iterrows()):
            ax1.hlines(i, df_sources["first_commit_date"].min(), 
                      row["first_commit_date"],
                      colors='#3498db', alpha=0.2, linewidth=2, zorder=1)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(df_sources["label"], fontsize=9)
        ax1.set_title("Sources de données", fontsize=13, fontweight='bold', pad=10)
        ax1.grid(True, axis='x', linestyle='--', alpha=0.3)
        ax1.set_ylabel("")
    
    # Graphique 2: Modes d'accès
    if not df_modes.empty:
        y_pos = range(len(df_modes))
        ax2.scatter(df_modes["first_commit_date"], y_pos, 
                   s=150, c='#e74c3c', alpha=0.7, 
                   edgecolors='black', linewidth=1.5, zorder=3)
        
        for i, (idx, row) in enumerate(df_modes.iterrows()):
            ax2.hlines(i, df_modes["first_commit_date"].min(), 
                      row["first_commit_date"],
                      colors='#e74c3c', alpha=0.2, linewidth=2, zorder=1)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(df_modes["label"], fontsize=9)
        ax2.set_title("Modes d'accès", fontsize=13, fontweight='bold', pad=10)
        ax2.grid(True, axis='x', linestyle='--', alpha=0.3)
        ax2.set_ylabel("")
    
    # Configuration commune de l'axe X
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha='right')
    ax2.set_xlabel("Date de première apparition", fontsize=11, fontweight='bold')
    
    plt.suptitle("Chronologie d'introduction - The Shift Data Portal",
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[✓] Timeline séparée sauvegardée: {output_path}")
    plt.close()


def plot_timeline_histogram(df, output_path):
    """Histogramme du nombre d'introductions par mois."""
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Grouper par mois et catégorie
    df['month'] = df['first_commit_date'].dt.to_period('M').dt.to_timestamp()
    
    monthly_sources = df[df["category"] == "Source"].groupby('month').size()
    monthly_modes = df[df["category"] == "Mode d'accès"].groupby('month').size()
    
    # Graphique 1: Sources
    if not monthly_sources.empty:
        axes[0].bar(monthly_sources.index, monthly_sources.values, 
                   width=20, color='#3498db', alpha=0.7, 
                   edgecolor='#2980b9', linewidth=1.5)
        axes[0].set_ylabel("Nombre de sources", fontsize=11)
        axes[0].set_title("Nouvelles sources par mois", 
                         fontsize=13, fontweight='bold', pad=10)
        axes[0].grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Graphique 2: Modes
    if not monthly_modes.empty:
        axes[1].bar(monthly_modes.index, monthly_modes.values, 
                   width=20, color='#e74c3c', alpha=0.7,
                   edgecolor='#c0392b', linewidth=1.5)
        axes[1].set_ylabel("Nombre de modes", fontsize=11)
        axes[1].set_title("Nouveaux modes d'accès par mois", 
                         fontsize=13, fontweight='bold', pad=10)
        axes[1].grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Configuration de l'axe X
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha='right')
    axes[1].set_xlabel("Mois", fontsize=11, fontweight='bold')
    
    plt.suptitle("Distribution temporelle des introductions - The Shift Data Portal",
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[✓] Histogramme sauvegardé: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("GÉNÉRATION DES VISUALISATIONS DE CHRONOLOGIE")
    print("="*70 + "\n")
    
    # Chargement des données
    df = load_and_prepare(
        "results/first_seen_sources.csv",
        "results/first_seen_modes.csv"
    )
    
    print(f"[+] Chargé {len(df)} éléments")
    print(f"    - Sources: {len(df[df['category'] == 'Source'])}")
    print(f"    - Modes: {len(df[df['category'] == 'Mode d accès'])}")
    print(f"    - Période: {df['first_commit_date'].min().date()} → {df['first_commit_date'].max().date()}\n")
    
    # Génération des 3 visualisations
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    plot_timeline_gantt_style(df, results_dir / "first_seen_timeline_gantt.png")
    plot_timeline_separate(df, results_dir / "first_seen_timeline_separate.png")
    plot_timeline_histogram(df, results_dir / "first_seen_timeline_histogram.png")
    
    print("\n✅ Toutes les visualisations ont été générées avec succès!")
    print("="*70 + "\n")