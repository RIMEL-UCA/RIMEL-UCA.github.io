import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Configuration
INPUT_CSV = Path("../results/enrichment_history.csv")
OUTPUT_DIR = Path("results")
START_DATE = "2024-01-01"

def plot_activity_heatmap():
    """Crée un heatmap de l'activité de développement."""
    
    if not INPUT_CSV.exists():
        print(f" Error: {INPUT_CSV} not found.")
        return
    
    df = pd.read_csv(INPUT_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= pd.to_datetime(START_DATE)]
    
    print(f"[+] Loaded {len(df)} changes from {INPUT_CSV}")
    
    # Agrégation par semaine et type de changement
    df['week'] = df['date'].dt.to_period('W').dt.to_timestamp()
    
    weekly = df.groupby(['week', 'change_type']).size().unstack(fill_value=0)
    
    # Graphique 1: Activity par type de changement
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # === Graphique supérieur: Aires empilées par type ===
    colors = {
        'ADD': '#2ecc71',      # Vert
        'MODIFY': '#3498db',   # Bleu
        'DELETE': '#e74c3c',   # Rouge
        'RENAME': '#f39c12'    # Orange
    }
    
    available_types = [t for t in ['ADD', 'MODIFY', 'DELETE', 'RENAME'] if t in weekly.columns]
    
    axes[0].stackplot(
        weekly.index,
        [weekly[t] for t in available_types],
        labels=available_types,
        colors=[colors[t] for t in available_types],
        alpha=0.8
    )
    
    axes[0].set_title('Activité hebdomadaire par type de changement - Data Preparation Scripts', 
                      fontsize=14, fontweight='bold', pad=15)
    axes[0].set_ylabel('Nombre de changements', fontsize=11)
    axes[0].legend(loc='upper left', fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.3)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[0].tick_params(axis='x', rotation=45)
    
    # === Graphique inférieur: Total d'activité ===
    total_weekly = df.groupby('week').size()
    
    axes[1].bar(total_weekly.index, total_weekly.values, 
                width=5, color='#9b59b6', alpha=0.7, edgecolor='#8e44ad', linewidth=1.5)
    
    # Ligne de tendance
    z = np.polyfit(range(len(total_weekly)), total_weekly.values, 3)
    p = np.poly1d(z)
    axes[1].plot(total_weekly.index, p(range(len(total_weekly))), 
                 "r--", alpha=0.8, linewidth=2, label='Tendance')
    
    axes[1].set_title('Volume total d\'activité hebdomadaire', 
                      fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlabel('Date', fontsize=11)
    axes[1].set_ylabel('Nombre total de changements', fontsize=11)
    axes[1].legend(loc='upper left', fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.3, axis='y')
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "data_preparation_activity_weekly.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"[✓] Weekly activity plot saved to: {output_file}")
    plt.close()


def plot_lines_changed():
    """Visualise les lignes ajoutées et supprimées."""
    
    df = pd.read_csv(INPUT_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= pd.to_datetime(START_DATE)]
    
    # Agrégation par semaine
    df['week'] = df['date'].dt.to_period('W').dt.to_timestamp()
    
    weekly_lines = df.groupby('week').agg({
        'lines_added': 'sum',
        'lines_removed': 'sum'
    })
    
    # Calcul du churn
    weekly_lines['churn'] = weekly_lines['lines_added'] + weekly_lines['lines_removed']
    weekly_lines['net'] = weekly_lines['lines_added'] - weekly_lines['lines_removed']
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # === Graphique 1: Lignes ajoutées vs supprimées ===
    axes[0].bar(weekly_lines.index, weekly_lines['lines_added'], 
                width=5, label='Lignes ajoutées', color='#2ecc71', alpha=0.7)
    axes[0].bar(weekly_lines.index, -weekly_lines['lines_removed'], 
                width=5, label='Lignes supprimées', color='#e74c3c', alpha=0.7)
    
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[0].set_title('Lignes de code ajoutées et supprimées (hebdomadaire)', 
                      fontsize=14, fontweight='bold', pad=15)
    axes[0].set_ylabel('Nombre de lignes', fontsize=11)
    axes[0].legend(loc='upper left', fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.3, axis='y')
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[0].tick_params(axis='x', rotation=45)
    
    # === Graphique 2: Code churn ===
    axes[1].fill_between(weekly_lines.index, weekly_lines['churn'], 
                         color='#9b59b6', alpha=0.3, label='Total churn')
    axes[1].plot(weekly_lines.index, weekly_lines['churn'], 
                 color='#8e44ad', linewidth=2, marker='o', markersize=4)
    
    axes[1].set_title('Code Churn (lignes ajoutées + supprimées)', 
                      fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlabel('Date', fontsize=11)
    axes[1].set_ylabel('Code churn', fontsize=11)
    axes[1].legend(loc='upper left', fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "data_preparation_code_churn.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"[✓] Code churn plot saved to: {output_file}")
    plt.close()


def plot_file_activity():
    """Montre quels fichiers sont les plus actifs."""
    
    df = pd.read_csv(INPUT_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= pd.to_datetime(START_DATE)]
    
    # Top 15 fichiers les plus modifiés
    top_files = df['file'].value_counts().head(15)
    
    # Raccourcir les noms de fichiers
    short_names = [Path(f).name for f in top_files.index]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(range(len(top_files)), top_files.values, color='#3498db', alpha=0.8)
    
    # Gradient de couleur
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.Blues(0.4 + 0.6 * (i / len(bars))))
    
    ax.set_yticks(range(len(top_files)))
    ax.set_yticklabels(short_names, fontsize=9)
    ax.invert_yaxis()
    
    ax.set_xlabel('Nombre de modifications', fontsize=11)
    ax.set_title('Top 15 des fichiers les plus modifiés', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for i, v in enumerate(top_files.values):
        ax.text(v + 0.3, i, str(v), va='center', fontsize=9)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "data_preparation_top_files.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"[✓] Top files plot saved to: {output_file}")
    plt.close()


def print_enhanced_statistics():
    """Affiche des statistiques détaillées."""
    
    df = pd.read_csv(INPUT_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= pd.to_datetime(START_DATE)]
    
    print("\n" + "="*70)
    print("STATISTIQUES D'ACTIVITÉ")
    print("="*70)
    
    print(f"\n Période analysée: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f" Total de changements: {len(df)}")
    
    # Répartition par type
    print("\n Répartition par type de changement:")
    for change_type, count in df['change_type'].value_counts().items():
        pct = 100 * count / len(df)
        print(f"   {change_type:10s}: {count:4d} ({pct:5.1f}%)")
    
    # Lignes de code
    total_added = df['lines_added'].sum()
    total_removed = df['lines_removed'].sum()
    print(f"\n Lignes de code:")
    print(f"   Ajoutées:    {total_added:6d}")
    print(f"   Supprimées:  {total_removed:6d}")
    print(f"   Net:         {total_added - total_removed:+6d}")
    print(f"   Churn total: {total_added + total_removed:6d}")
    
    # Contributeurs
    print(f"\n Contributeurs: {df['author'].nunique()}")
    print("   Top 5:")
    for author, count in df['author'].value_counts().head(5).items():
        pct = 100 * count / len(df)
        print(f"   {author:20s}: {count:4d} ({pct:5.1f}%)")
    
    # Fichiers uniques
    print(f"\n Fichiers touchés: {df['file'].nunique()}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    import numpy as np
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("GÉNÉRATION DES VISUALISATIONS D'ACTIVITÉ")
    print("="*70 + "\n")
    
    plot_activity_heatmap()
    plot_lines_changed()
    plot_file_activity()
    print_enhanced_statistics()
    
    print("\n Toutes les visualisations ont été générées avec succès!")