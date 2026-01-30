#!/usr/bin/env python3
"""
Generate French versions of hypothesis validation plots - EXACT translations of English plots.
Simply translates labels without r=, p= annotations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import ast

# Configure plotting  
try:
    plt.style.use('seaborn-whitegrid')
except OSError:
    plt.style.use('ggplot')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 10

def parse_dict(s):
    try:
        return ast.literal_eval(s) if isinstance(s, str) else (s if isinstance(s, dict) else {})
    except:
        return {}

def load_data():
    """Load chart evolution data - SAME as English version."""
    csv_path = 'mined_data/all_metrics.csv'
    df = pd.read_csv(csv_path)
    
    df['commit_datetime'] = pd.to_datetime(df['commit_date'], utc=True)
    df = df.sort_values(['chart_root', 'commit_datetime'])
    
    # Final state for cross-sectional analysis - group by chart_root like original
    final_df = df.groupby('chart_root').last().reset_index()
    
    # Create size categories
    final_df['size_category'] = pd.cut(
        final_df['template_lines'],
        bins=[0, 200, 500, 1000, float('inf')],
        labels=['Small (<200)', 'Medium (200-500)', 'Large (500-1000)', 'XLarge (>1000)']
    )
    
    # Calculate reuse ratio
    final_df['reuse_ratio'] = final_df['includes'] / final_df['template_files'].replace(0, np.nan)
    
    print(f"Chargé: {len(df)} commits, {final_df['chart_root'].nunique()} charts")
    
    return df, final_df


def plot_h1_french(final_df):
    """H1: Croissance des Charts → Modularisation - EXACT translation of English plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Filter outliers
    df_filtered = final_df[final_df['template_lines'] <= 3000].copy()
    df_filtered = df_filtered[df_filtered['template_lines'] > 0]
    
    # 1. Adoption de _helpers.tpl par catégorie de taille
    ax = axes[0, 0]
    size_order = ['Medium (200-500)', 'Large (500-1000)', 'XLarge (>1000)']
    size_labels_fr = ['Moyen (200-500)', 'Grand (500-1000)', 'Très Grand (>1000)']
    helper_rates = []
    for cat in size_order:
        subset = final_df[final_df['size_category'] == cat]
        if len(subset) > 0:
            helper_rates.append(subset['has_helpers'].mean() * 100)
        else:
            helper_rates.append(0)
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(size_order)))
    bars = ax.bar(range(len(size_order)), helper_rates, color=colors, edgecolor='black')
    ax.set_xticks(range(len(size_order)))
    ax.set_xticklabels(size_labels_fr, fontsize=9)
    ax.set_ylabel('% Utilisant _helpers.tpl', fontsize=11)
    ax.set_title('Adoption de Helper par Taille', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    for bar, rate in zip(bars, helper_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{rate:.0f}%', ha='center', fontsize=10, fontweight='bold')
    
    # 2. Scatter: LOC vs Définitions de Templates
    ax = axes[0, 1]
    ax.scatter(df_filtered['template_lines'], df_filtered['template_definitions'], 
               alpha=0.6, s=50, c='steelblue', edgecolors='black', linewidth=0.5)
    z = np.polyfit(df_filtered['template_lines'], df_filtered['template_definitions'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_filtered['template_lines'].min(), df_filtered['template_lines'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Tendance')
    ax.set_xlabel('Lignes de Code', fontsize=11)
    ax.set_ylabel('# Définitions de Templates (define)', fontsize=11)
    ax.set_title('Taille vs Modularisation (defines)', fontsize=12, fontweight='bold')
    
    # 3. Scatter: LOC vs Includes
    ax = axes[1, 0]
    ax.scatter(df_filtered['template_lines'], df_filtered['includes'],
               alpha=0.6, s=50, c='darkorange', edgecolors='black', linewidth=0.5)
    z = np.polyfit(df_filtered['template_lines'], df_filtered['includes'], 1)
    p = np.poly1d(z)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Tendance')
    ax.set_xlabel('Lignes de Code', fontsize=11)
    ax.set_ylabel('# Instructions Include', fontsize=11)
    ax.set_title('Taille vs Réutilisation (includes)', fontsize=12, fontweight='bold')
    
    # 4. Ratio de réutilisation par taille
    ax = axes[1, 1]
    df_for_reuse = final_df.copy()
    df_for_reuse['merged_size'] = df_for_reuse['size_category'].replace({
        'Small (<200)': 'Petit+Moyen',
        'Medium (200-500)': 'Petit+Moyen'
    })
    merged_order = ['Petit+Moyen', 'Large (500-1000)', 'XLarge (>1000)']
    merged_labels_fr = ['Petit+Moyen', 'Grand (500-1000)', 'Très Grand (>1000)']
    reuse_by_size = df_for_reuse.groupby('merged_size', observed=True)['reuse_ratio'].mean()
    reuse_vals = [reuse_by_size.get(cat, 0) for cat in merged_order]
    colors_merged = plt.cm.Blues(np.linspace(0.3, 0.9, len(merged_order)))
    bars = ax.bar(range(len(merged_order)), reuse_vals, color=colors_merged, edgecolor='black')
    ax.set_xticks(range(len(merged_order)))
    ax.set_xticklabels(merged_labels_fr, fontsize=9)
    ax.set_ylabel('Ratio de Réutilisation (includes/fichiers)', fontsize=11)
    ax.set_title('Ratio de Réutilisation par Taille', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, reuse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
    
    fig.suptitle("H1: Croissance des Charts → Modularisation [CONFIRMÉE ✓]", 
                 fontsize=14, fontweight='bold', color='#2ecc71', y=1.02)
    
    plt.tight_layout()
    Path('final_results/french_graphs').mkdir(parents=True, exist_ok=True)
    plt.savefig('final_results/french_graphs/h1_modularisation.png', dpi=150, bbox_inches='tight')
    print("Saved: final_results/french_graphs/h1_modularisation.png")
    plt.close()


def plot_h2_french(final_df):
    """H2: Imbrication Values → Complexité Contrôle - EXACT translation of English plot."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Calculate control density
    final_df = final_df.copy()
    final_df['control_density'] = final_df['control_structures'] / final_df['template_lines'].replace(0, np.nan)
    
    # 1. Scatter: Nesting vs Control Structures
    ax = axes[0]
    ax.scatter(final_df['values_nesting'], final_df['control_structures'],
               alpha=0.6, s=60, c='purple', edgecolors='black', linewidth=0.5)
    z = np.polyfit(final_df['values_nesting'], final_df['control_structures'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(final_df['values_nesting'].min(), final_df['values_nesting'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2)
    ax.set_xlabel('Profondeur d\'Imbrication des Values', fontsize=11)
    ax.set_ylabel('# Structures de Contrôle (if/with/range)', fontsize=11)
    ax.set_title('Imbrication vs Structures de Contrôle', fontsize=12, fontweight='bold')
    
    # 2. Bar: Flat vs Nested
    ax = axes[1]
    flat = final_df[final_df['values_nesting'] <= 2]['control_structures']
    nested = final_df[final_df['values_nesting'] > 2]['control_structures']
    flat_mean = flat.mean()
    nested_mean = nested.mean()
    bars = ax.bar(['Valeurs Plates\n(profondeur ≤2)', 'Valeurs Imbriquées\n(profondeur >2)'], 
                  [flat_mean, nested_mean],
                  color=['#2ecc71', '#e74c3c'], edgecolor='black', width=0.6)
    ax.set_ylabel('Nombre Moyen de Structures de Contrôle', fontsize=11)
    ax.set_title('Structures de Contrôle: Plat vs Imbriqué', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, [flat_mean, nested_mean]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}', ha='center', fontsize=12, fontweight='bold')
    
    # 3. Boxplot by nesting category
    ax = axes[2]
    final_df['nesting_cat'] = pd.cut(final_df['values_nesting'], 
                                      bins=[0, 2, 3, 4, float('inf')],
                                      labels=['1-2', '3', '4', '5+'])
    data_by_cat = [final_df[final_df['nesting_cat'] == cat]['control_structures'].dropna() 
                   for cat in ['1-2', '3', '4', '5+']]
    data_by_cat = [d for d in data_by_cat if len(d) > 0]
    labels = [cat for cat, d in zip(['1-2', '3', '4', '5+'], 
              [final_df[final_df['nesting_cat'] == c] for c in ['1-2', '3', '4', '5+']]) if len(d) > 0]
    
    bp = ax.boxplot(data_by_cat, labels=labels, patch_artist=True)
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(data_by_cat)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_xlabel('Catégorie de Profondeur d\'Imbrication', fontsize=11)
    ax.set_ylabel('Structures de Contrôle', fontsize=11)
    ax.set_title('Distribution par Niveau d\'Imbrication', fontsize=12, fontweight='bold')
    
    fig.suptitle("H2: Imbrication Plus Profonde → Plus de Complexité [CONFIRMÉE ✓]",
                 fontsize=14, fontweight='bold', color='#2ecc71', y=1.02)
    
    plt.tight_layout()
    plt.savefig('final_results/french_graphs/h2_imbrication_complexite.png', dpi=150, bbox_inches='tight')
    print("Saved: final_results/french_graphs/h2_imbrication_complexite.png")
    plt.close()


def plot_h3_french(final_df):
    """H3: Charts Plus Grands → Meilleure Structure - EXACT translation of English plot."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    df = final_df.copy()
    df['doc_dict'] = df['documentation'].apply(parse_dict)
    df['struct_dict'] = df['templates_structure'].apply(parse_dict)
    
    df['has_readme'] = df['doc_dict'].apply(lambda x: x.get('has_readme', False))
    df['has_notes'] = df['doc_dict'].apply(lambda x: x.get('has_notes', False))
    df['has_schema'] = df['doc_dict'].apply(lambda x: x.get('has_schema', False))
    df['has_helpers_extracted'] = df['struct_dict'].apply(lambda x: x.get('has_helpers', False))
    
    # Size categories with French labels
    df['size_cat'] = pd.cut(
        df['template_lines'],
        bins=[0, 200, 500, float('inf')],
        labels=['Petit (<200)', 'Moyen (200-500)', 'Grand (>500)']
    )
    
    # 1. Heatmap d'adoption des bonnes pratiques
    ax = axes[0]
    practices = ['has_readme', 'has_notes', 'has_schema', 'has_helpers']
    practice_labels = ['README', 'NOTES.txt', 'values.schema', '_helpers.tpl']
    size_cats = ['Petit (<200)', 'Moyen (200-500)', 'Grand (>500)']
    
    heatmap_data = []
    for practice in practices:
        row = []
        for cat in size_cats:
            subset = df[df['size_cat'] == cat]
            if len(subset) > 0:
                rate = subset[practice].sum() / len(subset) * 100
            else:
                rate = 0
            row.append(rate)
        heatmap_data.append(row)
    
    heatmap_array = np.array(heatmap_data)
    im = ax.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax.grid(False)
    
    for i in range(len(practices)):
        for j in range(len(size_cats)):
            val = heatmap_array[i, j]
            color = 'white' if val < 30 or val > 70 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center', 
                   fontsize=11, fontweight='bold', color=color)
    
    ax.set_xticks(range(len(size_cats)))
    ax.set_xticklabels(size_cats, fontsize=10)
    ax.set_yticks(range(len(practices)))
    ax.set_yticklabels(practice_labels, fontsize=10)
    ax.set_title('Taux d\'Adoption des Bonnes Pratiques', fontsize=12, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Adoption %', fontsize=10)
    
    # 2. Distribution des scores par taille (Box Plot)
    ax = axes[1]
    
    box_data = []
    positions = []
    colors_box = []
    color_map = {'Petit (<200)': '#3498db', 'Moyen (200-500)': '#f39c12', 'Grand (>500)': '#27ae60'}
    
    for idx, cat in enumerate(size_cats):
        subset = df[df['size_cat'] == cat]
        if len(subset) > 0:
            box_data.append(subset['overall_score'].dropna().values)
            positions.append(idx)
            colors_box.append(color_map[cat])
    
    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add individual points
    for idx, (data, pos) in enumerate(zip(box_data, positions)):
        jitter = np.random.normal(0, 0.08, len(data))
        ax.scatter([pos + j for j in jitter], data, alpha=0.5, s=30, 
                  c=colors_box[idx], edgecolors='black', linewidth=0.5, zorder=3)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(size_cats, fontsize=10)
    ax.set_ylabel('Score de Conformité Global', fontsize=11)
    ax.set_title('Distribution des Scores de Conformité', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Seuil 50%')
    ax.legend(loc='lower right', fontsize=9)
    
    # Add count labels
    for idx, cat in enumerate(size_cats):
        n = len(df[df['size_cat'] == cat])
        ax.text(idx, -0.08, f'n={n}', ha='center', fontsize=9, transform=ax.get_xaxis_transform())
    
    # 3. Complétude de la Documentation (Stacked Bar)
    ax = axes[2]
    
    doc_features = ['has_readme', 'has_notes', 'has_schema']
    df['doc_count'] = df[doc_features].sum(axis=1)
    
    categories = ['0 doc', '1 doc', '2 docs', '3 docs']
    stacked_data = {cat: [] for cat in categories}
    
    for size_cat in size_cats:
        subset = df[df['size_cat'] == size_cat]
        total = len(subset) if len(subset) > 0 else 1
        stacked_data['0 doc'].append(len(subset[subset['doc_count'] == 0]) / total * 100)
        stacked_data['1 doc'].append(len(subset[subset['doc_count'] == 1]) / total * 100)
        stacked_data['2 docs'].append(len(subset[subset['doc_count'] == 2]) / total * 100)
        stacked_data['3 docs'].append(len(subset[subset['doc_count'] == 3]) / total * 100)
    
    x = np.arange(len(size_cats))
    width = 0.6
    
    colors_stack = ['#e74c3c', '#f39c12', '#3498db', '#27ae60']
    bottom = np.zeros(len(size_cats))
    
    for cat, color in zip(categories, colors_stack):
        vals = stacked_data[cat]
        ax.bar(x, vals, width, bottom=bottom, label=cat, color=color, edgecolor='black', linewidth=0.5)
        bottom += vals
    
    ax.set_xticks(x)
    ax.set_xticklabels(size_cats, fontsize=10)
    ax.set_ylabel('Pourcentage de Charts', fontsize=11)
    ax.set_title('Complétude de la Documentation', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.set_ylim(0, 105)
    
    # Add percentage labels
    bottom = np.zeros(len(size_cats))
    for cat, color in zip(categories, colors_stack):
        vals = stacked_data[cat]
        for i, v in enumerate(vals):
            if v >= 15:
                ax.text(i, bottom[i] + v/2, f'{v:.0f}%', ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white')
        bottom += vals
    
    fig.suptitle("H3: Charts Plus Grands → Meilleure Structure [NON CONFIRMÉE ✗]",
                 fontsize=14, fontweight='bold', color='#e74c3c', y=1.02)
    
    plt.tight_layout()
    plt.savefig('final_results/french_graphs/h3_organisation.png', dpi=150, bbox_inches='tight')
    print("Saved: final_results/french_graphs/h3_organisation.png")
    plt.close()


def plot_h4_french(df, final_df):
    """H4: Charts Plus Matures → Meilleure Conformité - EXACT translation of English plot."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Calculate commit counts per chart - same as English version
    commit_counts = df.groupby('chart_name').size().reset_index(name='commit_count')
    analysis_df = final_df.merge(commit_counts, on='chart_name', how='left')
    analysis_df['commit_count'] = analysis_df['commit_count'].fillna(1)
    
    median_commits = analysis_df['commit_count'].median()
    
    # 1. Scatter: Commits vs Score
    ax = axes[0]
    colors = ['#e74c3c' if c <= median_commits else '#27ae60' 
              for c in analysis_df['commit_count']]
    
    ax.scatter(analysis_df['commit_count'], analysis_df['overall_score'],
               c=colors, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    z = np.polyfit(analysis_df['commit_count'], analysis_df['overall_score'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(analysis_df['commit_count'].min(), analysis_df['commit_count'].max(), 100)
    ax.plot(x_line, p(x_line), 'b--', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Nombre de Commits (Maturité)', fontsize=11)
    ax.set_ylabel('Score de Conformité Global', fontsize=11)
    ax.set_title('Maturité vs Conformité', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label=f'Récent (≤{median_commits:.0f})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#27ae60', markersize=10, label=f'Mature (>{median_commits:.0f})')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # 2. Box Plot: Mature vs Newer
    ax = axes[1]
    mature = analysis_df[analysis_df['commit_count'] > median_commits]['overall_score']
    newer = analysis_df[analysis_df['commit_count'] <= median_commits]['overall_score']
    
    bp = ax.boxplot([newer.values, mature.values], 
                    labels=[f'Récent\n(≤{median_commits:.0f} commits)', f'Mature\n(>{median_commits:.0f} commits)'],
                    patch_artist=True, widths=0.6)
    
    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#27ae60')
    bp['boxes'][1].set_alpha(0.7)
    
    # Add individual points
    for i, (data, pos) in enumerate([(newer.values, 1), (mature.values, 2)]):
        jitter = np.random.normal(0, 0.08, len(data))
        color = '#e74c3c' if i == 0 else '#27ae60'
        ax.scatter([pos + j for j in jitter], data, alpha=0.5, s=40, 
                  c=color, edgecolors='black', linewidth=0.5, zorder=3)
    
    ax.set_ylabel('Score de Conformité Global', fontsize=11)
    ax.set_title('Conformité: Mature vs Récent', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    
    # Add mean annotations
    ax.text(1, newer.mean() + 0.05, f'μ={newer.mean():.2f}', ha='center', fontsize=10, fontweight='bold')
    ax.text(2, mature.mean() + 0.05, f'μ={mature.mean():.2f}', ha='center', fontsize=10, fontweight='bold')
    
    # Add n labels
    ax.text(1, -0.08, f'n={len(newer)}', ha='center', fontsize=9, transform=ax.get_xaxis_transform())
    ax.text(2, -0.08, f'n={len(mature)}', ha='center', fontsize=9, transform=ax.get_xaxis_transform())
    
    # 3. Best Practices Adoption Comparison
    ax = axes[2]
    
    analysis_df['doc_dict'] = analysis_df['documentation'].apply(parse_dict)
    analysis_df['has_readme'] = analysis_df['doc_dict'].apply(lambda x: x.get('has_readme', False))
    analysis_df['has_notes'] = analysis_df['doc_dict'].apply(lambda x: x.get('has_notes', False))
    
    mature_df = analysis_df[analysis_df['commit_count'] > median_commits]
    newer_df = analysis_df[analysis_df['commit_count'] <= median_commits]
    
    practices = ['has_helpers', 'has_readme', 'has_notes']
    practice_labels = ['_helpers.tpl', 'README', 'NOTES.txt']
    
    newer_rates = [newer_df[p].mean() * 100 for p in practices]
    mature_rates = [mature_df[p].mean() * 100 for p in practices]
    
    x = np.arange(len(practices))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, newer_rates, width, label=f'Récent (≤{median_commits:.0f})', 
                   color='#e74c3c', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, mature_rates, width, label=f'Mature (>{median_commits:.0f})', 
                   color='#27ae60', edgecolor='black', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(practice_labels, fontsize=10)
    ax.set_ylabel('Taux d\'Adoption (%)', fontsize=11)
    ax.set_title('Bonnes Pratiques par Maturité', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 105)
    
    for bar, val in zip(bars1, newer_rates):
        if val > 5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.0f}%', ha='center', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, mature_rates):
        if val > 5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.0f}%', ha='center', fontsize=9, fontweight='bold')
    
    fig.suptitle("H4: Charts Plus Matures → Meilleure Conformité [NON CONFIRMÉE ✗]",
                 fontsize=14, fontweight='bold', color='#e74c3c', y=1.02)
    
    plt.tight_layout()
    plt.savefig('final_results/french_graphs/h4_maturite.png', dpi=150, bbox_inches='tight')
    print("Saved: final_results/french_graphs/h4_maturite.png")
    plt.close()


def plot_summary_french():
    """Résumé des hypothèses - EXACT translation of English summary."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    hypotheses = ['H1', 'H2', 'H3', 'H4']
    descriptions = [
        'Croissance des Charts → Modularisation',
        'Imbrication des Values → Complexité de Contrôle',
        'Charts Plus Grands → Meilleure Structure',
        'Charts Plus Matures → Meilleure Conformité'
    ]
    supported = [True, True, False, False]
    colors = ['#2ecc71' if s else '#e74c3c' for s in supported]
    
    y_pos = np.arange(len(hypotheses))
    bars = ax.barh(y_pos, [1]*len(hypotheses), color=colors, alpha=0.85, edgecolor='white', height=0.7)
    
    for i, (h, desc, supp) in enumerate(zip(hypotheses, descriptions, supported)):
        result = "✓ CONFIRMÉE" if supp else "✗ NON CONFIRMÉE"
        ax.text(0.02, i, f"{h}: {desc}", va='center', ha='left', fontsize=11, color='white', fontweight='bold')
        ax.text(0.98, i, result, va='center', ha='right', fontsize=11, color='white', fontweight='bold')
    
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.set_title('Résumé de la Validation des Hypothèses: 2/4 Confirmées', 
                 fontsize=14, fontweight='bold', pad=20)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='Confirmée'),
                       Patch(facecolor='#e74c3c', label='Non Confirmée')]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('final_results/french_graphs/resume.png', dpi=150, bbox_inches='tight')
    print("Saved: final_results/french_graphs/resume.png")
    plt.close()


def plot_dashboard_french(final_df):
    """Tableau de bord des bonnes pratiques - EXACT translation of English dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    df = final_df.copy()
    df['struct'] = df['templates_structure'].apply(parse_dict)
    df['labels'] = df['labels_compliance'].apply(parse_dict)
    df['naming'] = df['values_naming'].apply(parse_dict)
    df['docs'] = df['documentation'].apply(parse_dict)
    
    df['camelcase_files'] = df['struct'].apply(lambda x: x.get('camelcase_files', 0))
    df['template_files_count'] = df['struct'].apply(lambda x: x.get('template_files_count', 0))
    df['namespaced'] = df['struct'].apply(lambda x: x.get('namespaced_templates', 0))
    df['non_namespaced'] = df['struct'].apply(lambda x: x.get('non_namespaced_templates', 0))
    df['labels_coverage'] = df['labels'].apply(lambda x: x.get('labels_coverage', 0))
    df['values_violations'] = df['naming'].apply(lambda x: x.get('violations_count', 0))
    df['total_values_keys'] = df['naming'].apply(lambda x: x.get('total_keys', 0))
    df['values_has_comments'] = df['naming'].apply(lambda x: x.get('has_comments', False))
    df['has_readme'] = df['docs'].apply(lambda x: x.get('has_readme', False))
    df['has_notes'] = df['docs'].apply(lambda x: x.get('has_notes', False))
    df['has_schema'] = df['docs'].apply(lambda x: x.get('has_schema', False))
    
    # 1. Vue d'ensemble de l'adoption
    ax = axes[0, 0]
    
    total_files = df['template_files_count'].sum()
    camelcase = df['camelcase_files'].sum()
    file_naming_compliance = (total_files - camelcase) / total_files * 100 if total_files > 0 else 0
    
    total_ns = df['namespaced'].sum()
    total_non_ns = df['non_namespaced'].sum()
    namespacing_compliance = total_ns / (total_ns + total_non_ns) * 100 if (total_ns + total_non_ns) > 0 else 100
    
    practices = ['Nommage fichiers\n(tirets)', 'Définitions\nnamespace', '_helpers.tpl', 
                 'README.md', 'NOTES.txt', 'Commentaires\nvalues.yaml']
    compliance_rates = [file_naming_compliance, namespacing_compliance, df['has_helpers'].mean() * 100,
                       df['has_readme'].mean() * 100, df['has_notes'].mean() * 100, df['values_has_comments'].mean() * 100]
    
    colors_bar = ['#27ae60' if r >= 70 else '#f39c12' if r >= 40 else '#e74c3c' for r in compliance_rates]
    
    bars = ax.barh(practices, compliance_rates, color=colors_bar, edgecolor='black', height=0.6)
    ax.set_xlim(0, 105)
    ax.set_xlabel('Taux de Conformité (%)', fontsize=11)
    ax.set_title('Vue d\'Ensemble de l\'Adoption', fontsize=12, fontweight='bold')
    ax.axvline(x=70, color='green', linestyle='--', alpha=0.5, label='Bon (70%)')
    ax.axvline(x=40, color='orange', linestyle='--', alpha=0.5, label='Modéré (40%)')
    
    for bar, val in zip(bars, compliance_rates):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.0f}%', 
               va='center', fontsize=10, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    
    # 2. Distribution de la conformité des labels
    ax = axes[0, 1]
    ax.hist(df['labels_coverage'] * 100, bins=10, color='#3498db', edgecolor='black', alpha=0.7)
    ax.axvline(x=df['labels_coverage'].mean() * 100, color='red', linestyle='--', linewidth=2,
              label=f'Moyenne: {df["labels_coverage"].mean()*100:.1f}%')
    ax.set_xlabel('Couverture des Labels Standards (%)', fontsize=11)
    ax.set_ylabel('Nombre de Charts', fontsize=11)
    ax.set_title('Conformité des Labels Kubernetes', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    
    # 3. Qualité de values.yaml
    ax = axes[0, 2]
    df['violation_rate'] = df.apply(
        lambda x: x['values_violations'] / x['total_values_keys'] * 100 if x['total_values_keys'] > 0 else 0, 
        axis=1
    )
    
    low_violations = len(df[df['violation_rate'] == 0])
    some_violations = len(df[(df['violation_rate'] > 0) & (df['violation_rate'] <= 5)])
    high_violations = len(df[df['violation_rate'] > 5])
    
    sizes = [low_violations, some_violations, high_violations]
    labels_pie = [f'Aucune violation\n({low_violations})', f'≤5% violations\n({some_violations})', f'>5% violations\n({high_violations})']
    colors_pie = ['#27ae60', '#f39c12', '#e74c3c']
    
    ax.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.0f%%',
           startangle=90, textprops={'fontsize': 9})
    ax.set_title('Convention de Nommage Values\n(conformité camelCase)', fontsize=12, fontweight='bold')
    
    # 4. Documentation par dépôt
    ax = axes[1, 0]
    repo_docs = df.groupby('repo').agg({
        'has_readme': 'mean',
        'has_notes': 'mean', 
        'has_schema': 'mean',
        'chart_name': 'count'
    }).reset_index()
    repo_docs.columns = ['repo', 'readme', 'notes', 'schema', 'count']
    repo_docs = repo_docs[repo_docs['count'] >= 2].sort_values('count', ascending=True)
    
    y_pos = np.arange(len(repo_docs))
    width = 0.25
    
    ax.barh(y_pos - width, repo_docs['readme'] * 100, width, label='README', color='#3498db', alpha=0.8)
    ax.barh(y_pos, repo_docs['notes'] * 100, width, label='NOTES.txt', color='#27ae60', alpha=0.8)
    ax.barh(y_pos + width, repo_docs['schema'] * 100, width, label='values.schema', color='#9b59b6', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(repo_docs['repo'], fontsize=9)
    ax.set_xlabel('Taux d\'Adoption (%)', fontsize=11)
    ax.set_title('Documentation par Dépôt', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(0, 105)
    
    # 5. Complexité vs Bonnes pratiques
    ax = axes[1, 1]
    df_plot = df[df['template_lines'] <= 3000]
    colors_scatter = ['#27ae60' if h else '#e74c3c' for h in df_plot['has_helpers']]
    
    ax.scatter(df_plot['template_lines'], df_plot['overall_score'], 
              c=colors_scatter, s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Lignes de Code', fontsize=11)
    ax.set_ylabel('Score de Conformité Global', fontsize=11)
    ax.set_title('Complexité vs Conformité\n(couleur: _helpers.tpl)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#27ae60', markersize=10, label='Avec _helpers.tpl'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label='Sans _helpers.tpl')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # 6. Décomposition des scores
    ax = axes[1, 2]
    components = ['templates_structure_score', 'labels_compliance_score', 
                  'values_naming_score', 'documentation_score', 'overall_score']
    component_labels = ['Structure\nTemplates', 'Conformité\nLabels', 'Nommage\nValues', 
                       'Documentation', 'Global']
    avg_scores = [df[c].mean() for c in components]
    
    colors_comp = plt.cm.viridis(np.linspace(0.2, 0.8, len(components)))
    
    bars = ax.bar(component_labels, avg_scores, color=colors_comp, edgecolor='black', width=0.6)
    ax.set_ylabel('Score Moyen', fontsize=11)
    ax.set_title('Décomposition des Scores', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Seuil 50%')
    
    for bar, val in zip(bars, avg_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    
    fig.suptitle('Tableau de Bord des Bonnes Pratiques Helm\n(Basé sur helm.sh/docs/chart_best_practices/)',
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('final_results/french_graphs/tableau_de_bord.png', dpi=150, bbox_inches='tight')
    print("Saved: final_results/french_graphs/tableau_de_bord.png")
    plt.close()


def main():
    print("=" * 60)
    print("Génération des graphiques en français (EXACT copy of English)")
    print("=" * 60)
    
    df, final_df = load_data()
    
    print(f"\nChargé: {len(df)} commits, {len(final_df)} charts")
    
    # Create output directory
    Path('final_results/french_graphs').mkdir(parents=True, exist_ok=True)
    
    plot_h1_french(final_df.copy())
    plot_h2_french(final_df.copy())
    plot_h3_french(final_df.copy())
    plot_h4_french(df, final_df.copy())
    plot_summary_french()
    plot_dashboard_french(final_df.copy())
    
    print("\n✅ Tous les graphiques français générés dans final_results/french_graphs/")


if __name__ == '__main__':
    main()
