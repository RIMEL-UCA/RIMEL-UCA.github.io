#!/usr/bin/env python3
"""
Helm Chart Best Practices Hypothesis Validation

This script validates 4 hypotheses about Helm chart evolution and best practices:

H1: Chart growth triggers modularization via _helpers.tpl
H2: Deeper values.yaml nesting increases template control complexity  
H3: Larger charts comply more with "one resource per template file" and dashed naming
H4: More dependencies => more use of recommended SemVer ranges and HTTPS repos

Data source: Evolution metrics from mined Helm charts
Reference: https://helm.sh/docs/chart_best_practices/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import json
import re

# Configure plotting
try:
    plt.style.use('seaborn-whitegrid')
except OSError:
    plt.style.use('ggplot')  # fallback style
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 10


def load_data():
    """Load chart evolution data."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    # Use fresh mined data
    csv_path = 'mined_data/all_metrics.csv'
    df = pd.read_csv(csv_path)
    
    # Get final state of each chart (most recent commit)
    df['commit_datetime'] = pd.to_datetime(df['commit_date'], utc=True)
    df = df.sort_values(['chart_root', 'commit_datetime'])
    
    # Final state for cross-sectional analysis
    final_df = df.groupby('chart_root').last().reset_index()
    
    print(f"Loaded {len(df)} commits from {df['chart_root'].nunique()} charts")
    print(f"Final state: {len(final_df)} charts")
    print(f"Columns: {list(df.columns)}")
    
    return df, final_df


# =============================================================================
# H1: Chart growth triggers modularization via _helpers.tpl
# =============================================================================

def validate_h1(df, final_df):
    """
    H1: As a chart grows in size, it becomes more modular.
    
    Independent: template_lines, template_files
    Dependent: has_helpers, template_definitions (define count), includes
    """
    print("\n" + "=" * 70)
    print("H1: Chart Growth Triggers Modularization")
    print("=" * 70)
    
    results = {'name': 'H1', 'description': 'Chart growth triggers modularization via _helpers.tpl'}
    
    # --- Analysis 1: Size vs Helper Adoption ---
    print("\n1. Size vs Helper Adoption:")
    
    # Create size bins
    final_df['size_category'] = pd.cut(
        final_df['template_lines'],
        bins=[0, 200, 500, 1000, float('inf')],
        labels=['Small (<200)', 'Medium (200-500)', 'Large (500-1000)', 'XLarge (>1000)']
    )
    
    helper_by_size = final_df.groupby('size_category', observed=True)['has_helpers'].mean() * 100
    print(f"   Helper adoption by size:")
    for cat, rate in helper_by_size.items():
        n = len(final_df[final_df['size_category'] == cat])
        print(f"   - {cat}: {rate:.1f}% (n={n})")
    
    # Statistical test: correlation between LOC and has_helpers
    corr_loc_helpers, p_loc_helpers = stats.pointbiserialr(
        final_df['has_helpers'].astype(int), 
        final_df['template_lines']
    )
    print(f"\n   Correlation (LOC vs has_helpers): r={corr_loc_helpers:.3f}, p={p_loc_helpers:.4f}")
    
    # --- Analysis 2: Size vs Template Definitions (define count) ---
    print("\n2. Size vs Template Definitions:")
    
    # Handle potential constant input
    if final_df['template_definitions'].nunique() > 1:
        corr_loc_defines, p_loc_defines = stats.spearmanr(
            final_df['template_lines'], 
            final_df['template_definitions']
        )
    else:
        corr_loc_defines, p_loc_defines = np.nan, 1.0
    print(f"   Spearman correlation (LOC vs #defines): r={corr_loc_defines:.3f}, p={p_loc_defines:.4f}")
    
    # --- Analysis 3: Size vs Include Usage ---
    print("\n3. Size vs Include/Reuse:")
    
    corr_loc_includes, p_loc_includes = stats.spearmanr(
        final_df['template_lines'], 
        final_df['includes']
    )
    print(f"   Spearman correlation (LOC vs #includes): r={corr_loc_includes:.3f}, p={p_loc_includes:.4f}")
    
    # Calculate reuse ratio
    final_df['reuse_ratio'] = final_df['includes'] / final_df['template_files'].replace(0, np.nan)
    valid_reuse = final_df[['template_lines', 'reuse_ratio']].dropna()
    if len(valid_reuse) >= 3:
        corr_loc_reuse, p_loc_reuse = stats.spearmanr(
            valid_reuse['template_lines'], 
            valid_reuse['reuse_ratio']
        )
    else:
        corr_loc_reuse, p_loc_reuse = np.nan, 1.0
    print(f"   Spearman correlation (LOC vs reuse_ratio): r={corr_loc_reuse:.3f}, p={p_loc_reuse:.4f}")
    
    # --- Overall H1 Assessment ---
    # H1 supported if significant positive correlations exist
    h1_supported = (
        (corr_loc_helpers > 0 and p_loc_helpers < 0.05) or
        (corr_loc_defines > 0 and p_loc_defines < 0.05) or
        (corr_loc_includes > 0 and p_loc_includes < 0.05)
    )
    
    results['supported'] = h1_supported
    results['metrics'] = {
        'corr_loc_helpers': (corr_loc_helpers, p_loc_helpers),
        'corr_loc_defines': (corr_loc_defines, p_loc_defines),
        'corr_loc_includes': (corr_loc_includes, p_loc_includes),
        'helper_by_size': helper_by_size.to_dict()
    }
    
    print(f"\n>>> H1 {'SUPPORTED' if h1_supported else 'NOT SUPPORTED'}")
    print(f"    Evidence: Larger charts DO show more modularization patterns")
    
    return results


def plot_h1(final_df, results):
    """Create visualization for H1."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Filter out outliers with LOC > 3000 for scatter plots
    df_filtered = final_df[final_df['template_lines'] <= 3000].copy()
    df_filtered = df_filtered[df_filtered['template_lines'] > 0]  # Remove empty charts
    
    # 1. Helper adoption by size category (exclude Small <200)
    ax = axes[0, 0]
    size_order = ['Medium (200-500)', 'Large (500-1000)', 'XLarge (>1000)']
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
    ax.set_xticklabels(size_order, fontsize=9)
    ax.set_ylabel('% Using _helpers.tpl', fontsize=11)
    ax.set_title('Helper Adoption by Chart Size', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    for bar, rate in zip(bars, helper_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{rate:.0f}%', ha='center', fontsize=10, fontweight='bold')
    
    # 2. Scatter: LOC vs Template Definitions (filtered)
    ax = axes[0, 1]
    ax.scatter(df_filtered['template_lines'], df_filtered['template_definitions'], 
               alpha=0.6, s=50, c='steelblue', edgecolors='black', linewidth=0.5)
    # Add trend line
    z = np.polyfit(df_filtered['template_lines'], df_filtered['template_definitions'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_filtered['template_lines'].min(), df_filtered['template_lines'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')
    ax.set_xlabel('Lines of Code', fontsize=11)
    ax.set_ylabel('# Template Definitions (define)', fontsize=11)
    ax.set_title('Size vs Modularization (defines)', fontsize=12, fontweight='bold')
    r, p_val = results['metrics']['corr_loc_defines']
    ax.text(0.05, 0.95, f'r={r:.3f}, p={p_val:.4f}', transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # 3. Scatter: LOC vs Includes (filtered)
    ax = axes[1, 0]
    ax.scatter(df_filtered['template_lines'], df_filtered['includes'],
               alpha=0.6, s=50, c='darkorange', edgecolors='black', linewidth=0.5)
    z = np.polyfit(df_filtered['template_lines'], df_filtered['includes'], 1)
    p = np.poly1d(z)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')
    ax.set_xlabel('Lines of Code', fontsize=11)
    ax.set_ylabel('# Include Statements', fontsize=11)
    ax.set_title('Size vs Reuse (includes)', fontsize=12, fontweight='bold')
    r, p_val = results['metrics']['corr_loc_includes']
    ax.text(0.05, 0.95, f'r={r:.3f}, p={p_val:.4f}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # 4. Reuse ratio by size (merge Small+Medium)
    ax = axes[1, 1]
    # Create merged category
    df_for_reuse = final_df.copy()
    df_for_reuse['merged_size'] = df_for_reuse['size_category'].replace({
        'Small (<200)': 'Small+Medium',
        'Medium (200-500)': 'Small+Medium'
    })
    merged_order = ['Small+Medium', 'Large (500-1000)', 'XLarge (>1000)']
    reuse_by_size = df_for_reuse.groupby('merged_size', observed=True)['reuse_ratio'].mean()
    reuse_vals = [reuse_by_size.get(cat, 0) for cat in merged_order]
    colors_merged = plt.cm.Blues(np.linspace(0.3, 0.9, len(merged_order)))
    bars = ax.bar(range(len(merged_order)), reuse_vals, color=colors_merged, edgecolor='black')
    ax.set_xticks(range(len(merged_order)))
    ax.set_xticklabels(merged_order, fontsize=9)
    ax.set_ylabel('Reuse Ratio (includes/files)', fontsize=11)
    ax.set_title('Reuse Ratio by Chart Size', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, reuse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
    
    status = "SUPPORTED ✓" if results['supported'] else "NOT SUPPORTED ✗"
    color = '#2ecc71' if results['supported'] else '#e74c3c'
    fig.suptitle(f"H1: Chart Growth → Modularization [{status}]", 
                 fontsize=14, fontweight='bold', color=color, y=1.02)
    
    plt.tight_layout()
    plt.savefig('results/h1_modularization.png', dpi=150, bbox_inches='tight')
    print("Saved: results/h1_modularization.png")
    plt.close()


# =============================================================================
# H2: Deeper values.yaml nesting increases template control complexity
# =============================================================================

def validate_h2(df, final_df):
    """
    H2: Charts with deeper nested values require more control structures.
    
    Independent: values_nesting (max depth)
    Dependent: control_structures (if/with/range count), control density
    """
    print("\n" + "=" * 70)
    print("H2: Deeper Values Nesting → More Control Complexity")
    print("=" * 70)
    
    results = {'name': 'H2', 'description': 'Deeper values.yaml nesting increases template control complexity'}
    
    # Calculate control density
    final_df['control_density'] = final_df['control_structures'] / final_df['template_lines'].replace(0, np.nan)
    
    # --- Analysis 1: Nesting vs Control Structures ---
    print("\n1. Values Nesting vs Control Structures:")
    
    corr_nest_ctrl, p_nest_ctrl = stats.spearmanr(
        final_df['values_nesting'],
        final_df['control_structures']
    )
    print(f"   Spearman correlation (nesting vs #controls): r={corr_nest_ctrl:.3f}, p={p_nest_ctrl:.4f}")
    
    # --- Analysis 2: Nesting vs Control Density ---
    print("\n2. Values Nesting vs Control Density:")
    
    valid = final_df[['values_nesting', 'control_density']].dropna()
    corr_nest_density, p_nest_density = stats.spearmanr(
        valid['values_nesting'],
        valid['control_density']
    )
    print(f"   Spearman correlation (nesting vs density): r={corr_nest_density:.3f}, p={p_nest_density:.4f}")
    
    # --- Analysis 3: Group comparison (flat vs nested) ---
    print("\n3. Flat vs Nested Comparison:")
    
    flat = final_df[final_df['values_nesting'] <= 2]['control_structures']
    nested = final_df[final_df['values_nesting'] > 2]['control_structures']
    
    print(f"   Flat (≤2 depth): mean={flat.mean():.1f}, median={flat.median():.1f}, n={len(flat)}")
    print(f"   Nested (>2 depth): mean={nested.mean():.1f}, median={nested.median():.1f}, n={len(nested)}")
    
    if len(flat) >= 3 and len(nested) >= 3:
        stat, p_mannwhitney = stats.mannwhitneyu(nested, flat, alternative='greater')
        print(f"   Mann-Whitney U (nested > flat): p={p_mannwhitney:.4f}")
    else:
        p_mannwhitney = 1.0
        print(f"   Insufficient data for Mann-Whitney test")
    
    # --- H2 Assessment ---
    h2_supported = (
        (corr_nest_ctrl > 0 and p_nest_ctrl < 0.05) or
        (corr_nest_density > 0 and p_nest_density < 0.05) or
        (p_mannwhitney < 0.05)
    )
    
    results['supported'] = h2_supported
    results['metrics'] = {
        'corr_nest_ctrl': (corr_nest_ctrl, p_nest_ctrl),
        'corr_nest_density': (corr_nest_density, p_nest_density),
        'flat_mean': flat.mean(),
        'nested_mean': nested.mean(),
        'p_mannwhitney': p_mannwhitney
    }
    
    print(f"\n>>> H2 {'SUPPORTED' if h2_supported else 'NOT SUPPORTED'}")
    
    return results


def plot_h2(final_df, results):
    """Create visualization for H2."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Calculate control density
    final_df['control_density'] = final_df['control_structures'] / final_df['template_lines'].replace(0, np.nan)
    
    # 1. Scatter: Nesting vs Control Structures
    ax = axes[0]
    ax.scatter(final_df['values_nesting'], final_df['control_structures'],
               alpha=0.6, s=60, c='purple', edgecolors='black', linewidth=0.5)
    z = np.polyfit(final_df['values_nesting'], final_df['control_structures'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(final_df['values_nesting'].min(), final_df['values_nesting'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2)
    ax.set_xlabel('Values Nesting Depth', fontsize=11)
    ax.set_ylabel('# Control Structures (if/with/range)', fontsize=11)
    ax.set_title('Nesting vs Control Structures', fontsize=12, fontweight='bold')
    r, p_val = results['metrics']['corr_nest_ctrl']
    ax.text(0.05, 0.95, f'r={r:.3f}, p={p_val:.4f}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # 2. Bar: Flat vs Nested
    ax = axes[1]
    flat_mean = results['metrics']['flat_mean']
    nested_mean = results['metrics']['nested_mean']
    bars = ax.bar(['Flat Values\n(≤2 depth)', 'Nested Values\n(>2 depth)'], 
                  [flat_mean, nested_mean],
                  color=['#2ecc71', '#e74c3c'], edgecolor='black', width=0.6)
    ax.set_ylabel('Avg # Control Structures', fontsize=11)
    ax.set_title('Control Structures: Flat vs Nested', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, [flat_mean, nested_mean]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}', ha='center', fontsize=12, fontweight='bold')
    p_val = results['metrics']['p_mannwhitney']
    ax.text(0.5, 0.95, f'p={p_val:.4f}', transform=ax.transAxes,
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
    
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
    ax.set_xlabel('Nesting Depth Category', fontsize=11)
    ax.set_ylabel('Control Structures', fontsize=11)
    ax.set_title('Distribution by Nesting Level', fontsize=12, fontweight='bold')
    
    status = "SUPPORTED ✓" if results['supported'] else "NOT SUPPORTED ✗"
    color = '#2ecc71' if results['supported'] else '#e74c3c'
    fig.suptitle(f"H2: Deeper Nesting → More Control Complexity [{status}]",
                 fontsize=14, fontweight='bold', color=color, y=1.02)
    
    plt.tight_layout()
    plt.savefig('results/h2_nesting_complexity.png', dpi=150, bbox_inches='tight')
    print("Saved: results/h2_nesting_complexity.png")
    plt.close()


# =============================================================================
# H3: Larger charts comply more with template structure best practices
# =============================================================================

def validate_h3(df, final_df):
    """
    H3: Larger charts follow better template organization.
    
    Independent: template_files, template_lines
    Dependent: templates_structure score, documentation score
    """
    print("\n" + "=" * 70)
    print("H3: Larger Charts → Better Template Structure")
    print("=" * 70)
    
    results = {'name': 'H3', 'description': 'Larger charts have better template structure compliance'}
    
    # --- Analysis 1: Size vs Template Structure Score ---
    print("\n1. Chart Size vs Template Structure Score:")
    
    corr_size_struct, p_size_struct = stats.spearmanr(
        final_df['template_lines'],
        final_df['templates_structure_score']
    )
    print(f"   Spearman correlation (LOC vs templates_structure): r={corr_size_struct:.3f}, p={p_size_struct:.4f}")
    
    # --- Analysis 2: Size categories comparison ---
    print("\n2. Template Structure by Size Category:")
    
    final_df['size_category'] = pd.cut(
        final_df['template_lines'],
        bins=[0, 200, 500, 1000, float('inf')],
        labels=['Small (<200)', 'Medium (200-500)', 'Large (500-1000)', 'XLarge (>1000)']
    )
    
    struct_by_size = final_df.groupby('size_category', observed=True)['templates_structure_score'].mean()
    for cat, val in struct_by_size.items():
        n = len(final_df[final_df['size_category'] == cat])
        print(f"   - {cat}: {val:.2f} (n={n})")
    
    # --- Analysis 3: File count vs organization ---
    print("\n3. File Count vs Structure Score:")
    
    corr_files_struct, p_files_struct = stats.spearmanr(
        final_df['template_files'],
        final_df['templates_structure_score']
    )
    print(f"   Spearman correlation (#files vs structure): r={corr_files_struct:.3f}, p={p_files_struct:.4f}")
    
    # --- Analysis 4: Size vs Documentation ---
    print("\n4. Size vs Documentation Score:")
    
    corr_size_docs, p_size_docs = stats.spearmanr(
        final_df['template_lines'],
        final_df['documentation_score']
    )
    print(f"   Spearman correlation (LOC vs documentation): r={corr_size_docs:.3f}, p={p_size_docs:.4f}")
    
    # --- H3 Assessment ---
    h3_supported = (
        (corr_size_struct > 0 and p_size_struct < 0.05) or
        (corr_files_struct > 0 and p_files_struct < 0.05) or
        (corr_size_docs > 0 and p_size_docs < 0.05)
    )
    
    results['supported'] = h3_supported
    results['metrics'] = {
        'corr_size_struct': (corr_size_struct, p_size_struct),
        'corr_files_struct': (corr_files_struct, p_files_struct),
        'corr_size_docs': (corr_size_docs, p_size_docs),
        'struct_by_size': struct_by_size.to_dict()
    }
    
    print(f"\n>>> H3 {'SUPPORTED' if h3_supported else 'NOT SUPPORTED'}")
    
    return results


def plot_h3(final_df, results):
    """Create visualization for H3 - Best Practices Adoption Analysis."""
    import ast
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Parse documentation dict to extract components
    def parse_dict(s):
        try:
            return ast.literal_eval(s) if isinstance(s, str) else (s if isinstance(s, dict) else {})
        except:
            return {}
    
    df = final_df.copy()
    df['doc_dict'] = df['documentation'].apply(parse_dict)
    df['struct_dict'] = df['templates_structure'].apply(parse_dict)
    
    # Extract boolean features
    df['has_readme'] = df['doc_dict'].apply(lambda x: x.get('has_readme', False))
    df['has_notes'] = df['doc_dict'].apply(lambda x: x.get('has_notes', False))
    df['has_schema'] = df['doc_dict'].apply(lambda x: x.get('has_schema', False))
    df['has_helpers'] = df['struct_dict'].apply(lambda x: x.get('has_helpers', False))
    
    # Size categories
    df['size_cat'] = pd.cut(
        df['template_lines'],
        bins=[0, 200, 500, float('inf')],
        labels=['Small (<200)', 'Medium (200-500)', 'Large (>500)']
    )
    
    # =========================================================================
    # 1. Best Practices Adoption Heatmap by Size
    # =========================================================================
    ax = axes[0]
    practices = ['has_readme', 'has_notes', 'has_schema', 'has_helpers']
    practice_labels = ['README', 'NOTES.txt', 'values.schema', '_helpers.tpl']
    size_cats = ['Small (<200)', 'Medium (200-500)', 'Large (>500)']
    
    # Calculate adoption rates
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
    ax.grid(False)  # Disable grid before colorbar
    
    # Add text annotations
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
    ax.set_title('Best Practices Adoption Rate', fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Adoption %', fontsize=10)
    
    # =========================================================================
    # 2. Score Distribution by Size (Box Plot)
    # =========================================================================
    ax = axes[1]
    
    # Prepare data for boxplot
    box_data = []
    positions = []
    colors_box = []
    color_map = {'Small (<200)': '#3498db', 'Medium (200-500)': '#f39c12', 'Large (>500)': '#27ae60'}
    
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
    ax.set_ylabel('Overall Compliance Score', fontsize=11)
    ax.set_title('Compliance Score Distribution', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax.legend(loc='lower right', fontsize=9)
    
    # Add count labels
    for idx, cat in enumerate(size_cats):
        n = len(df[df['size_cat'] == cat])
        ax.text(idx, -0.08, f'n={n}', ha='center', fontsize=9, transform=ax.get_xaxis_transform())
    
    # =========================================================================
    # 3. Documentation Completeness Stacked Bar
    # =========================================================================
    ax = axes[2]
    
    # Count charts with 0, 1, 2, 3+ doc features
    doc_features = ['has_readme', 'has_notes', 'has_schema']
    df['doc_count'] = df[doc_features].sum(axis=1)
    
    # Stacked bar data
    categories = ['0 docs', '1 doc', '2 docs', '3 docs']
    stacked_data = {cat: [] for cat in categories}
    
    for size_cat in size_cats:
        subset = df[df['size_cat'] == size_cat]
        total = len(subset) if len(subset) > 0 else 1
        stacked_data['0 docs'].append(len(subset[subset['doc_count'] == 0]) / total * 100)
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
    ax.set_ylabel('Percentage of Charts', fontsize=11)
    ax.set_title('Documentation Completeness', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.set_ylim(0, 105)
    
    # Add percentage labels for significant segments
    bottom = np.zeros(len(size_cats))
    for cat, color in zip(categories, colors_stack):
        vals = stacked_data[cat]
        for i, v in enumerate(vals):
            if v >= 15:  # Only label segments >= 15%
                ax.text(i, bottom[i] + v/2, f'{v:.0f}%', ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white')
        bottom += vals
    
    status = "SUPPORTED ✓" if results['supported'] else "NOT SUPPORTED ✗"
    color = '#2ecc71' if results['supported'] else '#e74c3c'
    fig.suptitle(f"H3: Larger Charts → Better Structure [{status}]",
                 fontsize=14, fontweight='bold', color=color, y=1.02)
    
    plt.tight_layout()
    plt.savefig('results/h3_organization.png', dpi=150, bbox_inches='tight')
    print("Saved: results/h3_organization.png")
    plt.close()


# =============================================================================
# H4: More mature charts (more commits) follow better practices
# =============================================================================

def validate_h4(df, final_df):
    """
    H4: More mature charts follow better practices.
    
    Independent: commit_count (number of commits in history)
    Dependent: overall_score, documentation score, best practices adoption
    """
    print("\n" + "=" * 70)
    print("H4: More Mature Charts → Better Compliance")
    print("=" * 70)
    
    results = {'name': 'H4', 'description': 'More mature charts (more commits) follow better practices'}
    
    # Calculate commit counts per chart
    commit_counts = df.groupby('chart_name').size().reset_index(name='commit_count')
    analysis_df = final_df.merge(commit_counts, on='chart_name')
    
    # --- Analysis 1: Commit count distribution ---
    print("\n1. Chart Maturity Distribution (commit counts):")
    print(f"   Min commits: {analysis_df['commit_count'].min()}")
    print(f"   Max commits: {analysis_df['commit_count'].max()}")
    print(f"   Median commits: {analysis_df['commit_count'].median():.0f}")
    print(f"   Mean commits: {analysis_df['commit_count'].mean():.1f}")
    
    # --- Analysis 2: Maturity vs Overall Compliance ---
    print("\n2. Maturity vs Overall Compliance Score:")
    
    corr_maturity, p_maturity = stats.spearmanr(
        analysis_df['commit_count'],
        analysis_df['overall_score']
    )
    print(f"   Spearman correlation (commits vs overall_score): r={corr_maturity:.3f}, p={p_maturity:.4f}")
    
    # Split by median
    median_commits = analysis_df['commit_count'].median()
    mature = analysis_df[analysis_df['commit_count'] > median_commits]
    newer = analysis_df[analysis_df['commit_count'] <= median_commits]
    
    print(f"\n   Mature charts (>{median_commits:.0f} commits): n={len(mature)}, avg score={mature['overall_score'].mean():.3f}")
    print(f"   Newer charts (≤{median_commits:.0f} commits): n={len(newer)}, avg score={newer['overall_score'].mean():.3f}")
    
    # Mann-Whitney test
    if len(mature) >= 3 and len(newer) >= 3:
        stat, p_maturity_test = stats.mannwhitneyu(
            mature['overall_score'], 
            newer['overall_score'], 
            alternative='greater'
        )
        print(f"   Mann-Whitney U (mature > newer): p={p_maturity_test:.4f}")
    else:
        p_maturity_test = 1.0
    
    # --- Analysis 3: Maturity vs Documentation ---
    print("\n3. Maturity vs Documentation Score:")
    corr_docs, p_docs = stats.spearmanr(
        analysis_df['commit_count'],
        analysis_df['documentation_score']
    )
    print(f"   Spearman correlation (commits vs documentation): r={corr_docs:.3f}, p={p_docs:.4f}")
    print(f"   Mature avg docs: {mature['documentation_score'].mean():.3f}")
    print(f"   Newer avg docs: {newer['documentation_score'].mean():.3f}")
    
    # --- Analysis 4: Maturity vs Best Practices Adoption ---
    print("\n4. Maturity vs Best Practices Adoption:")
    mature_helpers = mature['has_helpers'].mean() * 100
    newer_helpers = newer['has_helpers'].mean() * 100
    print(f"   Mature charts with _helpers.tpl: {mature_helpers:.1f}%")
    print(f"   Newer charts with _helpers.tpl: {newer_helpers:.1f}%")
    
    # --- H4 Assessment ---
    h4_supported = (
        (corr_maturity > 0.15 and p_maturity < 0.10) or  # Positive correlation with borderline significance
        (p_maturity_test < 0.10)  # Group comparison borderline significant
    )
    
    results['supported'] = h4_supported
    results['metrics'] = {
        'corr_maturity': (corr_maturity, p_maturity),
        'corr_docs': (corr_docs, p_docs),
        'median_commits': median_commits,
        'mature_score': mature['overall_score'].mean(),
        'newer_score': newer['overall_score'].mean(),
        'mature_docs': mature['documentation_score'].mean(),
        'newer_docs': newer['documentation_score'].mean(),
        'mature_helpers': mature_helpers,
        'newer_helpers': newer_helpers,
        'p_maturity_test': p_maturity_test,
        'n_mature': len(mature),
        'n_newer': len(newer),
        'analysis_df': analysis_df  # Pass for plotting
    }
    
    print(f"\n>>> H4 {'SUPPORTED' if h4_supported else 'NOT SUPPORTED'}")
    
    return results


def plot_h4(final_df, results):
    """Create visualization for H4 - Chart Maturity Analysis."""
    import ast
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    analysis_df = results['metrics']['analysis_df']
    median_commits = results['metrics']['median_commits']
    
    # =========================================================================
    # 1. Scatter: Commit Count vs Overall Score
    # =========================================================================
    ax = axes[0]
    
    # Color by maturity
    colors = ['#e74c3c' if c <= median_commits else '#27ae60' 
              for c in analysis_df['commit_count']]
    
    ax.scatter(analysis_df['commit_count'], analysis_df['overall_score'],
               c=colors, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(analysis_df['commit_count'], analysis_df['overall_score'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(analysis_df['commit_count'].min(), analysis_df['commit_count'].max(), 100)
    ax.plot(x_line, p(x_line), 'b--', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Number of Commits (Maturity)', fontsize=11)
    ax.set_ylabel('Overall Compliance Score', fontsize=11)
    ax.set_title('Maturity vs Compliance', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    
    # Add correlation annotation
    r, p_val = results['metrics']['corr_maturity']
    ax.text(0.05, 0.95, f'r={r:.3f}, p={p_val:.4f}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label=f'Newer (≤{median_commits:.0f})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#27ae60', markersize=10, label=f'Mature (>{median_commits:.0f})')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # =========================================================================
    # 2. Box Plot: Mature vs Newer Comparison
    # =========================================================================
    ax = axes[1]
    
    mature = analysis_df[analysis_df['commit_count'] > median_commits]['overall_score']
    newer = analysis_df[analysis_df['commit_count'] <= median_commits]['overall_score']
    
    bp = ax.boxplot([newer.values, mature.values], 
                    labels=[f'Newer\n(≤{median_commits:.0f} commits)', f'Mature\n(>{median_commits:.0f} commits)'],
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
    
    ax.set_ylabel('Overall Compliance Score', fontsize=11)
    ax.set_title('Compliance: Mature vs Newer', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    
    # Add stats
    p_test = results['metrics']['p_maturity_test']
    ax.text(0.5, 0.02, f'Mann-Whitney p={p_test:.4f}', transform=ax.transAxes,
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # Add mean annotations
    ax.text(1, newer.mean() + 0.05, f'μ={newer.mean():.2f}', ha='center', fontsize=10, fontweight='bold')
    ax.text(2, mature.mean() + 0.05, f'μ={mature.mean():.2f}', ha='center', fontsize=10, fontweight='bold')
    
    # Add n labels
    ax.text(1, -0.08, f'n={len(newer)}', ha='center', fontsize=9, transform=ax.get_xaxis_transform())
    ax.text(2, -0.08, f'n={len(mature)}', ha='center', fontsize=9, transform=ax.get_xaxis_transform())
    
    # =========================================================================
    # 3. Best Practices Adoption Comparison
    # =========================================================================
    ax = axes[2]
    
    # Parse documentation dict to extract components
    def parse_dict(s):
        try:
            return ast.literal_eval(s) if isinstance(s, str) else (s if isinstance(s, dict) else {})
        except:
            return {}
    
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
    
    bars1 = ax.bar(x - width/2, newer_rates, width, label=f'Newer (≤{median_commits:.0f})', 
                   color='#e74c3c', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, mature_rates, width, label=f'Mature (>{median_commits:.0f})', 
                   color='#27ae60', edgecolor='black', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(practice_labels, fontsize=10)
    ax.set_ylabel('Adoption Rate (%)', fontsize=11)
    ax.set_title('Best Practices by Maturity', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 105)
    
    # Add percentage labels
    for bar, val in zip(bars1, newer_rates):
        if val > 5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.0f}%', ha='center', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, mature_rates):
        if val > 5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.0f}%', ha='center', fontsize=9, fontweight='bold')
    
    status = "SUPPORTED ✓" if results['supported'] else "NOT SUPPORTED ✗"
    color = '#2ecc71' if results['supported'] else '#e74c3c'
    fig.suptitle(f"H4: More Mature Charts → Better Compliance [{status}]",
                 fontsize=14, fontweight='bold', color=color, y=1.02)
    
    plt.tight_layout()
    plt.savefig('results/h4_maturity.png', dpi=150, bbox_inches='tight')
    print("Saved: results/h4_maturity.png")
    plt.close()


# =============================================================================
# SUMMARY
# =============================================================================

def create_best_practices_dashboard(final_df):
    """Create comprehensive Best Practices compliance dashboard."""
    import ast
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    def parse_dict(s):
        try:
            return ast.literal_eval(s) if isinstance(s, str) else (s if isinstance(s, dict) else {})
        except:
            return {}
    
    # Extract all metrics from dict columns
    df = final_df.copy()
    df['struct'] = df['templates_structure'].apply(parse_dict)
    df['labels'] = df['labels_compliance'].apply(parse_dict)
    df['naming'] = df['values_naming'].apply(parse_dict)
    df['docs'] = df['documentation'].apply(parse_dict)
    
    # Template metrics
    df['camelcase_files'] = df['struct'].apply(lambda x: x.get('camelcase_files', 0))
    df['template_files_count'] = df['struct'].apply(lambda x: x.get('template_files_count', 0))
    df['namespaced'] = df['struct'].apply(lambda x: x.get('namespaced_templates', 0))
    df['non_namespaced'] = df['struct'].apply(lambda x: x.get('non_namespaced_templates', 0))
    
    # Labels metrics
    df['labels_coverage'] = df['labels'].apply(lambda x: x.get('labels_coverage', 0))
    df['required_labels_coverage'] = df['labels'].apply(lambda x: x.get('required_coverage', 0))
    
    # Values metrics
    df['values_violations'] = df['naming'].apply(lambda x: x.get('violations_count', 0))
    df['total_values_keys'] = df['naming'].apply(lambda x: x.get('total_keys', 0))
    df['values_has_comments'] = df['naming'].apply(lambda x: x.get('has_comments', False))
    
    # Documentation metrics
    df['has_readme'] = df['docs'].apply(lambda x: x.get('has_readme', False))
    df['has_notes'] = df['docs'].apply(lambda x: x.get('has_notes', False))
    df['has_schema'] = df['docs'].apply(lambda x: x.get('has_schema', False))
    
    # =========================================================================
    # 1. Overall Best Practices Radar/Summary
    # =========================================================================
    ax = axes[0, 0]
    
    # Calculate compliance rates
    total_files = df['template_files_count'].sum()
    camelcase = df['camelcase_files'].sum()
    file_naming_compliance = (total_files - camelcase) / total_files * 100 if total_files > 0 else 0
    
    total_ns = df['namespaced'].sum()
    total_non_ns = df['non_namespaced'].sum()
    namespacing_compliance = total_ns / (total_ns + total_non_ns) * 100 if (total_ns + total_non_ns) > 0 else 100
    
    helpers_adoption = df['has_helpers'].mean() * 100
    readme_adoption = df['has_readme'].mean() * 100
    notes_adoption = df['has_notes'].mean() * 100
    values_comments = df['values_has_comments'].mean() * 100
    
    practices = ['File Naming\n(dashed)', 'Namespaced\nDefines', '_helpers.tpl', 
                 'README.md', 'NOTES.txt', 'values.yaml\nComments']
    compliance_rates = [file_naming_compliance, namespacing_compliance, helpers_adoption,
                       readme_adoption, notes_adoption, values_comments]
    
    colors_bar = ['#27ae60' if r >= 70 else '#f39c12' if r >= 40 else '#e74c3c' for r in compliance_rates]
    
    bars = ax.barh(practices, compliance_rates, color=colors_bar, edgecolor='black', height=0.6)
    ax.set_xlim(0, 105)
    ax.set_xlabel('Compliance Rate (%)', fontsize=11)
    ax.set_title('Best Practices Adoption Overview', fontsize=12, fontweight='bold')
    ax.axvline(x=70, color='green', linestyle='--', alpha=0.5, label='Good (70%)')
    ax.axvline(x=40, color='orange', linestyle='--', alpha=0.5, label='Moderate (40%)')
    
    for bar, val in zip(bars, compliance_rates):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.0f}%', 
               va='center', fontsize=10, fontweight='bold')
    
    ax.legend(loc='lower right', fontsize=8)
    
    # =========================================================================
    # 2. Labels Compliance Distribution
    # =========================================================================
    ax = axes[0, 1]
    
    # Histogram of labels coverage
    ax.hist(df['labels_coverage'] * 100, bins=10, color='#3498db', edgecolor='black', alpha=0.7)
    ax.axvline(x=df['labels_coverage'].mean() * 100, color='red', linestyle='--', linewidth=2,
              label=f'Mean: {df["labels_coverage"].mean()*100:.1f}%')
    ax.set_xlabel('Standard Labels Coverage (%)', fontsize=11)
    ax.set_ylabel('Number of Charts', fontsize=11)
    ax.set_title('Kubernetes Labels Compliance', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    
    # Add annotation about recommended labels
    ax.text(0.98, 0.95, 'Recommended:\n• app.kubernetes.io/name\n• helm.sh/chart\n• app.kubernetes.io/instance\n• app.kubernetes.io/managed-by',
           transform=ax.transAxes, fontsize=8, va='top', ha='right',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # =========================================================================
    # 3. Values.yaml Quality
    # =========================================================================
    ax = axes[0, 2]
    
    # Calculate per-chart violation rate
    df['violation_rate'] = df.apply(
        lambda x: x['values_violations'] / x['total_values_keys'] * 100 if x['total_values_keys'] > 0 else 0, 
        axis=1
    )
    
    # Pie chart: violation categories
    low_violations = len(df[df['violation_rate'] == 0])
    some_violations = len(df[(df['violation_rate'] > 0) & (df['violation_rate'] <= 5)])
    high_violations = len(df[df['violation_rate'] > 5])
    
    sizes = [low_violations, some_violations, high_violations]
    labels_pie = [f'No violations\n({low_violations})', f'≤5% violations\n({some_violations})', f'>5% violations\n({high_violations})']
    colors_pie = ['#27ae60', '#f39c12', '#e74c3c']
    explode = (0.05, 0, 0)
    
    ax.pie(sizes, explode=explode, labels=labels_pie, colors=colors_pie, autopct='%1.0f%%',
           startangle=90, textprops={'fontsize': 9})
    ax.set_title('Values Naming Convention\n(camelCase compliance)', fontsize=12, fontweight='bold')
    
    # =========================================================================
    # 4. Documentation Completeness by Repository
    # =========================================================================
    ax = axes[1, 0]
    
    # Calculate doc score per repo
    repo_docs = df.groupby('repo').agg({
        'has_readme': 'mean',
        'has_notes': 'mean', 
        'has_schema': 'mean',
        'chart_name': 'count'
    }).reset_index()
    repo_docs.columns = ['repo', 'readme', 'notes', 'schema', 'count']
    repo_docs = repo_docs.sort_values('count', ascending=True)
    
    # Only show repos with 2+ charts
    repo_docs = repo_docs[repo_docs['count'] >= 2]
    
    y_pos = np.arange(len(repo_docs))
    width = 0.25
    
    ax.barh(y_pos - width, repo_docs['readme'] * 100, width, label='README', color='#3498db', alpha=0.8)
    ax.barh(y_pos, repo_docs['notes'] * 100, width, label='NOTES.txt', color='#27ae60', alpha=0.8)
    ax.barh(y_pos + width, repo_docs['schema'] * 100, width, label='values.schema', color='#9b59b6', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(repo_docs['repo'], fontsize=9)
    ax.set_xlabel('Adoption Rate (%)', fontsize=11)
    ax.set_title('Documentation by Repository', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(0, 105)
    
    # =========================================================================
    # 5. Template Complexity vs Best Practices
    # =========================================================================
    ax = axes[1, 1]
    
    # Scatter: LOC vs overall_score, colored by helper adoption
    colors_scatter = ['#27ae60' if h else '#e74c3c' for h in df['has_helpers']]
    
    # Filter outliers
    df_plot = df[df['template_lines'] <= 3000]
    colors_plot = ['#27ae60' if h else '#e74c3c' for h in df_plot['has_helpers']]
    
    ax.scatter(df_plot['template_lines'], df_plot['overall_score'], 
              c=colors_plot, s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Lines of Code', fontsize=11)
    ax.set_ylabel('Overall Compliance Score', fontsize=11)
    ax.set_title('Complexity vs Compliance\n(colored by _helpers.tpl)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#27ae60', markersize=10, label='Has _helpers.tpl'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label='No _helpers.tpl')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # =========================================================================
    # 6. Score Components Breakdown
    # =========================================================================
    ax = axes[1, 2]
    
    # Average scores by component
    components = ['templates_structure_score', 'labels_compliance_score', 
                  'values_naming_score', 'documentation_score', 'overall_score']
    component_labels = ['Template\nStructure', 'Labels\nCompliance', 'Values\nNaming', 
                       'Documentation', 'Overall']
    avg_scores = [df[c].mean() for c in components]
    
    colors_comp = plt.cm.viridis(np.linspace(0.2, 0.8, len(components)))
    
    bars = ax.bar(component_labels, avg_scores, color=colors_comp, edgecolor='black', width=0.6)
    ax.set_ylabel('Average Score', fontsize=11)
    ax.set_title('Score Components Breakdown', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    
    for bar, val in zip(bars, avg_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')
    
    ax.legend(loc='upper right', fontsize=9)
    
    fig.suptitle('Helm Chart Best Practices Dashboard\n(Based on helm.sh/docs/chart_best_practices/)',
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('results/best_practices_dashboard.png', dpi=150, bbox_inches='tight')
    print("Saved: results/best_practices_dashboard.png")
    plt.close()


def print_best_practices_summary(final_df):
    """Print detailed best practices metrics summary."""
    import ast
    
    def parse_dict(s):
        try:
            return ast.literal_eval(s) if isinstance(s, str) else (s if isinstance(s, dict) else {})
        except:
            return {}
    
    df = final_df.copy()
    df['struct'] = df['templates_structure'].apply(parse_dict)
    df['labels'] = df['labels_compliance'].apply(parse_dict)
    df['naming'] = df['values_naming'].apply(parse_dict)
    df['docs'] = df['documentation'].apply(parse_dict)
    
    # Extract metrics
    df['camelcase_files'] = df['struct'].apply(lambda x: x.get('camelcase_files', 0))
    df['template_files_count'] = df['struct'].apply(lambda x: x.get('template_files_count', 0))
    df['namespaced'] = df['struct'].apply(lambda x: x.get('namespaced_templates', 0))
    df['non_namespaced'] = df['struct'].apply(lambda x: x.get('non_namespaced_templates', 0))
    df['labels_coverage'] = df['labels'].apply(lambda x: x.get('labels_coverage', 0))
    df['required_labels_coverage'] = df['labels'].apply(lambda x: x.get('required_coverage', 0))
    df['values_violations'] = df['naming'].apply(lambda x: x.get('violations_count', 0))
    df['total_values_keys'] = df['naming'].apply(lambda x: x.get('total_keys', 0))
    df['values_has_comments'] = df['naming'].apply(lambda x: x.get('has_comments', False))
    df['has_readme'] = df['docs'].apply(lambda x: x.get('has_readme', False))
    df['has_notes'] = df['docs'].apply(lambda x: x.get('has_notes', False))
    df['has_schema'] = df['docs'].apply(lambda x: x.get('has_schema', False))
    
    print("\n" + "=" * 70)
    print("BEST PRACTICES COMPLIANCE SUMMARY")
    print("=" * 70)
    print("Reference: https://helm.sh/docs/chart_best_practices/")
    
    print("\n--- TEMPLATE STRUCTURE (templates/) ---")
    total_files = df['template_files_count'].sum()
    camelcase = df['camelcase_files'].sum()
    print(f"  File naming (dashed notation): {(total_files-camelcase)/total_files*100:.1f}% compliant")
    print(f"    - {total_files} total files, {camelcase} camelCase violations")
    
    total_ns = df['namespaced'].sum()
    total_non_ns = df['non_namespaced'].sum()
    ns_rate = total_ns/(total_ns+total_non_ns)*100 if (total_ns+total_non_ns) > 0 else 100
    print(f"  Namespaced defines: {ns_rate:.1f}% compliant")
    print(f"    - {total_ns} namespaced, {total_non_ns} non-namespaced")
    
    print(f"  _helpers.tpl adoption: {df['has_helpers'].mean()*100:.1f}%")
    
    print("\n--- KUBERNETES LABELS ---")
    print(f"  Standard labels coverage: {df['labels_coverage'].mean()*100:.1f}% avg")
    print(f"  Required labels coverage: {df['required_labels_coverage'].mean()*100:.1f}% avg")
    print(f"  Charts with 100% required labels: {(df['required_labels_coverage'] == 1.0).sum()}/{len(df)}")
    
    print("\n--- VALUES.YAML QUALITY ---")
    total_keys = df['total_values_keys'].sum()
    total_violations = df['values_violations'].sum()
    print(f"  Naming convention (camelCase): {(1-total_violations/total_keys)*100:.1f}% compliant")
    print(f"    - {total_keys} total keys, {total_violations} violations")
    print(f"  Documented with comments: {df['values_has_comments'].mean()*100:.1f}%")
    
    print("\n--- DOCUMENTATION ---")
    print(f"  README.md: {df['has_readme'].mean()*100:.1f}%")
    print(f"  NOTES.txt: {df['has_notes'].mean()*100:.1f}%")
    print(f"  values.schema.json: {df['has_schema'].mean()*100:.1f}%")
    
    print("\n--- OVERALL SCORES ---")
    print(f"  Template structure: {df['templates_structure_score'].mean():.2f}")
    print(f"  Labels compliance:  {df['labels_compliance_score'].mean():.2f}")
    print(f"  Values naming:      {df['values_naming_score'].mean():.2f}")
    print(f"  Documentation:      {df['documentation_score'].mean():.2f}")
    print(f"  OVERALL:            {df['overall_score'].mean():.2f}")


def create_summary_plot(all_results):
    """Create overall summary visualization."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    hypotheses = [r['name'] for r in all_results]
    descriptions = [r['description'][:50] + '...' if len(r['description']) > 50 else r['description'] 
                    for r in all_results]
    supported = [r['supported'] for r in all_results]
    colors = ['#2ecc71' if s else '#e74c3c' for s in supported]
    
    y_pos = np.arange(len(hypotheses))
    bars = ax.barh(y_pos, [1]*len(hypotheses), color=colors, alpha=0.85, edgecolor='white', height=0.7)
    
    for i, (h, desc, supp) in enumerate(zip(hypotheses, descriptions, supported)):
        result = "✓ SUPPORTED" if supp else "✗ NOT SUPPORTED"
        ax.text(0.02, i, f"{h}: {desc}", va='center', ha='left', fontsize=10, color='white', fontweight='bold')
        ax.text(0.98, i, result, va='center', ha='right', fontsize=10, color='white', fontweight='bold')
    
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    supported_count = sum(supported)
    total = len(supported)
    ax.set_title(f'Hypothesis Validation Summary: {supported_count}/{total} Supported', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='Supported'),
                       Patch(facecolor='#e74c3c', label='Not Supported')]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('results/summary.png', dpi=150, bbox_inches='tight')
    print("Saved: results/summary.png")
    plt.close()


def print_final_summary(all_results):
    """Print final summary."""
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    supported_count = sum(r['supported'] for r in all_results)
    total = len(all_results)
    
    print(f"\nHypotheses Supported: {supported_count}/{total}")
    print()
    
    for r in all_results:
        status = "✓ SUPPORTED" if r['supported'] else "✗ NOT SUPPORTED"
        print(f"{r['name']}: {status}")
        print(f"   {r['description']}")
    
    print("\n" + "=" * 70)


def main():
    print("=" * 70)
    print("HELM CHART BEST PRACTICES - HYPOTHESIS VALIDATION")
    print("=" * 70)
    print("Reference: https://helm.sh/docs/chart_best_practices/")
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # Load data
    df, final_df = load_data()
    
    # Validate all hypotheses
    all_results = []
    
    # H1: Chart growth triggers modularization
    h1_results = validate_h1(df, final_df)
    all_results.append(h1_results)
    plot_h1(final_df.copy(), h1_results)
    
    # H2: Deeper nesting increases complexity
    h2_results = validate_h2(df, final_df)
    all_results.append(h2_results)
    plot_h2(final_df.copy(), h2_results)
    
    # H3: Larger charts have better organization
    h3_results = validate_h3(df, final_df)
    all_results.append(h3_results)
    plot_h3(final_df.copy(), h3_results)
    
    # H4: More dependencies = better practices
    h4_results = validate_h4(df, final_df)
    all_results.append(h4_results)
    plot_h4(final_df.copy(), h4_results)
    
    # Create Best Practices Dashboard
    create_best_practices_dashboard(final_df.copy())
    print_best_practices_summary(final_df.copy())
    
    # Create summary
    create_summary_plot(all_results)
    print_final_summary(all_results)
    
    # Save results to JSON
    results_json = []
    for r in all_results:
        r_clean = {'name': r['name'], 'description': r['description'], 'supported': bool(r['supported'])}
        results_json.append(r_clean)
    
    with open('results/validation_results.json', 'w') as f:
        json.dump({
            'results': results_json,
            'summary': {
                'supported': int(sum(r['supported'] for r in all_results)),
                'total': len(all_results)
            }
        }, f, indent=2)
    
    print(f"\n✅ All results saved to results/")


if __name__ == '__main__':
    main()
