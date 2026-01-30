#!/usr/bin/env python3
"""
Plot histograms for Helm chart cognitive complexity metrics.
Reads metrics_data.json and creates histogram visualizations for each metric.

Cognitive Complexity Metrics:
1. Size - Number of nodes in the chart
2. Comprehension Scope - Fraction of graph to understand per resource
3. Cognitive Diameter - Maximum distance between resources
4. Hub Dominance - Dependency on critical nodes
5. Modification Isolation - Independence of resource modifications
6. Helper Justification - Fraction of helpers that are reused
7. Blast Radius Variance - Predictability of change impact

Best Practice Metrics:
8.  Max Nesting Depth - Deepest template nesting level
9.  Unguarded Nested Access - Unsafe property accesses
10. Array Config Count - Array-based configurations
11. Hardcoded Image Count - Non-parameterized image references
12. Multi-Resource File Count - Files with multiple resources
13. Unquoted String Count - Potentially unsafe string values
14. Floating Image Tag Count - Non-pinned image versions
15. Mutable Selector Label - Changeable pod selectors
16. Missing Pod Selector - Network policies without selectors
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_metrics(filename="metrics_data.json"):
    """Load metrics data from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def plot_histograms(metrics_data, output_dir="plots"):
    """Create histogram plots for each metric."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)

    # Define metrics to plot (all numeric fields in GraphMetrics)
    metric_fields = [
        'size',
        'comprehension_scope',
        'cognitive_diameter',
        'hub_dominance',
        'modification_isolation',
        'helper_justification_ratio',
        'blast_radius_variance',
        # Best practice metrics
        'max_nesting_depth',
        'unguarded_nested_access',
        'array_config_count',
        'hardcoded_image_count',
        'multi_resource_file_count',
        'unquoted_string_count',
        'floating_image_tag_count',
        'mutable_selector_label',
        'missing_pod_selector'
    ]

    # Define nicer labels for plots
    metric_labels = {
        'size': 'Size: Number of nodes (|V|)',
        'comprehension_scope': 'Comprehension Scope: Fraction of graph to understand per resource',
        'cognitive_diameter': 'Cognitive Diameter: Max distance between resources',
        'hub_dominance': 'Hub Dominance: Dependency on critical nodes',
        'modification_isolation': 'Modification Isolation: Independent resource modification',
        'helper_justification_ratio': 'Helper Justification: Fraction of reused helpers',
        'blast_radius_variance': 'Blast Radius Variance: Change impact predictability',
        # Best practice metric labels
        'max_nesting_depth': 'Max Nesting Depth: Deepest template nesting level',
        'unguarded_nested_access': 'Unguarded Nested Access: Unsafe property accesses',
        'array_config_count': 'Array Config Count: Array-based configurations',
        'hardcoded_image_count': 'Hardcoded Images: Non-parameterized image refs',
        'multi_resource_file_count': 'Multi-Resource Files: Files with multiple resources',
        'unquoted_string_count': 'Unquoted Strings: Potentially unsafe string values',
        'floating_image_tag_count': 'Floating Image Tags: Non-pinned image versions',
        'mutable_selector_label': 'Mutable Selector Labels: Changeable pod selectors',
        'missing_pod_selector': 'Missing Pod Selectors: Network policies without selectors'
    }

    # Extract data for each metric, filtering out None and infinity values
    metric_values = {field: [] for field in metric_fields}
    for chart_metrics in metrics_data:
        for field in metric_fields:
            value = chart_metrics.get(field)
            # Filter out None, infinity, and NaN values
            if value is not None and np.isfinite(value):
                metric_values[field].append(value)

    # Create individual histogram for each metric
    for field in metric_fields:
        values = metric_values[field]

        if len(values) == 0:
            print(f'Skipping {field} - no valid data')
            continue

        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        plt.xlabel(metric_labels[field], fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Distribution of {metric_labels[field]}', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)

        # Add statistics text
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values)
        stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd Dev: {std_val:.2f}\nn={len(values)}'
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/{field}_histogram.png', dpi=150)
        plt.close()

        print(f'Created histogram for {field}')

    # Create a combined figure with all metrics
    fig, axes = plt.subplots(4, 4, figsize=(24, 20))
    axes = axes.flatten()

    for idx, field in enumerate(metric_fields):
        values = metric_values[field]

        if len(values) == 0:
            axes[idx].text(0.5, 0.5, 'No valid data',
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'{metric_labels[field]}', fontsize=10, fontweight='bold')
        else:
            axes[idx].hist(values, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
            axes[idx].set_xlabel(metric_labels[field], fontsize=8)
            axes[idx].set_ylabel('Frequency', fontsize=10)
            axes[idx].set_title(f'{metric_labels[field]}', fontsize=9, fontweight='bold')
            axes[idx].grid(axis='y', alpha=0.3)

            # Add mean line
            mean_val = np.mean(values)
            axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            axes[idx].legend(fontsize=8)

    # Hide any unused subplots
    for idx in range(len(metric_fields), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Helm Chart Metrics Distributions (Cognitive Complexity & Best Practices)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/all_metrics_combined.png', dpi=150)
    plt.close()

    print(f'\nCreated combined histogram: {output_dir}/all_metrics_combined.png')
    print(f'\nAll plots saved to {output_dir}/ directory')

    # Print summary statistics
    print('\n' + '='*60)
    print('SUMMARY STATISTICS')
    print('='*60)
    for field in metric_fields:
        values = metric_values[field]
        print(f'\n{metric_labels[field]}:')
        if len(values) == 0:
            print('  No valid data')
        else:
            print(f'  Count:  {len(values)}')
            print(f'  Min:    {np.min(values):.2f}')
            print(f'  Max:    {np.max(values):.2f}')
            print(f'  Mean:   {np.mean(values):.2f}')
            print(f'  Median: {np.median(values):.2f}')
            print(f'  Std:    {np.std(values):.2f}')

def main():
    """Main function."""
    print('Loading metrics data...')
    metrics_data = load_metrics()
    print(f'Loaded {len(metrics_data)} chart metrics')

    print('\nGenerating histograms...')
    plot_histograms(metrics_data)

    print('\nDone!')

if __name__ == '__main__':
    main()
