#!/usr/bin/env python3
"""
Fresh Mining Script for Hypothesis Validation
Improved mining with better chart filtering and complete commit history.
"""

import os
import json
import subprocess
import sys
import csv
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'part2_evolution_mining'))
from best_practices import BestPracticesAnalyzer


def run_git_command(repo_path: str, command: List[str]) -> str:
    """Execute a git command in the repository."""
    try:
        result = subprocess.run(
            ['git', '-C', repo_path] + command,
            capture_output=True,
            text=True,
            check=True,
            errors='replace'
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def is_valid_chart(chart_path: str) -> bool:
    """Check if this is a valid Helm chart (minimal filtering)."""
    # Only exclude obvious non-chart paths
    exclude_patterns = [
        r'/testdata/',  # Go test data
        r'_test\.yaml$',  # Test files
    ]
    
    path_lower = chart_path.lower()
    for pattern in exclude_patterns:
        if re.search(pattern, path_lower):
            return False
    
    return True


def get_commits_for_chart(repo_path: str, chart_root: str) -> List[Dict]:
    """Get all commits that touched the chart directory."""
    result = run_git_command(
        repo_path,
        ['log', '--format=%H|%ai|%an', '--follow', '--', chart_root]
    )
    
    if not result:
        return []
    
    commits = []
    seen_hashes = set()
    
    for line in result.split('\n'):
        if not line.strip():
            continue
        parts = line.split('|')
        if len(parts) >= 3:
            h = parts[0]
            if h not in seen_hashes:
                seen_hashes.add(h)
                commits.append({
                    'hash': h,
                    'date': parts[1],
                    'author': parts[2]
                })
    
    return list(reversed(commits))


def get_chart_files_at_commit(repo_path: str, commit_hash: str, chart_root: str) -> Dict[str, str]:
    """Get all chart files and their contents at a specific commit."""
    file_list = run_git_command(
        repo_path,
        ['ls-tree', '-r', '--name-only', commit_hash, chart_root]
    )
    
    if not file_list:
        return {}
    
    files = {}
    binary_exts = {'.gz', '.tgz', '.tar', '.zip', '.png', '.jpg', '.gif', '.ico', '.woff', '.woff2', '.ttf'}
    
    for file_path in file_list.split('\n'):
        if not file_path.strip():
            continue
        
        # Get relative path from chart root
        rel_path = file_path[len(chart_root):].lstrip('/')
        
        # Skip binary files
        if any(file_path.endswith(ext) for ext in binary_exts):
            continue
        
        content = run_git_command(repo_path, ['show', f'{commit_hash}:{file_path}'])
        if content is not None:
            files[rel_path] = content
    
    return files


def calculate_metrics(files: Dict[str, str], analyzer: BestPracticesAnalyzer) -> Dict:
    """Calculate all metrics for a chart state."""
    template_files = {k: v for k, v in files.items() 
                      if k.startswith('templates/') and (k.endswith('.yaml') or k.endswith('.tpl'))}
    
    # Basic template metrics
    total_lines = 0
    control_structures = 0
    includes = 0
    values_refs = 0
    defines = 0
    crd_count = 0
    
    for filename, content in template_files.items():
        lines = content.split('\n')
        total_lines += len(lines)
        
        # Control structures
        control_structures += len(re.findall(r'{{\s*-?\s*(if|range|with)\s', content))
        
        # Includes and template calls
        includes += len(re.findall(r'{{\s*-?\s*(include|template|tpl)\s', content))
        
        # Values references
        values_refs += len(re.findall(r'\.Values\.', content))
        
        # Template definitions (in _helpers.tpl)
        defines += len(re.findall(r'{{\s*-?\s*define\s+"', content))
        
        # CRDs
        if re.search(r'kind:\s*CustomResourceDefinition', content):
            crd_count += 1
    
    # Check for _helpers.tpl
    has_helpers = any('_helpers.tpl' in f for f in files.keys())
    
    # Values nesting depth
    values_yaml = files.get('values.yaml', '')
    values_nesting = calculate_nesting_depth(values_yaml)
    
    # Chart version
    chart_yaml = files.get('Chart.yaml', '')
    version_match = re.search(r'version:\s*["\']?([^"\'\n]+)', chart_yaml)
    chart_version = version_match.group(1).strip() if version_match else 'unknown'
    
    # Dependencies count
    deps_count = count_dependencies(chart_yaml)
    
    # Best practices analysis - extract scores only
    bp_results = analyzer.analyze_chart(files)
    
    return {
        'chart_version': chart_version,
        'dependencies_count': deps_count,
        'template_files': len(template_files),
        'template_lines': total_lines,
        'control_structures': control_structures,
        'includes': includes,
        'values_references': values_refs,
        'template_definitions': defines,
        'crd_count': crd_count,
        'avg_complexity': (control_structures + includes) / max(len(template_files), 1),
        'has_helpers': has_helpers,
        'values_nesting': values_nesting,
        # Extract just the scores from each category
        'templates_structure_score': bp_results['templates_structure']['score'],
        'labels_compliance_score': bp_results['labels_compliance']['score'],
        'values_naming_score': bp_results['values_naming']['score'],
        'documentation_score': bp_results['documentation']['score'],
        'advanced_features_score': bp_results['advanced_features']['score'],
        'overall_score': bp_results['overall_score'],
        # Also extract some useful boolean/count metrics
        'has_readme': bp_results['documentation'].get('has_readme', False),
        'has_notes': bp_results['documentation'].get('has_notes', False),
        'has_schema': bp_results['documentation'].get('has_schema', False),
        'has_tests': bp_results['advanced_features'].get('has_tests', False),
    }


def calculate_nesting_depth(yaml_content: str) -> int:
    """Calculate max nesting depth in values.yaml."""
    if not yaml_content:
        return 0
    
    max_depth = 0
    current_depth = 0
    
    for line in yaml_content.split('\n'):
        if not line.strip() or line.strip().startswith('#'):
            continue
        
        # Count leading spaces
        stripped = line.lstrip()
        if stripped:
            indent = len(line) - len(stripped)
            depth = indent // 2  # Assuming 2-space indent
            max_depth = max(max_depth, depth)
    
    return max_depth


def count_dependencies(chart_yaml: str) -> int:
    """Count dependencies in Chart.yaml."""
    if not chart_yaml:
        return 0
    
    count = 0
    in_deps = False
    
    for line in chart_yaml.split('\n'):
        if line.strip().startswith('dependencies:'):
            in_deps = True
        elif in_deps:
            if re.match(r'^\s*-\s*name:', line):
                count += 1
            elif line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                break
    
    return count


def discover_charts_in_repo(repo_path: str) -> List[Dict]:
    """Discover all Helm charts in a repository."""
    charts = []
    
    # Find all Chart.yaml files
    result = run_git_command(repo_path, ['ls-files', '*/Chart.yaml', 'Chart.yaml'])
    
    for chart_file in result.split('\n'):
        if not chart_file.strip():
            continue
        
        chart_root = str(Path(chart_file).parent)
        if chart_root == '.':
            chart_root = ''
        
        # Minimal filtering - only exclude obvious testdata
        if not is_valid_chart(chart_root):
            continue
        
        # Get chart name from Chart.yaml
        content = run_git_command(repo_path, ['show', f'HEAD:{chart_file}'])
        name_match = re.search(r'name:\s*([^\n]+)', content)
        name = name_match.group(1).strip() if name_match else Path(chart_root).name
        
        charts.append({
            'name': name,
            'chart_root': chart_root
        })
    
    return charts


def mine_repository(repo_name: str, repo_path: str, output_dir: Path) -> int:
    """Mine all charts in a repository."""
    print(f"\n{'='*60}")
    print(f"Mining repository: {repo_name}")
    print(f"{'='*60}")
    
    # Discover charts
    charts = discover_charts_in_repo(repo_path)
    print(f"Found {len(charts)} charts")
    
    for chart in charts:
        print(f"  - {chart['name']} ({chart['chart_root']})")
    
    analyzer = BestPracticesAnalyzer()
    all_metrics = []
    
    for chart in charts:
        chart_root = chart['chart_root']
        chart_name = chart['name']
        
        print(f"\n  Mining: {chart_name}")
        
        # Get commits
        commits = get_commits_for_chart(repo_path, chart_root)
        if not commits:
            print(f"    No commits found")
            continue
        
        print(f"    Found {len(commits)} commits")
        
        # Apply sampling ONLY for charts with many commits (>100)
        SAMPLING_THRESHOLD = 100
        if len(commits) > SAMPLING_THRESHOLD:
            # Sample every Nth commit to get ~50-100 data points
            rate = max(2, len(commits) // 50)
            sampled = commits[::rate]
            # Always include first and last commit
            if commits[0] not in sampled:
                sampled.insert(0, commits[0])
            if commits[-1] not in sampled:
                sampled.append(commits[-1])
            commits = sorted(sampled, key=lambda c: c['date'])
            print(f"    Sampled {len(commits)} commits (from {len(get_commits_for_chart(repo_path, chart_root))})")
        
        # Mine each commit
        for i, commit in enumerate(commits):
            print(f"    Processing {i+1}/{len(commits)}: {commit['hash'][:8]}", end='\r')
            
            files = get_chart_files_at_commit(repo_path, commit['hash'], chart_root)
            if not files:
                continue
            
            try:
                metrics = calculate_metrics(files, analyzer)
                metrics['repo'] = repo_name
                metrics['chart_name'] = chart_name
                metrics['chart_root'] = chart_root
                metrics['commit_hash'] = commit['hash']
                metrics['commit_date'] = commit['date']
                metrics['commit_author'] = commit['author']
                
                all_metrics.append(metrics)
            except Exception as e:
                print(f"\n    Error processing {commit['hash'][:8]}: {e}")
        
        print(f"    Collected {sum(1 for m in all_metrics if m['chart_root'] == chart_root)} data points")
    
    # Save metrics
    if all_metrics:
        csv_path = output_dir / f'{repo_name}_metrics.csv'
        fieldnames = list(all_metrics[0].keys())
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)
        
        print(f"\n  Saved {len(all_metrics)} data points to {csv_path}")
    
    return len(all_metrics)


def main():
    """Main mining function."""
    print("=" * 70)
    print("FRESH MINING FOR HYPOTHESIS VALIDATION")
    print("=" * 70)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent / 'part2_evolution_mining'
    repos_dir = base_dir / 'repos'
    output_dir = Path(__file__).parent / 'mined_data'
    output_dir.mkdir(exist_ok=True)
    
    print(f"Sampling: only for charts with >100 commits")
    
    # Find all cloned repos
    if not repos_dir.exists():
        print(f"Repos directory not found: {repos_dir}")
        sys.exit(1)
    
    repos = [d for d in repos_dir.iterdir() if d.is_dir() and (d / '.git').exists()]
    print(f"\nFound {len(repos)} repositories")
    
    total_points = 0
    
    for repo_dir in sorted(repos):
        repo_name = repo_dir.name
        points = mine_repository(repo_name, str(repo_dir), output_dir)
        total_points += points
    
    # Combine all CSVs
    print("\n" + "=" * 70)
    print("COMBINING RESULTS")
    print("=" * 70)
    
    import pandas as pd
    
    all_dfs = []
    for csv_file in output_dir.glob('*_metrics.csv'):
        df = pd.read_csv(csv_file)
        all_dfs.append(df)
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = output_dir / 'all_metrics.csv'
        combined.to_csv(combined_path, index=False)
        
        print(f"Combined {len(all_dfs)} files")
        print(f"Total rows: {len(combined)}")
        print(f"Unique charts: {combined['chart_root'].nunique()}")
        print(f"Saved to: {combined_path}")
    
    print(f"\n✅ Mining complete! Total data points: {total_points}")


if __name__ == '__main__':
    main()
