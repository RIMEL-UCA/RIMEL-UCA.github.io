#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys
import re


class TimeoutError(Exception):
    pass


def timeout(seconds):
    from contextlib import nullcontext
    return nullcontext()


def is_filled_value(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return bool(val.strip())
    if isinstance(val, list):
        return len(val) > 0
    if isinstance(val, (int, float)):
        return True
    return False


def jaccard_similarity(list1: List, list2: List) -> float:
    set1 = set(str(item).lower().strip() for item in list1 if item)
    set2 = set(str(item).lower().strip() for item in list2 if item)
    
    union = set1.union(set2)
    if len(union) == 0:
        return 0.0
    
    intersection = set1.intersection(set2)
    return len(intersection) / len(union)


def compare_values(val1: Any, val2: Any) -> Tuple[float | None, bool]:
    filled1 = is_filled_value(val1)
    filled2 = is_filled_value(val2)
    
    if not filled1 and not filled2:
        return None, True
    
    if filled1 != filled2:
        return 0.0, True
    
    if isinstance(val1, bool) and isinstance(val2, bool):
        return float(val1 == val2), True
    
    if isinstance(val1, str) and isinstance(val2, str):
        normalized_match = val1.lower().strip() == val2.lower().strip()
        return float(normalized_match), True
    
    if isinstance(val1, list) and isinstance(val2, list):
        return jaccard_similarity(val1, val2), True
    
    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
        return float(val1 == val2), True
    
    return 0.0, False


def flatten_json(obj: Any, prefix: str = "", depth: int = 0, max_depth: int = 8) -> Dict[str, Any]:
    result = {}
    
    if depth >= max_depth:
        return {prefix: str(obj)} if prefix else {}
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (dict, list)):
                result.update(flatten_json(value, new_key, depth + 1, max_depth))
            else:
                result[new_key] = value
    
    elif isinstance(obj, list):
        result[prefix] = obj
    
    else:
        result[prefix] = obj
    
    return result


def compare_solutions(sol1: Dict, sol2: Dict, timeout_seconds: int = 5) -> float:
    try:
        with timeout(timeout_seconds):
            flat1 = flatten_json(sol1)
            flat2 = flatten_json(sol2)
        
        all_keys = set(flat1.keys()).union(set(flat2.keys()))
        
        if not all_keys:
            return 0.0
        
        scores = []
        
        for key in all_keys:
            val1 = flat1.get(key)
            val2 = flat2.get(key)
            
            similarity, is_comparable = compare_values(val1, val2)
            
            if is_comparable and similarity is not None:
                scores.append(similarity)
        
        if not scores:
            return 0.0
        
        return sum(scores) / len(scores)
    
    except TimeoutError:
        print(f"    [WARNING] Timeout during comparison")
        return 0.5


def load_solutions(directory: Path) -> Dict[str, Dict]:
    solutions = {}
    json_files = list(directory.glob("*.json"))
    
    if not json_files:
        print(f"[WARNING] No JSON files found in {directory}")
        return solutions
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                solutions[json_file.stem] = data
                print(f"[LOADED] {json_file.name}")
        except Exception as e:
            print(f"[ERROR] Failed to load {json_file.name}: {e}")
    
    return solutions


def extract_rank(name: str) -> int:
    match = re.search(r'\d+', name)
    return int(match.group()) if match else float('inf')


def sort_solution_names(names: List[str]) -> List[str]:
    return sorted(names, key=extract_rank)


def create_similarity_matrix(solutions: Dict[str, Dict]) -> Tuple[List[List[float]], List[str]]:
    names = sort_solution_names(list(solutions.keys()))
    n = len(names)
    matrix = [[0.0] * n for _ in range(n)]
    
    print("\n[COMPUTING] Similarity matrix...")
    total = (n * (n - 1)) // 2
    count = 0
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                matrix[i][j] = 1.0
            else:
                count += 1
                sim = compare_solutions(solutions[names[i]], solutions[names[j]], timeout_seconds=3)
                matrix[i][j] = sim
                matrix[j][i] = sim
                print(f"  [{count:2d}/{total}] {names[i]:15s} <-> {names[j]:15s}: {sim:.3f}")
    
    return matrix, names


def print_statistics(matrix: List[List[float]], names: List[str]) -> None:
    n = len(names)
    
    upper_triangle = []
    for i in range(n):
        for j in range(i + 1, n):
            upper_triangle.append(matrix[i][j])
    if not upper_triangle:
        print("\n[WARNING] No comparisons to analyze")
        return
    
    mean_val = sum(upper_triangle) / len(upper_triangle)
    min_val = min(upper_triangle)
    max_val = max(upper_triangle)
    
    sorted_vals = sorted(upper_triangle)
    median_val = sorted_vals[len(sorted_vals) // 2]
    
    print("\n" + "=" * 70)
    print("SIMILARITY STATISTICS")
    print("=" * 70)
    print(f"Total solutions compared:  {n}")
    print(f"Total pairwise comparisons: {len(upper_triangle)}")
    print(f"\nSimilarity Score Distribution:")
    print(f"  Mean:                    {mean_val:.3f}")
    print(f"  Median:                  {median_val:.3f}")
    print(f"  Minimum:                 {min_val:.3f}")
    print(f"  Maximum:                 {max_val:.3f}")
    
    if len(upper_triangle) > 0:
        max_idx = upper_triangle.index(max(upper_triangle))
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if count == max_idx:
                    print(f"\nMost similar pair:         {names[i]} <-> {names[j]}")
                    print(f"  Similarity score:        {max_val:.3f}")
                count += 1
        
        min_idx = upper_triangle.index(min(upper_triangle))
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if count == min_idx:
                    print(f"\nMost different pair:       {names[i]} <-> {names[j]}")
                    print(f"  Similarity score:        {min_val:.3f}")
                count += 1
    
    print("=" * 70)


def compute_group_metrics(matrix: List[List[float]], names: List[str]) -> Dict[str, Any]:
    n = len(names)
    
    # Extraire les rangs et trier les indices par rang
    ranked_items = [(i, extract_rank(names[i])) for i in range(n)]
    ranked_items.sort(key=lambda x: x[1])
    
    # Top 5 = les 5 meilleures solutions (rangs les plus bas)
    top5_indices = [idx for idx, rank in ranked_items[:min(5, n)]]
    
    # Bottom 5 = les 5 moins bonnes solutions (rangs les plus élevés)
    bottom5_indices = [idx for idx, rank in ranked_items[-min(5, n):]]
    
    metrics = {
        "total_solutions": n,
        "top5_count": len(top5_indices),
        "bottom5_count": len(bottom5_indices),
        "top5_solutions": [names[i] for i in top5_indices],
        "bottom5_solutions": [names[i] for i in bottom5_indices]
    }
    
    if len(top5_indices) >= 2:
        top5_scores = []
        for i in top5_indices:
            for j in top5_indices:
                if i < j:
                    top5_scores.append(matrix[i][j])
        
        if top5_scores:
            metrics["top5_intra_mean"] = sum(top5_scores) / len(top5_scores)
            metrics["top5_intra_std"] = (sum((x - metrics["top5_intra_mean"])**2 for x in top5_scores) / len(top5_scores))**0.5
            metrics["top5_intra_min"] = min(top5_scores)
            metrics["top5_intra_max"] = max(top5_scores)
            metrics["top5_intra_count"] = len(top5_scores)
    
    if len(bottom5_indices) >= 2:
        bottom5_scores = []
        for i in bottom5_indices:
            for j in bottom5_indices:
                if i < j:
                    bottom5_scores.append(matrix[i][j])
        
        if bottom5_scores:
            metrics["bottom5_intra_mean"] = sum(bottom5_scores) / len(bottom5_scores)
            metrics["bottom5_intra_std"] = (sum((x - metrics["bottom5_intra_mean"])**2 for x in bottom5_scores) / len(bottom5_scores))**0.5
            metrics["bottom5_intra_min"] = min(bottom5_scores)
            metrics["bottom5_intra_max"] = max(bottom5_scores)
            metrics["bottom5_intra_count"] = len(bottom5_scores)
    
    if len(top5_indices) >= 1 and len(bottom5_indices) >= 1:
        inter_scores = []
        for i in top5_indices:
            for j in bottom5_indices:
                inter_scores.append(matrix[i][j])
        
        if inter_scores:
            metrics["inter_group_mean"] = sum(inter_scores) / len(inter_scores)
            metrics["inter_group_std"] = (sum((x - metrics["inter_group_mean"])**2 for x in inter_scores) / len(inter_scores))**0.5
            metrics["inter_group_min"] = min(inter_scores)
            metrics["inter_group_max"] = max(inter_scores)
            metrics["inter_group_count"] = len(inter_scores)
    
    if "top5_intra_mean" in metrics and "bottom5_intra_mean" in metrics:
        metrics["difference_top5_vs_bottom5"] = metrics["top5_intra_mean"] - metrics["bottom5_intra_mean"]
        metrics["ratio_top5_vs_bottom5"] = metrics["top5_intra_mean"] / metrics["bottom5_intra_mean"] if metrics["bottom5_intra_mean"] > 0 else None
    
    if "top5_intra_mean" in metrics and "inter_group_mean" in metrics:
        metrics["difference_top5_vs_inter"] = metrics["top5_intra_mean"] - metrics["inter_group_mean"]
    
    return metrics


def save_metrics_report(metrics: Dict[str, Any], output_path: Path):
    report_lines = [
        "=" * 80,
        "ARCHITECTURAL CONVERGENCE ANALYSIS - GROUP METRICS REPORT",
        "=" * 80,
        "",
        f"Total solutions analyzed: {metrics['total_solutions']}",
        f"Top 5 solutions (best): {metrics['top5_count']} ({', '.join(metrics.get('top5_solutions', []))})",
        f"Bottom 5 solutions (worst): {metrics['bottom5_count']} ({', '.join(metrics.get('bottom5_solutions', []))})",
        "",
        "=" * 80,
        "INTRA-GROUP SIMILARITY (Top 5 - Best Solutions)",
        "=" * 80,
    ]
    
    if "top5_intra_mean" in metrics:
        report_lines.extend([
            f"Mean similarity:     {metrics['top5_intra_mean']:.4f}",
            f"Standard deviation:  {metrics['top5_intra_std']:.4f}",
            f"Minimum similarity:  {metrics['top5_intra_min']:.4f}",
            f"Maximum similarity:  {metrics['top5_intra_max']:.4f}",
            f"Number of pairs:     {metrics['top5_intra_count']}",
        ])
    else:
        report_lines.append("Not enough Top 5 solutions for analysis (need at least 2)")
    
    report_lines.extend([
        "",
        "=" * 80,
        "INTRA-GROUP SIMILARITY (Bottom 5 - Worst Solutions)",
        "=" * 80,
    ])
    
    if "bottom5_intra_mean" in metrics:
        report_lines.extend([
            f"Mean similarity:     {metrics['bottom5_intra_mean']:.4f}",
            f"Standard deviation:  {metrics['bottom5_intra_std']:.4f}",
            f"Minimum similarity:  {metrics['bottom5_intra_min']:.4f}",
            f"Maximum similarity:  {metrics['bottom5_intra_max']:.4f}",
            f"Number of pairs:     {metrics['bottom5_intra_count']}",
        ])
    else:
        report_lines.append("Not enough Bottom 5 solutions for analysis (need at least 2)")
    
    report_lines.extend([
        "",
        "=" * 80,
        "INTER-GROUP SIMILARITY (Top 5 vs Bottom 5)",
        "=" * 80,
    ])
    
    if "inter_group_mean" in metrics:
        report_lines.extend([
            f"Mean similarity:     {metrics['inter_group_mean']:.4f}",
            f"Standard deviation:  {metrics['inter_group_std']:.4f}",
            f"Minimum similarity:  {metrics['inter_group_min']:.4f}",
            f"Maximum similarity:  {metrics['inter_group_max']:.4f}",
            f"Number of pairs:     {metrics['inter_group_count']}",
        ])
    else:
        report_lines.append("Cannot compute inter-group metrics (need both Top 5 and Bottom 5 solutions)")
    
    report_lines.extend([
        "",
        "=" * 80,
        "COMPARATIVE ANALYSIS",
        "=" * 80,
    ])
    
    if "difference_top5_vs_bottom5" in metrics:
        diff = metrics['difference_top5_vs_bottom5']
        ratio = metrics.get('ratio_top5_vs_bottom5')
        
        report_lines.extend([
            f"Top 5 intra-group mean:      {metrics['top5_intra_mean']:.4f}",
            f"Bottom 5 intra-group mean:   {metrics['bottom5_intra_mean']:.4f}",
            f"Difference (Top5 - Bottom5): {diff:.4f}",
        ])
        
        if ratio is not None:
            report_lines.append(f"Ratio (Top5 / Bottom5):      {ratio:.4f}x")
    
    if "difference_top5_vs_inter" in metrics:
        diff_inter = metrics['difference_top5_vs_inter']
        report_lines.extend([
            "",
            f"Top 5 intra-group mean:      {metrics['top5_intra_mean']:.4f}",
            f"Inter-group mean:            {metrics['inter_group_mean']:.4f}",
            f"Difference (Intra - Inter):  {diff_inter:.4f}",
        ])
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"[METRICS] Report saved to: {output_path}")


def print_group_statistics(matrix: List[List[float]], names: List[str]) -> None:
    n = len(names)
    group_size = min(5, n)
    
    if n < 3:
        return
    
    first_group = list(range(group_size))
    last_group = list(range(max(group_size, n - group_size), n))
    
    first_scores = []
    for i in first_group:
        for j in first_group:
            if i < j:
                first_scores.append(matrix[i][j])
    
    last_scores = []
    for i in last_group:
        for j in last_group:
            if i < j:
                last_scores.append(matrix[i][j])
    
    cross_scores = []
    for i in first_group:
        for j in last_group:
            cross_scores.append(matrix[i][j])
    
    print("\n" + "=" * 70)
    print("GROUP SIMILARITY ANALYSIS")
    print("=" * 70)
    
    if first_scores:
        first_mean = sum(first_scores) / len(first_scores)
        print(f"First {group_size} solutions (mean):        {first_mean:.3f}")
    
    if last_scores:
        last_mean = sum(last_scores) / len(last_scores)
        print(f"Last {group_size} solutions (mean):         {last_mean:.3f}")
    
    if cross_scores:
        cross_mean = sum(cross_scores) / len(cross_scores)
        print(f"First {group_size} vs Last {group_size} (mean):      {cross_mean:.3f}")
    
    print("=" * 70)


def print_matrix(matrix: List[List[float]], names: List[str]):
    n = len(names)
    
    print("\n" + "=" * 60)
    print("SIMILARITY MATRIX")
    print("=" * 60)
    
    header = "Solution".ljust(15) + " ".join(f"{name:8}" for name in names)
    print(header)
    print("-" * len(header))
    
    for i, name in enumerate(names):
        row = name.ljust(15) + " ".join(f"{matrix[i][j]:.2f}".ljust(8) for j in range(n))
        print(row)
    
    print("=" * 60)


def save_csv_matrix(matrix: List[List[float]], names: List[str], output_path: Path):
    n = len(names)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("," + ",".join(names) + "\n")
        
        for i, name in enumerate(names):
            f.write(name + "," + ",".join(f"{matrix[i][j]:.4f}" for j in range(n)) + "\n")
    
    print(f"[CSV] Matrix saved to: {output_path}")


def plot_heatmap(matrix: List[List[float]], names: List[str], output_path: Path):
    n = len(names)
    cell_size = 60
    margin = 150
    total_width = margin + (n * cell_size) + 20
    total_height = margin + (n * cell_size) + 20
    
    def score_to_color(score):
        if score < 0.8:
            r = 220
            g = int(120 + 135 * (score / 0.8))
            b = 100
        else:
            r = int(220 - 100 * ((score - 0.8) / 0.2))
            g = 230
            b = int(100 + 120 * ((score - 0.8) / 0.2))
        return f"#{r:02x}{g:02x}{b:02x}"
    
    svg_lines = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg width="{total_width}" height="{total_height}" xmlns="http://www.w3.org/2000/svg">',
        f'<style>',
        f'  .label {{ font-size: 11px; font-family: Arial; }}',
        f'  .cell-text {{ font-size: 10px; font-family: monospace; font-weight: bold; text-anchor: middle; dominant-baseline: middle; }}',
        f'  .title {{ font-size: 16px; font-family: Arial; font-weight: bold; }}',
        f'</style>',
        f'<rect width="{total_width}" height="{total_height}" fill="white"/>',
        f'<text x="{total_width // 2}" y="30" class="title" text-anchor="middle">Solutions Similarity Heatmap</text>',
        f'<text x="{total_width // 2}" y="50" style="font-size: 11px; font-family: Arial; text-anchor: middle;">(0.0 = different, 1.0 = identical)</text>',
    ]
    
    for i in range(n):
        for j in range(n):
            score = matrix[i][j]
            x = margin + j * cell_size
            y = margin + i * cell_size
            color = score_to_color(score)
            
            svg_lines.append(
                f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" '
                f'fill="{color}" stroke="white" stroke-width="0.5"/>'
            )
            
            text_color = "white" if score < 0.5 else "black"
            svg_lines.append(
                f'<text x="{x + cell_size // 2}" y="{y + cell_size // 2}" '
                f'class="cell-text" fill="{text_color}">{score:.2f}</text>'
            )
    
    for j, name in enumerate(names):
        x = margin + j * cell_size + cell_size // 2
        y = margin - 20
        svg_lines.append(
            f'<text x="{x}" y="{y}" class="label" text-anchor="middle" '
            f'transform="rotate(-45 {x} {y})">{name}</text>'
        )
    
    for i, name in enumerate(names):
        x = margin - 20
        y = margin + i * cell_size + cell_size // 2
        svg_lines.append(
            f'<text x="{x}" y="{y}" class="label" text-anchor="end" '
            f'dominant-baseline="middle">{name}</text>'
        )
    
    svg_lines.append('</svg>')
    
    svg_content = '\n'.join(svg_lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    print(f"[SVG] Heatmap saved: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_solutions.py <directory> [output_csv]")
        print("\nExample:")
        print("  python compare_solutions.py ./normalized/")
        print("  python compare_solutions.py ./normalized/ results.csv")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    csv_output = Path(sys.argv[2]) if len(sys.argv) >= 3 else None
    
    if not input_dir.exists():
        print(f"[ERROR] Directory not found: {input_dir}")
        sys.exit(1)
    
    print(f"\n[LOADING] Solutions from: {input_dir}")
    solutions = load_solutions(input_dir)
    
    if len(solutions) < 2:
        print("[ERROR] Need at least 2 solutions to compare")
        sys.exit(1)
    
    print(f"\n[SUCCESS] Loaded {len(solutions)} solutions")
    
    matrix, names = create_similarity_matrix(solutions)
    print_statistics(matrix, names)
    print_group_statistics(matrix, names)
    print_matrix(matrix, names)
    
    if csv_output:
        print(f"\n[SAVING] CSV matrix...")
        save_csv_matrix(matrix, names, csv_output)
    
    svg_output = input_dir / "similarity_heatmap.svg"
    print("\n[GENERATING] Heatmap...")
    plot_heatmap(matrix, names, svg_output)
    
    metrics_output = input_dir / "metrics_report.txt"
    print("\n[COMPUTING] Group metrics...")
    metrics = compute_group_metrics(matrix, names)
    save_metrics_report(metrics, metrics_output)
    
    print("\n[COMPLETE] Analysis finished successfully.\n")


if __name__ == "__main__":
    main()
