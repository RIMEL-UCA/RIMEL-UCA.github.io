import json
import sys
from pathlib import Path
from collections import defaultdict

def flatten_dict(d, parent_key=''):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)

def is_filled(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, list):
        return len(value) > 0
    return False

def analyze_folder(folder_path):
    folder = Path(folder_path)
    json_files = list(folder.glob("*.json"))
    
    field_counts = defaultdict(lambda: {"filled": 0, "total": 0})
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        flat_data = flatten_dict(data)
        for field, value in flat_data.items():
            field_counts[field]["total"] += 1
            if is_filled(value):
                field_counts[field]["filled"] += 1
    
    results = []
    for field, counts in field_counts.items():
        rate = counts["filled"] / counts["total"] if counts["total"] > 0 else 0
        results.append((field, rate, counts["filled"], counts["total"]))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python field_completion_ratio.py <folder>")
        sys.exit(1)
    
    folder = Path(sys.argv[1])
    results = analyze_folder(folder)
    
    output_file = folder.parent / "field-completion-ratio.txt"
    
    lines = []
    lines.append(f"Field completion analysis for: {folder.name}")
    lines.append(f"{'Field':<50} {'Rate':>8} {'Filled':>8} {'Total':>8}")
    lines.append("-" * 80)
    
    for field, rate, filled, total in results:
        lines.append(f"{field:<50} {rate:>7.1%} {filled:>8} {total:>8}")
    
    content = "\n".join(lines)
    
    print(content)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\nReport saved to: {output_file}")
