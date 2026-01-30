import json
import sys
from pathlib import Path

def is_filled(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, dict):
        return any(is_filled(v) for v in value.values())
    return False

def count_fields(obj, filled_only=False):
    count = 0
    if isinstance(obj, dict):
        for value in obj.values():
            if isinstance(value, dict):
                count += count_fields(value, filled_only)
            elif isinstance(value, list):
                if not filled_only or (filled_only and len(value) > 0):
                    count += 1
            else:
                if not filled_only or (filled_only and is_filled(value)):
                    count += 1
    return count

def compute_completion_rate(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total = count_fields(data, filled_only=False)
    filled = count_fields(data, filled_only=True)
    rate = filled / total if total > 0 else 0
    
    return rate, filled, total

def compute_folder_completion(folder_path):
    folder = Path(folder_path)
    json_files = list(folder.glob("*.json"))
    
    if not json_files:
        return 0, 0, []
    
    rates = []
    for json_file in json_files:
        rate, filled, total = compute_completion_rate(json_file)
        rates.append((json_file.name, rate, filled, total))
    
    avg_rate = sum(r[1] for r in rates) / len(rates)
    return avg_rate, len(rates), rates

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_completion.py <folder1> <folder2>")
        sys.exit(1)
    
    folder1 = Path(sys.argv[1])
    folder2 = Path(sys.argv[2])
    
    avg1, count1, details1 = compute_folder_completion(folder1)
    avg2, count2, details2 = compute_folder_completion(folder2)
    
    print(f"{folder1.name}: {count1} files, average completion {avg1:.1%}")
    for name, rate, filled, total in details1:
        print(f"  {name}: {filled}/{total} ({rate:.1%})")
    print()
    
    print(f"{folder2.name}: {count2} files, average completion {avg2:.1%}")
    for name, rate, filled, total in details2:
        print(f"  {name}: {filled}/{total} ({rate:.1%})")
    print()
    
    if avg2 > avg1:
        print(f"Winner: {folder2.name} (+{avg2-avg1:.1%})")
    elif avg1 > avg2:
        print(f"Winner: {folder1.name} (+{avg1-avg2:.1%})")
    else:
        print("Tie")
