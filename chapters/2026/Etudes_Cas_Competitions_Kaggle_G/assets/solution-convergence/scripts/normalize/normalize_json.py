import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict


def load_schema(schema_path: str) -> Dict:
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Schema file not found: {schema_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in schema file: {e}")
        sys.exit(1)


def is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        cleaned = value.strip().lower()
        return cleaned in ["", "null", "none", "n/a", "na", "-", "0"]
    if isinstance(value, (list, dict)):
        return len(value) == 0
    if isinstance(value, (int, float)):
        return value == 0
    return False


def normalize_string(text: str, remove_separators: bool = False) -> Tuple[str, Dict[str, int]]:
    if not isinstance(text, str):
        return text, {}
    
    stats = defaultdict(int)
    original = text
    text = text.strip()
    
    if text != original:
        stats['whitespace_trimmed'] += 1
    
    original = text
    text = text.lower()
    if text != original:
        stats['lowercased'] += 1
    
    if remove_separators:
        original = text
        text = re.sub(r'[\s\-_—–\'\"\'\'""]+', '', text)
        if text != original:
            stats['separators_removed'] += 1
    else:
        original = text
        text = re.sub(r'[\s\-_—–\'\"\'\'""]+', '-', text)
        if text != original:
            stats['separators_normalized'] += 1
        text = text.strip('-')
        text = re.sub(r'-+', '-', text)
    
    return text, stats


def normalize_list(lst: List[Any]) -> Tuple[List[Any], Dict[str, int]]:
    normalized = []
    seen = set()
    stats = defaultdict(int)
    
    for item in lst:
        if is_empty_value(item):
            stats['empty_values_removed'] += 1
            continue
        if isinstance(item, str):
            item_normalized, item_stats = normalize_string(item)
            for key, count in item_stats.items():
                stats[key] += count
            
            if not item_normalized:
                stats['empty_values_removed'] += 1
                continue
            item_key, _ = normalize_string(item, remove_separators=True)
            if item_key not in seen:
                seen.add(item_key)
                normalized.append(item_normalized)
            else:
                stats['duplicates_removed'] += 1
        else:
            if item not in seen:
                seen.add(item)
                normalized.append(item)
            else:
                stats['duplicates_removed'] += 1
    
    return normalized, stats


def get_expected_type(reference_value: Any) -> type:
    if reference_value is None:
        return type(None)
    return type(reference_value)


def normalize_value(value: Any, reference_value: Any, key: str = "") -> Tuple[Any, Dict[str, int]]:
    expected_type = get_expected_type(reference_value)
    stats = defaultdict(int)
    
    if isinstance(value, str) and is_empty_value(value):
        value = None
        stats['empty_string_to_null'] += 1
    
    if expected_type == bool:
        if value is None:
            stats['null_to_false'] += 1
            return False, stats
        if isinstance(value, str):
            value_lower = value.strip().lower()
            if value_lower in ["true", "1", "yes"]:
                stats['string_to_bool'] += 1
                return True, stats
            elif value_lower in ["false", "0", "no"]:
                stats['string_to_bool'] += 1
                return False, stats
        return bool(value), stats
    
    if expected_type in [int, float, type(None)]:
        if value is None:
            return None, stats
        if isinstance(value, (int, float)):
            return value, stats
        if isinstance(value, str):
            value_stripped = value.strip()
            try:
                if '.' not in value_stripped:
                    stats['string_to_int'] += 1
                    return int(value_stripped), stats
                stats['string_to_float'] += 1
                return float(value_stripped), stats
            except ValueError:
                pass
        return None, stats
    
    if expected_type == str:
        if value is None:
            return "", stats
        normalized_str, str_stats = normalize_string(str(value))
        for k, v in str_stats.items():
            stats[k] += v
        return normalized_str, stats
    
    if expected_type == list:
        if value is None:
            return [], stats
        if not isinstance(value, list):
            return [], stats
        normalized_list, list_stats = normalize_list(value)
        for k, v in list_stats.items():
            stats[k] += v
        return normalized_list, stats
    
    if expected_type == dict:
        if value is None:
            return reference_value.copy(), stats
        if not isinstance(value, dict):
            return reference_value.copy(), stats
        normalized_dict, dict_stats = normalize_dict(value, reference_value)
        for k, v in dict_stats.items():
            stats[k] += v
        return normalized_dict, stats
    
    return value, stats


def normalize_dict(data: Dict, reference: Dict) -> Tuple[Dict, Dict[str, int]]:
    normalized = {}
    stats = defaultdict(int)
    
    for key, ref_value in reference.items():
        if key in data:
            normalized[key], value_stats = normalize_value(data[key], ref_value, key)
            for k, v in value_stats.items():
                stats[k] += v
        else:
            stats['missing_fields_added'] += 1
            if isinstance(ref_value, dict):
                normalized[key], dict_stats = normalize_dict({}, ref_value)
                for k, v in dict_stats.items():
                    stats[k] += v
            else:
                normalized[key], value_stats = normalize_value(None, ref_value, key)
                for k, v in value_stats.items():
                    stats[k] += v
    
    return normalized, stats


def validate_no_extra_fields(data: Dict, reference: Dict, path: str = "") -> List[str]:
    extra_fields = []
    for key in data.keys():
        current_path = f"{path}.{key}" if path else key
        if key not in reference:
            extra_fields.append(current_path)
        elif isinstance(data[key], dict) and isinstance(reference[key], dict):
            extra_fields.extend(validate_no_extra_fields(data[key], reference[key], current_path))
    return extra_fields


def process_json_file(input_path: Path, output_path: Path, schema: Dict) -> Dict[str, Any]:
    stats = {"success": False, "extra_fields": [], "error": None, "modifications": defaultdict(int)}
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        extra_fields = validate_no_extra_fields(data, schema)
        if extra_fields:
            stats["extra_fields"] = extra_fields
            stats["modifications"]["extra_fields_detected"] = len(extra_fields)
            print(f"[WARNING] {input_path.name}: Extra fields: {', '.join(extra_fields)}")
        normalized_data, norm_stats = normalize_dict(data, schema)
        stats["modifications"] = dict(norm_stats)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(normalized_data, f, indent=2, ensure_ascii=False)
        stats["success"] = True
        print(f"[DONE] {input_path.name}")
    except json.JSONDecodeError as e:
        stats["error"] = f"JSON error: {e}"
        print(f"[ERROR] {input_path.name}: {stats['error']}")
    except Exception as e:
        stats["error"] = str(e)
        print(f"[ERROR] {input_path.name}: {stats['error']}")
    return stats


def process_directory(input_dir: str, output_dir: str, schema_path: str):
    schema = load_schema(schema_path)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if not input_path.exists():
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        return
    json_files = list(input_path.glob("*.json"))
    if not json_files:
        print(f"[WARNING] No JSON files found in {input_dir}")
        return
    print(f"\n[LOADING] Processing {len(json_files)} files")
    print("=" * 60)
    total = len(json_files)
    success_count = 0
    error_count = 0
    extra_fields_count = 0
    error_report = []
    total_modifications = defaultdict(int)
    
    for json_file in json_files:
        output_file = output_path / json_file.name
        stats = process_json_file(json_file, output_file, schema)
        if stats["success"]:
            success_count += 1
        else:
            error_count += 1
            error_report.append({
                "file": json_file.name,
                "type": "error",
                "message": stats["error"]
            })
        if stats["extra_fields"]:
            extra_fields_count += 1
            error_report.append({
                "file": json_file.name,
                "type": "warning",
                "message": f"Extra fields: {', '.join(stats['extra_fields'])}"
            })
        
        # Aggregate modifications
        for mod_type, count in stats["modifications"].items():
            total_modifications[mod_type] += count
    
    print("=" * 60)
    print(f"\n[SUMMARY]")
    print(f"  Total: {total}")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Extra fields: {extra_fields_count}")
    print(f"\n[OUTPUT] {output_dir}\n")
    
    # Write error report
    report_path = output_path / "normalization_report.txt"
    output_path.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("NORMALIZATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total files processed: {total}\n")
        f.write(f"Successful: {success_count}\n")
        f.write(f"Errors: {error_count}\n")
        f.write(f"Files with extra fields: {extra_fields_count}\n\n")
        
        f.write("MODIFICATIONS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        if total_modifications:
            for mod_type in sorted(total_modifications.keys()):
                count = total_modifications[mod_type]
                f.write(f"  {mod_type:.<40} {count:>6}\n")
        else:
            f.write("  No modifications performed.\n")
        
        if error_report:
            f.write("\n" + "=" * 60 + "\n\n")
            f.write("ERRORS AND WARNINGS\n")
            f.write("=" * 60 + "\n\n")
            
            for item in error_report:
                f.write(f"[{item['type'].upper()}] {item['file']}\n")
                f.write(f"  {item['message']}\n\n")
    
    print(f"[REPORT] Report written to: {report_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python normalize_json.py <schema_file> <input_directory> <output_directory>")
        sys.exit(1)
    schema_file = sys.argv[1]
    input_directory = sys.argv[2]
    output_directory = sys.argv[3]
    process_directory(input_directory, output_directory, schema_file)
