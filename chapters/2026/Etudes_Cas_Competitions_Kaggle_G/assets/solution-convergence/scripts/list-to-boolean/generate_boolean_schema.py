import json
import sys
from pathlib import Path

def collect_list_values(data, path=""):
    list_values = {}
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            if isinstance(value, list):
                if current_path not in list_values:
                    list_values[current_path] = set()
                for item in value:
                    if item is not None and item != "" and item is not False:
                        list_values[current_path].add(str(item))
            elif isinstance(value, dict):
                nested_values = collect_list_values(value, current_path)
                for nested_path, nested_set in nested_values.items():
                    if nested_path not in list_values:
                        list_values[nested_path] = set()
                    list_values[nested_path].update(nested_set)
    return list_values

def transform_to_boolean_schema(template, all_list_values, path=""):
    if isinstance(template, dict):
        result = {}
        for key, value in template.items():
            current_path = f"{path}.{key}" if path else key
            if isinstance(value, list):
                boolean_obj = {}
                if current_path in all_list_values:
                    for item in sorted(all_list_values[current_path]):
                        boolean_obj[item] = False
                result[key] = boolean_obj
            elif isinstance(value, dict):
                result[key] = transform_to_boolean_schema(value, all_list_values, current_path)
            elif isinstance(value, (int, float)):
                result[key] = None
            else:
                result[key] = value
        return result
    return template

input_folder = sys.argv[1]
output_file = sys.argv[2] if len(sys.argv) > 2 else "boolean_schema.json"

all_list_values = {}
template_structure = None

for json_file in Path(input_folder).glob("*.json"):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if template_structure is None:
            template_structure = data
        file_list_values = collect_list_values(data)
        for path, values in file_list_values.items():
            if path not in all_list_values:
                all_list_values[path] = set()
            all_list_values[path].update(values)

boolean_schema = transform_to_boolean_schema(template_structure, all_list_values)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(boolean_schema, f, indent=2, ensure_ascii=False)
