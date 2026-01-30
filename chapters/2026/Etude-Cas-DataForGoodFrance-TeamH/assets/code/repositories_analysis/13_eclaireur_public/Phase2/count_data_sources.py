import yaml
import csv
import sys
from pathlib import Path
from pydriller import Repository

OUTPUT_CSV = Path("results/datasource_references_history.csv")

TARGET_FILE_TO_TRACK = "back/config.yaml"

EXTENSIONS = {".yml", ".yaml"} 

VALID_EXTENSIONS = ('.csv', '.xls', '.xlsx', '.parquet', '.json', '.xml', '.zip', '.geojson')

IGNORE_KEYS = ["save_to", "output", "log", "logging", "warehouse", "geolocator", "combined_filename"]

def looks_like_source(key: str, value: str) -> bool:
    key = key.lower()
    value = str(value).strip()

    if any(ignore in key for ignore in IGNORE_KEYS):
        return False
    
    if value.startswith(("http://", "https://", "ftp://")):
        return True

    if value.lower().endswith(VALID_EXTENSIONS):
        if "/" in value or "\\" in value or "file" in key or "input" in key:
            return True
            
    return False

def extract_sources_from_yaml(data, parent_key="") -> set[str]:
    sources = set()

    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                sources.update(extract_sources_from_yaml(v, k))
            elif isinstance(v, str):
                if looks_like_source(k, v):
                    sources.add(v.replace("file:", "").strip())

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                if looks_like_source(parent_key, item):
                    sources.add(item.replace("file:", "").strip())
            else:
                sources.update(extract_sources_from_yaml(item, parent_key))

    return sources

def find_datasource_references(repo_path: str, sources: set[str]) -> list[dict]:
    results = []
    print(f"Scanning repository at: {repo_path}")
    print(f"Tracking references ONLY in: {TARGET_FILE_TO_TRACK}")
    print(f"Tracking {len(sources)} unique data sources found in config.")

    repo = Repository(path_to_repo=repo_path, only_modifications_with_file_types=EXTENSIONS)

    for commit in repo.traverse_commits():
        commit_date = commit.committer_date.date().isoformat()
        
        for mod in commit.modified_files:
            if not mod.source_code:
                continue
            
            current_path = (mod.new_path or mod.old_path or "").replace("\\", "/")
            
            if current_path != TARGET_FILE_TO_TRACK:
                continue

            for source in sources:
                if source in mod.source_code:
                    count = mod.source_code.count(source)
                    results.append({
                        "commit": commit.hash,
                        "date": commit_date,
                        "file": current_path,
                        "source_reference": source,
                        "count": count
                    })
                    
    return results

def save_results(rows: list[dict]):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda x: x['date'])

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "commit", "file", "source_reference", "count"])
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Done. {len(rows)} references found. Saved to {OUTPUT_CSV.resolve()}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python count_data_sources.py <path_to_repo>")
        return

    repo_path = Path(sys.argv[1])

    config_path = repo_path / "back" / "config.yaml"

    if not config_path.exists():
        print(f"Config not found at: {config_path.resolve()}")
        print("Searching for alternatives...")
        if (repo_path / "config.yaml").exists():
             config_path = repo_path / "config.yaml"
        else:
            print("Error: Could not find 'back/config.yaml' or 'config.yaml' inside the repository.")
            return

    print(f"Using config file: {config_path.resolve()}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_content = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading YAML: {e}")
        return

    target_sources = extract_sources_from_yaml(yaml_content)
    
    if not target_sources:
        print("No data sources found. Check your YAML or the script filters.")
        return

    print("Identified the following sources to track:")
    for s in sorted(target_sources):
        print(f" - {s}")
    print("-" * 30)

    rows = find_datasource_references(str(repo_path), target_sources)
    save_results(rows)

if __name__ == "__main__":
    main()