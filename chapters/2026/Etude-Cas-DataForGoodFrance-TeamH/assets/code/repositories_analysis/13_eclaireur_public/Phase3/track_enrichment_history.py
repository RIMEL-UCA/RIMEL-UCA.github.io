import csv
import sys
from pathlib import Path
from pydriller import Repository

# --- Configuration ---
ENRICHMENT_DIR = "back/scripts/enrichment"
OUTPUT_CSV = Path("results/enrichment_history.csv")

# Exact filenames to ignore
IGNORE_EXACT_NAMES = {"base_enricher.py", "__init__.py"}

# Keyword to ignore (e.g. 'utils' -> ignores 'enrichment_utils.py')
IGNORE_KEYWORD = "utils"

EXTENSIONS = {".py"}
# ---------------------

def is_ignored(filename: str) -> bool:
    """Returns True if the file should be ignored."""
    name = Path(filename).name.lower()
    
    # 1. Check exact names
    if name in IGNORE_EXACT_NAMES:
        return True
        
    # 2. Check keyword
    if IGNORE_KEYWORD in name:
        return True
        
    return False

def get_enrichment_files(repo_path: Path) -> set[str]:
    """Lists current files in the target directory."""
    target_dir = repo_path / ENRICHMENT_DIR
    if not target_dir.exists():
        print(f"Warning: Folder {target_dir} does not exist.")
        return set()

    files = set()
    for f in target_dir.iterdir():
        if f.is_file() and f.suffix in EXTENSIONS:
            if not is_ignored(f.name):
                # Normalize path to match PyDriller output
                rel_path = f.relative_to(repo_path).as_posix()
                files.add(rel_path)
    
    return files

def track_history(repo_path: str, target_files: set[str]) -> list[dict]:
    results = []
    print(f"Analyzing repository: {repo_path}")
    print(f"Target folder: {ENRICHMENT_DIR}")
    
    repo = Repository(path_to_repo=repo_path, only_modifications_with_file_types=EXTENSIONS)

    for commit in repo.traverse_commits():
        commit_date = commit.committer_date.date().isoformat()
        
        for mod in commit.modified_files:
            current_path = (mod.new_path or mod.old_path or "").replace("\\", "/")
            
            if not current_path.startswith(ENRICHMENT_DIR):
                continue

            if is_ignored(current_path):
                continue

            results.append({
                "date": commit_date,
                "commit": commit.hash,
                "author": commit.author.name,
                "file": current_path,
                "change_type": mod.change_type.name,
                "lines_added": mod.added_lines,
                "lines_removed": mod.deleted_lines,
                "msg": commit.msg.split('\n')[0]
            })
            
    return results

def save_to_csv(rows: list[dict]):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda x: x['date'])

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["date", "commit", "author", "file", "change_type", "lines_added", "lines_removed", "msg"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Done. {len(rows)} entries saved to {OUTPUT_CSV.resolve()}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python track_enrichment_history.py <path_to_repo>")
        return

    repo_path_str = sys.argv[1]
    repo_path = Path(repo_path_str)

    if not repo_path.exists():
        print(f"Error: Path {repo_path} not found.")
        return

    current_files = get_enrichment_files(repo_path)
    
    print(f"Currently tracked files ({len(current_files)}):")
    for f in sorted(current_files):
        print(f" - {f}")
    print("-" * 30)
    history = track_history(repo_path_str, current_files)
    save_to_csv(history)

if __name__ == "__main__":
    main()