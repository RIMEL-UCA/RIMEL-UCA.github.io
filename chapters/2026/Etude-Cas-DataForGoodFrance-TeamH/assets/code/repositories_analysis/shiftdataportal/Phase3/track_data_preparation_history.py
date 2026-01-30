import csv
import sys
from pathlib import Path
from pydriller import Repository

# --- Configuration pour The Shift Data Portal ---
DATA_PREPARATION_DIR = "data-preparation"
OUTPUT_CSV = Path("results/data_preparation_history.csv")

# Exact filenames to ignore
IGNORE_EXACT_NAMES = {
    "__init__.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
}

# Keywords to ignore (files containing these patterns)
IGNORE_KEYWORDS = ["utils", "test", "config", "__pycache__"]

# Extensions to track
EXTENSIONS = {".py", ".ipynb", ".sh"}

# ---------------------

def is_ignored(filename: str) -> bool:
    """Returns True if the file should be ignored."""
    name = Path(filename).name.lower()
    
    # 1. Check exact names
    if name in IGNORE_EXACT_NAMES:
        return True
        
    # 2. Check keywords
    for keyword in IGNORE_KEYWORDS:
        if keyword in name:
            return True
        
    return False


def get_data_preparation_files(repo_path: Path) -> set[str]:
    """Lists current files in the data-preparation directory."""
    target_dir = repo_path / DATA_PREPARATION_DIR
    if not target_dir.exists():
        print(f"  Warning: Folder {target_dir} does not exist.")
        return set()
    
    files = set()
    for f in target_dir.rglob("*"):  # Recursive to catch subdirectories
        if f.is_file() and f.suffix in EXTENSIONS:
            if not is_ignored(f.name):
                # Normalize path to match PyDriller output
                rel_path = f.relative_to(repo_path).as_posix()
                files.add(rel_path)
    
    return files


def track_history(repo_path: str, target_files: set[str]) -> list[dict]:
    """Track commit history for files in data-preparation directory."""
    results = []
    print(f"[+] Analyzing repository: {repo_path}")
    print(f"[+] Target folder: {DATA_PREPARATION_DIR}")
    
    repo = Repository(
        path_to_repo=repo_path,
        only_modifications_with_file_types=list(EXTENSIONS)
    )
    
    commit_count = 0
    for commit in repo.traverse_commits():
        commit_count += 1
        if commit_count % 50 == 0:
            print(f"    Processed {commit_count} commits...")
            
        commit_date = commit.committer_date.date().isoformat()
        
        for mod in commit.modified_files:
            current_path = (mod.new_path or mod.old_path or "").replace("\\", "/")
            
            # Only track files in data-preparation directory
            if not current_path.startswith(DATA_PREPARATION_DIR):
                continue
                
            if is_ignored(current_path):
                continue
            
            results.append({
                "date": commit_date,
                "commit": commit.hash[:8],  # Short hash for readability
                "author": commit.author.name,
                "file": current_path,
                "change_type": mod.change_type.name,
                "lines_added": mod.added_lines,
                "lines_removed": mod.deleted_lines,
                "msg": commit.msg.split('\n')[0][:100]  # First line, max 100 chars
            })
            
    print(f"[✓] Processed {commit_count} commits total")
    return results


def save_to_csv(rows: list[dict]):
    """Save results to CSV file."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda x: x['date'])
    
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "date", "commit", "author", "file", "change_type",
            "lines_added", "lines_removed", "msg"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"[✓] Done. {len(rows)} entries saved to {OUTPUT_CSV.resolve()}")


def print_summary(history: list[dict]):
    """Print a summary of the analysis."""
    if not history:
        print("\n No changes found in data-preparation directory")
        return
    
    files_modified = set(row['file'] for row in history)
    authors = set(row['author'] for row in history)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total changes:        {len(history)}")
    print(f"Unique files:         {len(files_modified)}")
    print(f"Contributors:         {len(authors)}")
    print(f"Date range:           {history[0]['date']} to {history[-1]['date']}")
    print("="*60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python track_data_preparation_history.py <path_to_repo>")
        print("\nExample:")
        print("  python track_data_preparation_history.py /path/to/shiftdataportal")
        return
    
    repo_path_str = sys.argv[1]
    repo_path = Path(repo_path_str).resolve()
    
    if not repo_path.exists():
        print(f" Error: Path {repo_path} not found.")
        return
    
    if not (repo_path / ".git").exists():
        print(f" Error: {repo_path} is not a git repository.")
        return
    
    print(f"\n{'='*60}")
    print(f"TRACKING DATA PREPARATION HISTORY")
    print(f"{'='*60}\n")
    
    # Get current files
    current_files = get_data_preparation_files(repo_path)
    
    print(f"[i] Currently tracked files ({len(current_files)}):")
    for f in sorted(current_files):
        print(f"    • {f}")
    print("-" * 60)
    
    # Track history
    history = track_history(repo_path_str, current_files)
    
    # Sort and save
    history.sort(key=lambda x: x['date'])
    save_to_csv(history)
    
    # Print summary
    print_summary(history)


if __name__ == "__main__":
    main()