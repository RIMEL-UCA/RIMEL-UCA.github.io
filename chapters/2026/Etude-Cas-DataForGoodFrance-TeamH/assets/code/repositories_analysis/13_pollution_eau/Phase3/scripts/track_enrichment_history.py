import csv
import sys
from pathlib import Path
from pydriller import Repository

# --- Configuration ---
ENRICHMENT_DIRS = ["dbt_", "analytics"]
OUTPUT_CSV = Path("../results/enrichment_history.csv")

IGNORE_EXACT_NAMES = {".gitkeep"}
IGNORE_KEYWORD = "test"

EXTENSIONS = {".sql", "ipynb"}
# ---------------------

def is_ignored(path: str) -> bool:
    """Returns True if the file should be ignored."""
    name = Path(path).name.lower()

    if name in IGNORE_EXACT_NAMES:
        return True

    if IGNORE_KEYWORD in name:
        return True

    return False

def is_in_enrichment_dir(path: str) -> bool:
    """Check if file is inside one of the enrichment directories."""
    return any(path.startswith(d) for d in ENRICHMENT_DIRS)

def has_valid_extension(path: str) -> bool:
    """Check file extension explicitly."""
    return Path(path).suffix in EXTENSIONS

def get_enrichment_files(repo_path: Path) -> set[str]:
    """Lists current files in the enrichment directories."""
    files = set()

    for enrichment_dir in ENRICHMENT_DIRS:
        target_dir = repo_path / enrichment_dir

        if not target_dir.exists():
            print(f"Warning: Folder {target_dir} does not exist.")
            continue

        for f in target_dir.rglob("*"):
            if not f.is_file():
                continue

            if not has_valid_extension(f.name):
                continue

            if is_ignored(f.name):
                continue

            rel_path = f.relative_to(repo_path).as_posix()
            files.add(rel_path)

    return files


def track_history(repo_path: str) -> list[dict]:
    results = []
    print(f"Analyzing repository: {repo_path}")
    print("Tracked folders:")
    for d in ENRICHMENT_DIRS:
        print(f" - {d}")

    repo = Repository(path_to_repo=repo_path)

    for commit in repo.traverse_commits():
        commit_date = commit.committer_date.date().isoformat()

        for mod in commit.modified_files:
            current_path = (mod.new_path or mod.old_path or "").replace("\\", "/")

            if not current_path:
                continue

            if not is_in_enrichment_dir(current_path):
                continue

            if not has_valid_extension(current_path):
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
    rows.sort(key=lambda x: x["date"])

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "date",
            "commit",
            "author",
            "file",
            "change_type",
            "lines_added",
            "lines_removed",
            "msg",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. {len(rows)} entries saved to {OUTPUT_CSV.resolve()}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python track_enrichment_history.py <path_to_repo>")
        return

    repo_path = Path(sys.argv[1])

    if not repo_path.exists():
        print(f"Error: Path {repo_path} not found.")
        return

    current_files = get_enrichment_files(repo_path)

    print(f"Currently tracked files ({len(current_files)}):")
    for f in sorted(current_files):
        print(f" - {f}")
    print("-" * 30)

    history = track_history(str(repo_path))
    save_to_csv(history)

if __name__ == "__main__":
    main()