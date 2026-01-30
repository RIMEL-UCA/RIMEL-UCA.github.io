import csv
import sys
from pathlib import Path
from pydriller import Repository

# =========================
# Configuration
# =========================

BASE_DATA_DIR = "database/data"

EXCLUDED_DIRS = {
    "database/data/old_db_init",
    "database/data/sample",
    "database/data/models",
}

OUTPUT_CSV = Path("results/data_processing_history.csv")

# =========================
# Classification
# =========================

def classify_activity(path: str) -> str:
    path = path.replace("\\", "/").lower()

    if path.endswith(".ipynb"):
        return "exploration"

    if "machine_learning_predictions" in path:
        return "ml_integration"

    if any(k in path for k in ["scraper", "scraping"]):
        return "scraping"

    if any(k in path for k in ["extract", "matcher", "enrich", "runner"]):
        return "data_processing"

    if path.endswith((".csv", ".xlsx", ".json", ".parquet")):
        return "data_output"

    return "other"


def is_excluded(path: str) -> bool:
    return any(path.startswith(excl) for excl in EXCLUDED_DIRS)


# =========================
# History extraction
# =========================

def extract_history(repo_path: str) -> list[dict]:
    results = []

    repo = Repository(path_to_repo=repo_path)

    for commit in repo.traverse_commits():
        commit_date = commit.committer_date.date().isoformat()

        for mod in commit.modified_files:
            file_path = (mod.new_path or mod.old_path or "").replace("\\", "/")

            if not file_path.startswith(BASE_DATA_DIR):
                continue

            if is_excluded(file_path):
                continue

            activity = classify_activity(file_path)

            results.append({
                "date": commit_date,
                "month": commit_date[:7],
                "commit": commit.hash,
                "author": commit.author.name,
                "file_path": file_path,
                "activity": activity,
                "change_type": mod.change_type.name,
                "lines_added": mod.added_lines,
                "lines_removed": mod.deleted_lines,
            })

    return results


# =========================
# Output
# =========================

def save_csv(rows: list[dict]):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda x: x["date"])

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "date",
                "month",
                "commit",
                "author",
                "file_path",
                "activity",
                "change_type",
                "lines_added",
                "lines_removed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUTPUT_CSV.resolve()} ({len(rows)} rows)")


# =========================
# Main
# =========================

def main():
    if len(sys.argv) < 2:
        print("Usage: python track_data_processing_history_erwan.py <path_to_repo>")
        sys.exit(1)

    repo_path = sys.argv[1]
    save_csv(extract_history(repo_path))


if __name__ == "__main__":
    main()
