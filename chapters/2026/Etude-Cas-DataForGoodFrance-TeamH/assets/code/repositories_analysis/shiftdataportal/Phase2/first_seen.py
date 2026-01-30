import csv
import sys
from pathlib import Path
from datetime import datetime
from pydriller import Repository

def load_access_inventory(csv_path):
    path_to_source = {}
    path_to_mode = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row["path"].replace("\\", "/")
            path_to_source[path] = row["source"]
            path_to_mode[path] = row["access_mode"]

    return path_to_source, path_to_mode


def main():
    if len(sys.argv) != 4:
        print("Usage: python first_seen.py <repo_path> <access_inventory.csv> <results_dir>")
        sys.exit(1)

    repo_path = Path(sys.argv[1]).resolve()
    inventory_csv = Path(sys.argv[2]).resolve()
    results_dir = Path(sys.argv[3]).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    path_to_source, path_to_mode = load_access_inventory(inventory_csv)

    first_seen_source = {}
    first_seen_mode = {}

    for commit in Repository(str(repo_path)).traverse_commits():
        commit_date = commit.committer_date

        for mod in commit.modified_files:
            if not mod.new_path:
                continue

            path = mod.new_path.replace("\\", "/")

            if path in path_to_source:
                source = path_to_source[path]
                if source not in first_seen_source:
                    first_seen_source[source] = (commit.hash, commit_date)

            if path in path_to_mode:
                mode = path_to_mode[path]
                if mode not in first_seen_mode:
                    first_seen_mode[mode] = (commit.hash, commit_date)

    with open(results_dir / "first_seen_sources.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "first_commit_hash", "first_commit_date", "first_commit_month"])
        for source, (h, d) in sorted(first_seen_source.items(), key=lambda x: x[1][1]):
            writer.writerow([
                source,
                h,
                d.date().isoformat(),
                d.strftime("%Y-%m")
            ])

    with open(results_dir / "first_seen_modes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["access_mode", "first_commit_hash", "first_commit_date", "first_commit_month"])
        for mode, (h, d) in sorted(first_seen_mode.items(), key=lambda x: x[1][1]):
            writer.writerow([
                mode,
                h,
                d.date().isoformat(),
                d.strftime("%Y-%m")
            ])

    print("Phase 2 first-seen analysis completed.")


if __name__ == "__main__":
    main()