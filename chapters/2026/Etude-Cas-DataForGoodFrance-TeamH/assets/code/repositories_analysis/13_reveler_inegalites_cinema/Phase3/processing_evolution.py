import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import csv

from pydriller import Repository


def month_key(dt: datetime) -> str:
    return f"{dt.year}-{dt.month:02d}"


def classify_path(path: str, counters: dict):
    path = path.replace("\\", "/")

    if "database/notebooks/" in path and path.endswith(".ipynb"):
        counters["notebooks"] += 1

    elif "database/data/machine_learning_predictions/" in path and path.endswith(".csv"):
        counters["ml_integration"] += 1

    elif "database/alembic/versions/" in path and path.endswith(".py"):
        counters["db_migrations"] += 1

    elif "database/seed/" in path and path.endswith(".py"):
        counters["db_seeds"] += 1

    elif "database/data/" in path:
        counters["data_processing"] += 1


def main(repo_path: Path, output_csv: Path):
    evolution = defaultdict(lambda: {
        "notebooks": 0,
        "data_processing": 0,
        "db_migrations": 0,
        "db_seeds": 0,
        "ml_integration": 0
    })

    for commit in Repository(str(repo_path)).traverse_commits():
        month = month_key(commit.committer_date)

        for m in commit.modified_files:
            path = m.new_path or m.old_path
            if not path:
                continue

            classify_path(path, evolution[month])

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "month",
            "notebooks",
            "data_processing",
            "db_migrations",
            "db_seeds",
            "ml_integration"
        ])

        for month in sorted(evolution.keys()):
            row = evolution[month]
            writer.writerow([
                month,
                row["notebooks"],
                row["data_processing"],
                row["db_migrations"],
                row["db_seeds"],
                row["ml_integration"]
            ])

    print(f"Written {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python processing_evolution.py <repo_path>")
        sys.exit(1)

    repo_path = Path(sys.argv[1]).resolve()
    if not (repo_path / ".git").exists():
        raise RuntimeError("Provided path is not a Git repository")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    output_csv = results_dir / "processing_evolution.csv"
    main(repo_path, output_csv)
