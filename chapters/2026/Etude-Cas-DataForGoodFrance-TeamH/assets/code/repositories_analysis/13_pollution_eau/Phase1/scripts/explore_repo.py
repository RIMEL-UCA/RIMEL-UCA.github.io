import os
import sys
import json
import csv
from collections import defaultdict
from pathlib import Path

DATA_EXTENSIONS = {
    ".csv", ".json", ".parquet", ".xlsx", ".sql", ".ipynb", "py"
}

DATA_KEYWORDS = {
    "data", "dataset", "datasets", "raw", "processed",
    "scraping", "etl", "ingestion", "pipelines"
}

DATA_RAW_EXTENSIONS = {".csv", ".json", ".xlsx", ".parquet", ".sql"}
DATA_FORMAT_EXTENSIONS = DATA_RAW_EXTENSIONS | {".ipynb"}


def build_tree(root_path):
    tree = {}
    for root, dirs, files in os.walk(root_path):
        rel_path = os.path.relpath(root, root_path)
        node = tree
        if rel_path != ".":
            for part in rel_path.split(os.sep):
                node = node.setdefault(part, {})
        for d in dirs:
            node.setdefault(d, {})
        for f in files:
            node[f] = None
    return tree


def explore_files(root_path):
    extension_count = defaultdict(int)
    data_files = []

    for path in Path(root_path).rglob("*"):
        if path.is_file():
            ext = path.suffix.lower()
            extension_count[ext] += 1

            if (
                ext in DATA_EXTENSIONS or
                any(k in str(path).lower() for k in DATA_KEYWORDS)
            ):
                data_files.append({
                    "path": str(path.relative_to(root_path)),
                    "extension": ext,
                    "size_bytes": path.stat().st_size
                })

    return extension_count, data_files


def compute_repo_summary(extension_count, total_files, data_files):
    data_raw_count = sum(
        count for ext, count in extension_count.items()
        if ext in DATA_RAW_EXTENSIONS
    )

    data_code_count = extension_count.get(".py", 0)
    notebooks_count = extension_count.get(".ipynb", 0)

    data_formats_count = len([
        ext for ext in extension_count
        if ext in DATA_FORMAT_EXTENSIONS
    ])

    return {
        "data_file_ratio": round(len(data_files) / total_files, 3),
        "data_raw_ratio": round(data_raw_count / total_files, 3),
        "data_code_ratio": round(data_code_count / total_files, 3),
        "data_formats_count": data_formats_count,
        "notebooks_ratio": round(notebooks_count / total_files, 3)
    }


def main(repo_path):
    repo_path = Path(repo_path).resolve()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    tree = build_tree(repo_path)
    ext_count, data_files = explore_files(repo_path)
    total_files = sum(ext_count.values())

    with open(results_dir / "repo_structure.json", "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2)

    with open(results_dir / "file_distribution.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["extension", "count"])
        for ext, count in sorted(ext_count.items()):
            writer.writerow([ext or "NO_EXT", count])

    with open(results_dir / "data_files.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "extension", "size_bytes"])
        writer.writeheader()
        writer.writerows(data_files)

    profile = {
        "total_files": total_files,
        "data_files_count": len(data_files),
        "data_file_ratio": round(len(data_files) / max(1, total_files), 3),
        "extensions": dict(ext_count)
    }

    with open(results_dir / "repo_profile.json", "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    summary = compute_repo_summary(ext_count, total_files, data_files)

    with open(results_dir / "repo_stats_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Phase 1 exploration completed.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python explore_repo.py <repo_path>")
        sys.exit(1)
    main(sys.argv[1])
