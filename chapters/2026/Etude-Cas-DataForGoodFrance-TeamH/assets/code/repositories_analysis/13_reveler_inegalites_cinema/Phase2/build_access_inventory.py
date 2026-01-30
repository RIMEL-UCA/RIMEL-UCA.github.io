import csv
import sys
from pathlib import Path

RAW_EXTENSIONS = {".csv", ".json", ".xlsx", ".parquet", ".sql"}

def classify_source(rel_path: str) -> str:
    p = rel_path.lower().replace("\\", "/")

    if "database/data/allocine/" in p:
        return "allocine"
    if "database/data/cnc/" in p:
        return "cnc"
    if "database/data/mubi/" in p:
        return "mubi"
    if "database/data/machine_learning_predictions/" in p:
        return "ml_predictions"
    if "database/data/old_db_init/" in p:
        return "old_init"
    if "database/data/sample/" in p:
        return "sample"
    return "other"

def classify_access_mode(rel_path: str, ext: str) -> str:
    p = rel_path.lower().replace("\\", "/")
    name = Path(p).name

    if "database/alembic/versions/" in p:
        return "db_migration"
    if "/seed/" in p or name.startswith("seed_"):
        return "db_seed"
    if ext == ".ipynb":
        return "notebook"

    scraping_markers = ("scraper", "scraping_browser", "page_scraper")
    extraction_markers = ("extract_", "matcher", "enricher", "runner", "parse_", "matching")

    if any(m in name for m in scraping_markers) or "/scraping" in p:
        return "scraping"
    if any(m in name for m in extraction_markers):
        return "extraction"

    if ext in RAW_EXTENSIONS:
        return "raw_file"

    return "other"

def main():
    if len(sys.argv) not in (2, 3):
        print("Usage: python build_access_inventory.py <data_files.csv> [results_dir]")
        sys.exit(1)

    data_files_csv = Path(sys.argv[1]).resolve()
    results_dir = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 else data_files_csv.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / "access_inventory.csv"

    with open(data_files_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            rel_path = (r.get("path") or "").strip()
            ext = (r.get("extension") or "").strip().lower()
            size_bytes = r.get("size_bytes")

            source = classify_source(rel_path)
            access_mode = classify_access_mode(rel_path, ext)

            rows.append({
                "path": rel_path,
                "extension": ext,
                "size_bytes": size_bytes,
                "source": source,
                "access_mode": access_mode
            })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "extension", "size_bytes", "source", "access_mode"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
