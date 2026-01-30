import csv
import sys
from pathlib import Path

# Extensions pour les fichiers de données brutes
RAW_EXTENSIONS = {".csv", ".json", ".xlsx", ".parquet", ".sql", ".txt"}

def classify_source(rel_path: str) -> str:
    """Classifie la source de données pour The Shift Data Portal."""
    p = rel_path.lower().replace("\\", "/")

    # Données dans le dossier data/
    if "data/" in p:
        # Extraire le sous-dossier si possible
        parts = p.split("data/")
        if len(parts) > 1:
            subfolder = parts[1].split("/")[0]
            return f"data_{subfolder}"
        return "data_root"
    
    # Données de préparation
    if "data-preparation/" in p:
        return "data_preparation"
    
    # Serveur (base de données embarquée)
    if "server/data/" in p:
        return "server_embedded_db"
    
    return "other"


def classify_access_mode(rel_path: str, ext: str) -> str:
    """Classifie le mode d'accès aux données."""
    p = rel_path.lower().replace("\\", "/")
    name = Path(p).name

    # Scripts de préparation Python
    if "data-preparation/" in p and ext == ".py":
        if "main_" in name:
            return "data_pipeline"
        if "transformation" in p or "transform" in name:
            return "data_transformation"
        if "utils" in p or "util" in name:
            return "utility"
        return "data_script"
    
    # Notebooks Jupyter
    if ext == ".ipynb":
        return "notebook"
    
    # Fichiers de données brutes
    if ext in RAW_EXTENSIONS:
        if "data-preparation/" in p:
            return "intermediate_data"
        if "server/data/" in p:
            return "embedded_db"
        if "data/" in p:
            return "raw_data"
        return "data_file"
    
    # Configuration
    if name in ("pyproject.toml", "poetry.lock", "requirements.txt", "package.json"):
        return "config"
    
    # Scripts shell
    if ext == ".sh":
        return "shell_script"
    
    # API/Backend
    if "server/" in p and ext in (".ts", ".js"):
        if "graphql" in p or ext == ".graphql":
            return "api_graphql"
        return "api_backend"
    
    # Frontend
    if "client/" in p and ext in (".tsx", ".jsx", ".ts", ".js"):
        return "frontend"
    
    return "other"


def main():
    if len(sys.argv) not in (2, 3):
        print("Usage: python build_access_inventory_shiftdataportal.py <data_files.csv> [results_dir]")
        print("\nExample:")
        print("  python build_access_inventory_shiftdataportal.py results/data_files.csv")
        sys.exit(1)

    data_files_csv = Path(sys.argv[1]).resolve()
    
    if not data_files_csv.exists():
        print(f" Error: File not found: {data_files_csv}")
        sys.exit(1)
    
    results_dir = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 else data_files_csv.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / "access_inventory_shiftdataportal.csv"

    print(f"[+] Reading: {data_files_csv}")
    
    with open(data_files_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        
        for r in reader:
            rel_path = (r.get("path") or "").strip()
            ext = (r.get("extension") or "").strip().lower()
            size_bytes = r.get("size_bytes", "")

            source = classify_source(rel_path)
            access_mode = classify_access_mode(rel_path, ext)

            rows.append({
                "path": rel_path,
                "extension": ext,
                "size_bytes": size_bytes,
                "source": source,
                "access_mode": access_mode
            })

    print(f"[+] Processed {len(rows)} files")
    
    # Statistiques rapides
    sources = {}
    access_modes = {}
    for row in rows:
        sources[row["source"]] = sources.get(row["source"], 0) + 1
        access_modes[row["access_mode"]] = access_modes.get(row["access_mode"], 0) + 1
    
    print(f"\n Sources found ({len(sources)}):")
    for source, count in sorted(sources.items(), key=lambda x: -x[1])[:10]:
        print(f"   {count:4d} - {source}")
    
    print(f"\n Access modes found ({len(access_modes)}):")
    for mode, count in sorted(access_modes.items(), key=lambda x: -x[1])[:10]:
        print(f"   {count:4d} - {mode}")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "extension", "size_bytes", "source", "access_mode"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[✓] Wrote: {out_path}")

if __name__ == "__main__":
    main()