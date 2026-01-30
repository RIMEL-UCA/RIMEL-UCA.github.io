import subprocess
import sys
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import csv


def run(cmd, cwd):
    """Execute a shell command and return its output."""
    return subprocess.check_output(cmd, shell=True, text=True, cwd=cwd)


def extract_month(date_str):
    """Extract month from date string."""
    return datetime.strptime(date_str, "%Y-%m").strftime("%Y-%m")


# Categories adapted for The Shift Data Portal structure
CATEGORIES = {
    "raw_data_files": [
        re.compile(r"^data/.*\.(csv|xlsx|json|txt|sql)$"),
    ],
    "data_preparation_scripts": [
        re.compile(r"^data-preparation/.*\.(py|ipynb|sh)$"),
    ],
    "backend_api": [
        re.compile(r"^server/.*\.(ts|js|graphql)$"),
        re.compile(r"^server/.*\.json$"),  # config files
    ],
    "frontend_client": [
        re.compile(r"^client/.*\.(tsx?|jsx?|css|scss)$"),
    ],
    "infrastructure_deployment": [
        re.compile(r"^\.circleci/"),
        re.compile(r"Dockerfile"),
        re.compile(r"\.yml$"),
        re.compile(r"\.yaml$"),
    ],
    "documentation": [
        re.compile(r"README\.md$", re.IGNORECASE),
        re.compile(r"^docs/"),
        re.compile(r"\.md$"),
    ],
}


if len(sys.argv) != 2:
    print("Usage: python access_churn_shiftdataportal.py <path_to_repo>")
    sys.exit(1)

repo_path = Path(sys.argv[1]).resolve()

# Validation du chemin
if not repo_path.exists():
    print(f" Path does not exist: {repo_path}")
    sys.exit(1)

if not (repo_path / ".git").exists():
    print(f" Not a git repository: {repo_path}")
    sys.exit(1)

print(f"[+] Analyzing Shift Data Portal repo at {repo_path}")

# Extraction des logs Git
log = run(
    "git log --name-only --pretty=format:%ad --date=format:%Y-%m",
    cwd=repo_path
)

# Traitement des commits
monthly = defaultdict(lambda: defaultdict(int))
current_month = None
touched_categories = set()

for line in log.splitlines():
    line = line.strip()
    
    # Nouvelle date de commit
    if re.match(r"\d{4}-\d{2}", line):
        # Sauvegarder les catégories touchées du commit précédent
        if current_month is not None:
            for cat in touched_categories:
                monthly[current_month][cat] += 1
        
        current_month = extract_month(line)
        touched_categories = set()
    
    # Fichier modifié
    elif line:
        for cat, patterns in CATEGORIES.items():
            for pattern in patterns:
                if pattern.search(line):
                    touched_categories.add(cat)
                    break

# Sauvegarder le dernier commit
if current_month is not None:
    for cat in touched_categories:
        monthly[current_month][cat] += 1

# Création du dossier results et écriture du CSV
results_dir = Path.cwd() / "results"
results_dir.mkdir(exist_ok=True)
output_file = results_dir / "access_churn_shiftdataportal.csv"

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["month"] + list(CATEGORIES.keys()))
    
    for month in sorted(monthly.keys()):
        writer.writerow(
            [month] + [monthly[month].get(cat, 0) for cat in CATEGORIES]
        )

print(f"[✓] Result written to {output_file}")
print(f"[i] Analyzed {len(monthly)} months of commits")
print(f"[i] Categories tracked: {', '.join(CATEGORIES.keys())}")