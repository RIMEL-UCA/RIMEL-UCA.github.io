import subprocess
import sys
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import csv

def run(cmd, cwd):
    return subprocess.check_output(cmd, shell=True, text=True, cwd=cwd)

def extract_month(date_str):
    return datetime.strptime(date_str, "%Y-%m").strftime("%Y-%m")

CATEGORIES = {
    "raw_data_files": [
        re.compile(r"^database/data/.*\.(csv|xlsx|json|sql)$")
    ],
    "scraping_code": [
        re.compile(r"(scraper|scraping_browser).*\.(py|ipynb)$")
    ],
    "extraction_processing_code": [
        re.compile(r"(extract_|matcher|enricher|runner).*\.(py|ipynb)$")
    ],
    "db_migrations": [
        re.compile(r"^database/alembic/versions/")
    ],
    "db_seeds": [
        re.compile(r"^database/seed/")
    ],
    "data_access_code": [
        re.compile(r"(repository|dao|loader|datasource|db_client).*\.(py|ipynb)$")
    ],
}

if len(sys.argv) != 2:
    print("Usage: python access_churn.py <path_to_git_repo>")
    sys.exit(1)

repo_path = Path(sys.argv[1]).resolve()

if not repo_path.exists():
    print(f"❌ Path does not exist: {repo_path}")
    sys.exit(1)

if not (repo_path / ".git").exists():
    print(f"❌ Not a git repository: {repo_path}")
    sys.exit(1)

print(f"[+] Analyzing repo at {repo_path}")

log = run(
    "git log --name-only --pretty=format:%ad --date=format:%Y-%m",
    cwd=repo_path
)

monthly = defaultdict(lambda: defaultdict(int))
current_month = None
touched_categories = set()

for line in log.splitlines():
    line = line.strip()

    if re.match(r"\d{4}-\d{2}", line):
        if current_month is not None:
            for cat in touched_categories:
                monthly[current_month][cat] += 1

        current_month = extract_month(line)
        touched_categories = set()

    elif line:
        for cat, patterns in CATEGORIES.items():
            for p in patterns:
                if p.search(line):
                    touched_categories.add(cat)

if current_month is not None:
    for cat in touched_categories:
        monthly[current_month][cat] += 1


results_dir = Path.cwd() / "results"
results_dir.mkdir(exist_ok=True)

output_file = results_dir / "access_churn.csv"

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["month"] + list(CATEGORIES.keys()))

    for month in sorted(monthly.keys()):
        writer.writerow(
            [month] + [monthly[month].get(cat, 0) for cat in CATEGORIES]
        )

print(f"[✓] Result written to {output_file}")
