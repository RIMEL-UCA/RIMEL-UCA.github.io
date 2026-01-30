import json
import csv
import re
from collections import Counter
from pathlib import Path
import unicodedata
CONVENTIONAL = [
    ("feat", re.compile(r"^\s*feat(\([^)]+\))?(!)?:", re.I)),
    ("fix", re.compile(r"^\s*fix(\([^)]+\))?(!)?:", re.I)),
    ("refactor", re.compile(r"^\s*refactor(\([^)]+\))?(!)?:", re.I)),
    ("ci", re.compile(r"^\s*ci(\([^)]+\))?(!)?:", re.I)),
    ("chore", re.compile(r"^\s*chore(\([^)]+\))?(!)?:", re.I)),
]

PATTERNS = {
    "feat": re.compile(
        r"\b(feature|feat(ure)?|implement|ajout|ajouter|nouveau|nouvelle|fonctionnalite|implementation)\b",
        re.I,
    ),
    "fix": re.compile(
        r"\b(fix|bug|hotfix|patch|corrige(r)?|correction|repare(r)?|resolution)\b",
        re.I,
    ),
    "refactor": re.compile(
        r"\b(refactor|cleanup|rework|simplify|refacto|refactorisation|nettoyage|restructur(e|ation)|simplifi(er|e|cation))\b",
        re.I,
    ),
    "ci": re.compile(
        r"\b(pipeline|workflow|ci|github actions|gitlab ci|jenkins|integration continue|build|compilation|tests? auto)\b",
        re.I,
    ),
    "chore": re.compile(
        r"\b(chore|deps?|dependencies|bump|release|version|dependances?|mise a jour|maj)\b",
        re.I,
    ),
}

def normalize_msg(msg) -> str:
    if msg is None:
        return ""
    s = str(msg).strip().replace("\n", " ")
    s = unicodedata.normalize("NFKD", s)
    return s

def classify_with_patterns(msg):
    msg = normalize_msg(msg)

    # 1) conventional commits (prioritaire)
    for cat, pattern in CONVENTIONAL:
        if pattern.search(msg):
            return cat

    # 2) fallback keywords
    for cat, pattern in PATTERNS.items():
        if pattern.search(msg):
            return cat

    return "other"

DATA_FILE = Path("3-activite-contributeurs/data/raw_commits_data.json")
CSV_OUT = Path("3-activite-contributeurs/data/commits_types.csv")
UNCLASSIFIED_OUT = Path("3-activite-contributeurs/data/commits_unclassified.json")

with open(DATA_FILE, encoding="utf-8") as f:
    repos_data = json.load(f)

results = []
unclassified = []

for repo, repo_data in repos_data.items():
    commits = repo_data.get("commits", [])
    counts = Counter()
    
    for msg in commits:
        label = classify_with_patterns(msg)
        counts[label] += 1
        
        if label == "other":
            unclassified.append({
                "repo": repo,
                "message": msg
            })

    results.append({
        "repo": repo,
        "feat": counts.get("feat", 0),
        "fix": counts.get("fix", 0),
        "refactor": counts.get("refactor", 0),
        "ci": counts.get("ci", 0),
        "chore": counts.get("chore", 0),
        "other": counts.get("other", 0),
        "total_commits": len(commits)
    })

with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["repo", "feat", "fix", "refactor", "ci", "chore", "other", "total_commits"]
    )
    writer.writeheader()
    writer.writerows(results)

with open(UNCLASSIFIED_OUT, "w", encoding="utf-8") as f:
    json.dump(unclassified, f, ensure_ascii=False, indent=2)

print(f"[OK] CSV généré : {CSV_OUT}")
print(f"[OK] Commits non classés : {len(unclassified)} → {UNCLASSIFIED_OUT}")
