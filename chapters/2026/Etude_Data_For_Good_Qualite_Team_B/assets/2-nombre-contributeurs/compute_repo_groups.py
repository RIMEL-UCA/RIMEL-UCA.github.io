#!/usr/bin/env python3
"""Groupes de dépôts par nombre de contributeurs (3 groupes).
Sort et écrit `2-nombre-contributeurs/repos_groups.csv`.
"""
import csv
from collections import defaultdict
from pathlib import Path

CONTRIB_CSV = Path("2-nombre-contributeurs/data/contributors.csv")
OUT_CSV = Path("2-nombre-contributeurs/repos_groups.csv")

if not CONTRIB_CSV.exists():
    print(f"Fichier introuvable: {CONTRIB_CSV}")
    raise SystemExit(1)

# Lire le fichier de contributeurs
repos = []
with open(CONTRIB_CSV, newline='', encoding='utf-8') as f:
    r = csv.reader(f)
    header = next(r, None)
    for row in r:
        if not row:
            continue
        repo = row[0].strip()
        try:
            count = int(row[1])
        except Exception:
            count = 0
        repos.append((repo, count))

if not repos:
    OUT_CSV.write_text("repo_name,repo_group\n")
    raise SystemExit(0)

# Regrouper par nombre de contributeurs
buckets = defaultdict(list)
for repo, c in repos:
    buckets[c].append(repo)

sorted_counts = sorted(buckets.keys())

# Liste triée des comptes
items = [(c, buckets[c]) for c in sorted_counts]

# Total de dépôts
total = sum(len(lst) for _, lst in items)
if total == 0:
    OUT_CSV.write_text("repo_name,repo_group\n")
    raise SystemExit(0)

# Cibles de répartition
import math
first_target = math.ceil(total / 3)
second_target = math.ceil(2 * total / 3)

# Trouver coupes en respectant les buckets
groups = {}
cum = 0
i_cut = None
j_cut = None
for idx, (cnt, lst) in enumerate(items):
    cum += len(lst)
    if i_cut is None and cum >= first_target:
        i_cut = idx
    if j_cut is None and cum >= second_target:
        j_cut = idx

# Assigner groupe par bucket
for idx, (cnt, lst) in enumerate(items):
    if i_cut is None:
        g = 1
    elif j_cut is None:
        g = 1 if idx <= i_cut else 2
    else:
        if idx <= i_cut:
            g = 1
        elif idx <= j_cut:
            g = 2
        else:
            g = 3
    for repo in lst:
        groups[repo] = g

# Écrire CSV de sortie
with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['repo_name', 'repo_group'])
    for repo, _ in sorted(repos, key=lambda x: x[0]):
        w.writerow([repo, groups.get(repo, 1)])

print(f"Écrit {OUT_CSV} ({len(groups)} dépôts)")
