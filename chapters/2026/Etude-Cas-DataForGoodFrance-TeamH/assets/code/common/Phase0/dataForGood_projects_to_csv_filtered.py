from github import Github
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import os
from tqdm import tqdm

# Load env
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
ORG_NAME = "dataforgoodfr"

g = Github(GITHUB_TOKEN)
org = g.get_organization(ORG_NAME)
repos = org.get_repos()

now = datetime.now(timezone.utc)
results = []

ten_months_ago = now - timedelta(days=int(365.25 * 10 / 12))
one_year_ago = now - timedelta(days=365)

for repo in tqdm(repos, desc="Processing repos"):
    try:
        commits = repo.get_commits().totalCount
        contributors = repo.get_contributors().totalCount
        created_at = repo.created_at
        last_update = repo.pushed_at
    except Exception:
        continue

    # FILTER PIPELINE
    if commits < 100:
        continue

    if created_at > ten_months_ago:
        continue

    if last_update < one_year_ago:
        continue

    if contributors < 10:
        continue

    results.append({
        "repo_name": repo.name,
        "nb_commits": commits,
        "nb_contributors": contributors,
        "created_at": created_at,
        "last_update": last_update
    })

df = pd.DataFrame(results).sort_values(
    by=["nb_commits", "nb_contributors", "last_update"],
    ascending=[False, False, False]
)

df.to_csv("filtered_projects.csv", index=False)
print(f"CSV généré : {len(df)} projets retenus → filtered_projects.csv")
