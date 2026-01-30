from github import Github
import pandas as pd
from dotenv import load_dotenv
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

REPO_NAME = "dataforgoodfr/13_odis"
FILE_PATH = "datasources.yaml"

g = Github(GITHUB_TOKEN)
repo = g.get_repo(REPO_NAME)

rows = []

prs = repo.get_pulls(state="closed", base="main")

for pr in tqdm(prs, desc="PRs"):
    if pr.merged_at:
        files = [f.filename for f in pr.get_files()]
        if FILE_PATH in files:
            try:
                file = repo.get_contents(FILE_PATH, ref=pr.merge_commit_sha)
                rows.append({
                    "date": pr.merged_at,
                    "pr": pr.number,
                    "size_bytes": file.size
                })
                print(f"{pr.merged_at} | PR #{pr.number} | {file.size} bytes")
            except Exception:
                continue

df = pd.DataFrame(rows).sort_values("date")

plt.figure()
plt.plot(df["date"], df["size_bytes"], color="blue", linewidth=2)
plt.xlabel("Date")
plt.ylabel("File size (bytes)")
plt.title("File size evolution on main (per PR)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
