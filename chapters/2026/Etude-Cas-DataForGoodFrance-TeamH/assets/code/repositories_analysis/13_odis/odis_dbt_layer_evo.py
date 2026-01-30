from github import Github
from dotenv import load_dotenv
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# --- Config ---
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

REPO_NAME = "dataforgoodfr/13_odis"
BASE_PATH = "dbt_odis/models"
LAYERS = ["bronze", "silver", "gold"]

g = Github(GITHUB_TOKEN)
repo = g.get_repo(REPO_NAME)

rows = []

commits = repo.get_commits(sha="main")

for commit in tqdm(commits):
    sha = commit.sha
    date = commit.commit.committer.date

    layer_counts = {layer: 0 for layer in LAYERS}

    try:
        contents = repo.get_contents(BASE_PATH, ref=sha)

        stack = contents[:]
        while stack:
            item = stack.pop()

            if item.type == "dir":
                stack.extend(repo.get_contents(item.path, ref=sha))
            else:
                for layer in LAYERS:
                    if f"/{layer}/" in item.path:
                        layer_counts[layer] += 1

        rows.append({
            "date": date,
            "commit": sha,
            **layer_counts
        })

        print(
            f"{date.date()} | "
            f"bronze={layer_counts['bronze']} | "
            f"silver={layer_counts['silver']} | "
            f"gold={layer_counts['gold']}"
        )

    except Exception:
        continue

df = pd.DataFrame(rows).sort_values("date")

plt.figure(figsize=(10, 6))
plt.plot(df["date"], df["bronze"], label="Bronze", linewidth=2)
plt.plot(df["date"], df["silver"], label="Silver", linewidth=2)
plt.plot(df["date"], df["gold"], label="Gold", linewidth=2)

plt.xlabel("Date")
plt.ylabel("File count")
plt.title("DBT Layer File Count Evolution on main")
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nSummary :")
print(df.tail())
