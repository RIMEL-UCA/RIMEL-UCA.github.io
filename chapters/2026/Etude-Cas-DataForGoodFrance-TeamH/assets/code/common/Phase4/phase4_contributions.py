import sys
import os
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from pydriller import Repository
import requests


def month_key(dt):
    return f"{dt.year}-{dt.month:02d}"


def get_github_profile(email, token):
    if not token:
        return {}

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }

    search_url = "https://api.github.com/search/users"
    r = requests.get(search_url, headers=headers, params={"q": email})

    if r.status_code != 200 or r.json().get("total_count", 0) == 0:
        return {}

    login = r.json()["items"][0]["login"]

    user_url = f"https://api.github.com/users/{login}"
    r = requests.get(user_url, headers=headers)
    if r.status_code != 200:
        return {}

    u = r.json()
    return {
        "github_login": login,
        "github_name": u.get("name"),
        "github_company": u.get("company"),
        "github_location": u.get("location"),
        "github_bio": u.get("bio"),
        "github_followers": u.get("followers"),
        "github_public_repos": u.get("public_repos")
    }


def main(repo_path: Path, output_dir: Path):
    output_dir.mkdir(exist_ok=True)

    commits_csv = output_dir / "commits_detailed.csv"
    contributors_csv = output_dir / "contributors_profiles.csv"

    github_token = os.getenv("GITHUB_TOKEN")

    contributors = {}
    rows = []

    for commit in Repository(str(repo_path)).traverse_commits():
        for mf in commit.modified_files:
            path = mf.new_path or mf.old_path
            if not path:
                continue

            path = path.replace("\\", "/")

            rows.append({
                "commit_hash": commit.hash,
                "commit_date": commit.committer_date.isoformat(sep=" "),
                "author_name": commit.author.name,
                "author_email": commit.author.email,
                "message": commit.msg,
                "file_path": path,
                "added_lines": mf.added_lines,
                "removed_lines": mf.deleted_lines,
                "change_type": mf.change_type.name if mf.change_type else None,
                "file_extension": Path(path).suffix,
                "year": commit.committer_date.year,
                "month": month_key(commit.committer_date)
            })

            contributors[commit.author.email] = {
                "author_name": commit.author.name,
                "author_email": commit.author.email
            }

    # Write commits CSV
    with open(commits_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Enrich contributors with GitHub metadata
    contributor_rows = []
    for email, base in contributors.items():
        profile = get_github_profile(email, github_token)
        contributor_rows.append({**base, **profile})

    with open(contributors_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=contributor_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(contributor_rows)

    print(f"Wrote {commits_csv}")
    print(f"Wrote {contributors_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python phase4_contributions.py <repo_path>")
        sys.exit(1)

    repo_path = Path(sys.argv[1]).resolve()
    if not (repo_path / ".git").exists():
        raise RuntimeError("Not a Git repository")

    output_dir = Path(__file__).parent / "results"
    main(repo_path, output_dir)
