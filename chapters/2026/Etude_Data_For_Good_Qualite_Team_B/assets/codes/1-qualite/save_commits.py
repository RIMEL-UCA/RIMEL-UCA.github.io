#!/usr/bin/env python3
import sys
import json
import os


def main():
    if len(sys.argv) < 3:
        print("Usage: save_commits.py <repo> <owner>", file=sys.stderr)
        sys.exit(2)

    repo = sys.argv[1]
    owner = sys.argv[2]

    commits = [l for l in sys.stdin.read().splitlines() if l]

    out = os.path.join('3-activite-contributeurs', 'data', 'raw_commits_data.json')
    data = {}
    if os.path.exists(out):
        try:
            with open(out, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            data = {}

    data[repo] = {
        'repo': repo,
        'owner': owner,
        'commits': commits
    }

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"{repo}: {len(commits)} commits enregistrés")


if __name__ == '__main__':
    main()
