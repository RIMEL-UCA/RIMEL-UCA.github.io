from pathlib import Path
import re
import csv
import subprocess
from datetime import datetime
import sys

BACK_DIR = Path("pipelines")
OUTPUT_CSV = "results/http_references_history.csv"

HTTP_PATTERN = re.compile(r"https?://[^\s\"'\)\]\s]+")
IGNORE_KEYWORD = "test"
EXTENSIONS = {".py", ".js", ".ts", ".java", ".html", ".json", ".yml", ".yaml", ".txt"}


def check_git_repo():
    """Exit if we are not inside a Git repository"""
    try:
        subprocess.check_output(["git", "rev-parse", "--is-inside-work-tree"], stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        sys.exit("Error: You are not inside a Git repository.")


def is_test_file(file_path: Path) -> bool:
    return IGNORE_KEYWORD.lower() in str(file_path).lower()


def get_commits(file_path: Path) -> list[str]:
    """List all commits for a file"""
    try:
        output = subprocess.check_output(
            ["git", "log", "--format=%H", "--", file_path.as_posix()],
            stderr=subprocess.DEVNULL
        ).decode("utf-8", errors="ignore")
        return output.splitlines()
    except subprocess.CalledProcessError:
        return []


def get_file_at_commit(file_path: Path, commit_hash: str) -> str:
    """Return file content at a given commit"""
    try:
        return subprocess.check_output(
            ["git", "show", f"{commit_hash}:{file_path.as_posix()}"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8", errors="ignore")
    except subprocess.CalledProcessError:
        return ""


def get_commit_date(commit_hash: str) -> str:
    """Return commit date as YYYY-MM-DD"""
    try:
        date_str = subprocess.check_output(
            ["git", "show", "-s", "--format=%ci", commit_hash],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S %z").date().isoformat()
    except Exception:
        return ""


def should_skip_line(line: str, file_suffix: str, inside_block_comment: bool) -> (bool, bool):
    """Return (skip_line, new_inside_block_comment)"""
    stripped = line.strip()

    # Python multi-line comments
    if file_suffix == ".py":
        if stripped.startswith(('"""', "'''")):
            if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                return True, inside_block_comment
            return True, not inside_block_comment
        if inside_block_comment:
            return True, inside_block_comment

    # JS/TS/Java multi-line comments
    if file_suffix in {".js", ".ts", ".java"}:
        if "/*" in stripped:
            inside_block_comment = True
            if "*/" in stripped and stripped.index("*/") > stripped.index("/*"):
                inside_block_comment = False
            return True, inside_block_comment
        if inside_block_comment:
            if "*/" in stripped:
                inside_block_comment = False
            return True, inside_block_comment

    # Single-line comments
    if (file_suffix == ".py" and stripped.startswith("#")) or \
       (file_suffix in {".js", ".ts", ".java"} and stripped.startswith("//")):
        return True, inside_block_comment

    return False, inside_block_comment


def extract_urls_from_content(file_path: Path, commit_hash: str, content: str, commit_date: str) -> list[dict]:
    rows = []
    lines = content.splitlines()
    inside_block_comment = False

    for line_no, line in enumerate(lines, start=1):
        skip, inside_block_comment = should_skip_line(line, file_path.suffix, inside_block_comment)
        if skip:
            continue

        for url in HTTP_PATTERN.findall(line):
            rows.append({
                "file": str(file_path),
                "commit": commit_hash,
                "date": commit_date,
                "line": line.strip()
            })
    return rows


def find_http_references_history() -> list[dict]:
    all_rows = []
    for file_path in BACK_DIR.rglob("*"):
        if not file_path.is_file():
            continue
        if is_test_file(file_path):
            continue
        if file_path.suffix.lower() not in EXTENSIONS:
            continue

        commits = get_commits(file_path)
        for commit in commits:
            content = get_file_at_commit(file_path, commit)
            if not content:
                continue
            commit_date = get_commit_date(commit)
            if not commit_date:
                continue
            all_rows.extend(extract_urls_from_content(file_path, commit, content, commit_date))

    return all_rows


def save_to_csv(rows: list[dict], output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "commit", "date", "line"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    check_git_repo()
    rows = find_http_references_history()
    save_to_csv(rows, OUTPUT_CSV)
    print(f"CSV historique généré : {OUTPUT_CSV} ({len(rows)} références HTTP trouvées)")


if __name__ == "__main__":
    main()
