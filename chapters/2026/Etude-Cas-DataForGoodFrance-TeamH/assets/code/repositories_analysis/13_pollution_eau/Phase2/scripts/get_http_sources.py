from pathlib import Path
import re
import csv

BACK_DIR = Path("13_pollution_eau/pipelines")
OUTPUT_CSV = "results/http_references.csv"

HTTP_PATTERN = re.compile(r"https?://[^\s\"']+")
IGNORE_KEYWORD = "test"
EXTENSIONS = {".py", ".js", ".ts", ".java", ".html", ".json", ".yml", ".yaml", ".txt"}


def is_test_file(file_path: Path) -> bool:
    return IGNORE_KEYWORD.lower() in str(file_path).lower()


def should_skip_line(line: str, file_suffix: str, inside_block_comment: bool) -> (bool, bool):
    """Return (skip_line, new_inside_block_comment)"""
    stripped = line.strip()
    
    # Python multi-line comments
    if file_suffix == ".py":
        if stripped.startswith(('"""', "'''")):
            # If it opens and closes on same line, no need to enter block
            if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                return True, inside_block_comment
            return True, not inside_block_comment
        if inside_block_comment:
            return True, inside_block_comment

    # JS/TS/Java multi-line comments
    if file_suffix in {".js", ".ts", ".java"}:
        if "/*" in stripped:
            # Block starts
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


def extract_urls_from_file(file_path: Path) -> list[dict]:
    rows = []
    try:
        lines = file_path.read_text(errors="ignore").splitlines()
    except Exception:
        return rows

    inside_block_comment = False
    for line_no, line in enumerate(lines, start=1):
        skip, inside_block_comment = should_skip_line(line, file_path.suffix, inside_block_comment)
        if skip:
            continue

        for url in HTTP_PATTERN.findall(line):
            rows.append({
                "file": str(file_path),
                "line": line_no,
                "url": url
            })

    return rows


def find_http_references() -> list[dict]:
    all_rows = []
    for file_path in BACK_DIR.rglob("*"):
        if not file_path.is_file():
            continue
        if is_test_file(file_path):
            continue
        if file_path.suffix.lower() not in EXTENSIONS:
            continue

        all_rows.extend(extract_urls_from_file(file_path))
    return all_rows


def save_to_csv(rows: list[dict], output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "line", "url"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = find_http_references()
    save_to_csv(rows, OUTPUT_CSV)
    print(f"CSV généré : {OUTPUT_CSV} ({len(rows)} références HTTP trouvées)")


if __name__ == "__main__":
    main()
