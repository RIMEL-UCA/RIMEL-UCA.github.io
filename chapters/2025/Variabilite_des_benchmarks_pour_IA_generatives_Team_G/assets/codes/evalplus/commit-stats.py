import re
from collections import Counter

def parse_commit_file(file_path):
    stats = {
        "total_commits": 0,
        "total_issues": set(),
        "files_modified": Counter(),
        "lines_added": 0,
        "lines_removed": 0,
        "commit_types": Counter(),
    }

    commit_pattern = re.compile(r"^Commit SHA:")
    issue_pattern = re.compile(r"Issues: (\d+)")
    files_modified_pattern = re.compile(r"Fichiers modifiés: (.+)")
    lines_added_pattern = re.compile(r"Lignes ajoutées: (\d+)")
    lines_removed_pattern = re.compile(r"Lignes supprimées: (\d+)")
    commit_type_pattern = re.compile(r"Message: (\w+):")
    commit_type_pattern2 = re.compile(r"Message: (\w+)")
    merge_pattern = re.compile(r"Message: Merge\b")
    merge_pattern2 = re.compile(r"Message: merge\b")

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if commit_pattern.match(line):
                stats["total_commits"] += 1
            
            issue_match = issue_pattern.search(line)
            if issue_match:
                # Verifier si l'issue n'est pas déjà dans la liste
                if issue_match.group(1) not in stats["total_issues"]:
                    stats["total_issues"].add(issue_match.group(1))

            files_modified_match = files_modified_pattern.search(line)
            if files_modified_match:
                files = files_modified_match.group(1).split(", ")
                for file in files:
                    stats["files_modified"][file] += 1

            lines_added_match = lines_added_pattern.search(line)
            if lines_added_match:
                stats["lines_added"] += int(lines_added_match.group(1))

            lines_removed_match = lines_removed_pattern.search(line)
            if lines_removed_match:
                stats["lines_removed"] += int(lines_removed_match.group(1))

            if merge_pattern.search(line) or merge_pattern2.search(line):
                stats["commit_types"]["merge"] += 1
            elif commit_type_pattern2.search(line):
                if commit_type_pattern.search(line):
                    commit_type_match = commit_type_pattern.search(line)
                else:
                    commit_type_match = commit_type_pattern2.search(line)
                if commit_type_match:
                    normalized_type = normalize_commit_type(commit_type_match.group(1))
                    if(normalized_type == "uncategorized"):
                        print("Commit type not found")
                        print(line)
                    stats["commit_types"][normalized_type] += 1
                else:
                    print("Commit type not found")
                    print(line)
                    stats["commit_types"]["uncategorized"] += 1

    print("Total commits : ", stats["total_commits"])
    return stats

def normalize_commit_type(commit_type):
    '''Normalise les types de commits en catégories standard'''
    mapping = {
        "feat": "feature",
        "fix": "fix",
        "refactor": "refactor",
        "add": "feature", 
        "Refactor": "refactor",
        "refact": "refactor",
        "minor": "chore",
        "chore": "chore",
        "docs": "documentation",
        "hotfix": "fix",
        "misc": "miscellaneous",
        "re": "miscellaneous",
        "revert": "revert",
        "Fix": "fix",
        "Add": "feature",
        "Adding": "feature",
        "Update": "refactor",
    }
    
    return mapping.get(commit_type, "uncategorized")


def sort_modified_files(stats):
    '''Trie les fichiers modifiés par ordre décroissant de nombre de modifications'''
    return sorted(stats["files_modified"].items(), key=lambda item: item[1], reverse=True)

    

def save_stats_file(stats):
    '''Sauvegarde les statistiques dans un fichier texte'''
    with open("commit_stats.txt", "w") as file:
        file.write("=== Statistiques sur les commits et issues ===\n")
        file.write(f"Nombre total de commits: {stats['total_commits']}\n")
        file.write(f"Nombre total d'issues liées: {len(stats['total_issues'])}\n")
        file.write("\nFichiers modifiés:\n")
        sorted_files = sort_modified_files(stats)
        for file_name, count in sorted_files:
            file.write(f"  - {file_name}: {count} fois\n")
        file.write(f"\nLignes ajoutées: {stats['lines_added']}\n")
        file.write(f"Lignes supprimées: {stats['lines_removed']}\n")
        file.write("\nTypes de commits:\n")
        for commit_type, count in stats["commit_types"].items():
            file.write(f"  - {commit_type}: {count}\n")
        file.write("\nListes des issues:\n")
        for issue in stats["total_issues"]:
            file.write(f"  - #{issue}\n")
        file.write("=== Fin des statistiques ===\n")



if __name__ == "__main__":
    file_path = "commit_details.txt"
    stats = parse_commit_file(file_path)
    save_stats_file(stats)