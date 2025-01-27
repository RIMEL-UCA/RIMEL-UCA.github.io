import requests
import time
import re

# Variables pour le repository
owner = "evalplus"
repo = "evalplus"
token = "token"  

# URL de l'API GitHub pour récupérer les commits
url = f"https://api.github.com/repos/{owner}/{repo}/commits"

# En-têtes pour l'API (inclut le token si fourni)
headers = {
    "Authorization": f"Bearer {token}" if token else None,
    "Accept": "application/vnd.github+json"
}

def extract_issues_from_commit_message(commit_message):
    """
    Extrait les numéros d'issues à partir du message de commit.
    """
    """
    Extrait les numéros d'issues à partir du message de commit.
    Les numéros d'issues sont identifiés par des motifs comme (#123).
    """
    pattern = r"#(\d+)"  # Cherche des références au format #123
    matches = re.findall(pattern, commit_message)
    return [int(match) for match in matches]

def get_commits_and_stats(repo_full_name, headers):
    """
    Récupère le nombre total de commits, les lignes ajoutées/supprimées,
    et les détails des commits avec fichiers modifiés/ajoutés/supprimés.
    """
    total_commits = 0
    total_lines_added = 0
    total_lines_removed = 0
    commit_details = []

    page = 1
    
    while True:
        commits_url = f"https://api.github.com/repos/{repo_full_name}/commits"
        response = requests.get(commits_url, headers=headers, params={"page": page, "per_page": 100})
        if response.status_code != 200:
            print(f"Erreur lors de la récupération des commits pour {repo_full_name}: {response.status_code}")
            break

        commits = response.json()
        if not commits:
            break

        if len(commits) == 5:
            return 0, 0, 0, []

        for commit in commits:
            total_commits += 1
            print(total_commits)
            commit_sha = commit['sha']
            commit_message = commit['commit']['message']

            # Extraire les numéros d'issues du message de commit
            issues = extract_issues_from_commit_message(commit_message)

            # Récupérer les détails de chaque commit
            commit_detail_url = f"https://api.github.com/repos/{repo_full_name}/commits/{commit_sha}"
            commit_response = requests.get(commit_detail_url, headers=headers)
            if commit_response.status_code == 200:
                commit_data = commit_response.json()
                files = commit_data.get('files', [])
                
                # Calculer les stats pour ce commit
                lines_added = sum(file['additions'] for file in files)
                lines_removed = sum(file['deletions'] for file in files)
                total_lines_added += lines_added
                total_lines_removed += lines_removed

                modified_files = [file['filename'] for file in files if file['status'] == 'modified']
                added_files = [file['filename'] for file in files if file['status'] == 'added']
                removed_files = [file['filename'] for file in files if file['status'] == 'removed']

                commit_details.append({
                    "sha": commit_sha,
                    "message": commit_message,
                    "issues": issues,
                    "modified_files": modified_files,
                    "added_files": added_files,
                    "removed_files": removed_files,
                    "lines_added": lines_added,
                    "lines_removed": lines_removed,
                })
            else:
                print(f"Erreur lors de la récupération des détails du commit {commit_sha} du repository {repo_full_name}: {commit_response.status_code}")
            time.sleep(1)  # Pause pour éviter de dépasser les limites API
        
        page += 1

    return total_commits, total_lines_added, total_lines_removed, commit_details


# Récupérer les commits et les statistiques pour le repository
total_commits, total_lines_added, total_lines_removed, commit_details = get_commits_and_stats(f"{owner}/{repo}", headers)

print(f"Nombre total de commits: {total_commits}")
print(f"Nombre total de lignes ajoutées: {total_lines_added}")
print(f"Nombre total de lignes supprimées: {total_lines_removed}")
print(f"Nombre total de fichiers modifiés: {sum(len(commit['modified_files']) for commit in commit_details)}")
print(f"Nombre total de fichiers ajoutés: {sum(len(commit['added_files']) for commit in commit_details)}")
print(f"Nombre total de fichiers supprimés: {sum(len(commit['removed_files']) for commit in commit_details)}")
print(f"Nombre total de commits avec détails: {len(commit_details)}")


#Enregistrement des données dans un fichier texte
with open("commit_details.txt", "w") as file:
    file.write("Détails des commits pour evalplus/evalplus\n")
    file.write(f"Nombre total de commits: {total_commits}\n")
    for commit in commit_details:
        file.write(f"Commit SHA: {commit['sha']}\n")
        file.write(f"Message: {commit['message']}\n")
        file.write(f"Issues: {', '.join(map(str, commit['issues']))}\n")
        file.write(f"Fichiers modifiés: {', '.join(commit['modified_files'])}\n")
        file.write(f"Fichiers ajoutés: {', '.join(commit['added_files'])}\n")
        file.write(f"Fichiers supprimés: {', '.join(commit['removed_files'])}\n")
        file.write(f"Lignes ajoutées: {commit['lines_added']}\n")
        file.write(f"Lignes supprimées: {commit['lines_removed']}\n")
        file.write("-------------------------------------------------------------------\n")
        file.write("\n")
