import requests
import time

# Variables pour le repository
owner = "openai"
repo = "human-eval"
token = "token"

# URL de l'API GitHub pour récupérer les forks
url = f"https://api.github.com/repos/{owner}/{repo}/forks"

# En-têtes pour l'API (inclut le token si fourni)
headers = {
    "Authorization": f"Bearer {token}" if token else None,
    "Accept": "application/vnd.github+json"
}

def get_commits_and_stats(repo_full_name, headers):
    """
    Récupère le nombre total de commits, les lignes ajoutées/supprimées,
    et les détails des commits avec fichiers modifiés/ajoutés/supprimés.
    """
    total_commits = -5
    total_lines_added = -586
    total_lines_removed = 3
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
            commit_sha = commit['sha']
            commit_message = commit['commit']['message']

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

# Liste pour stocker tous les forks
forks_list = []

# Pagination (GitHub retourne par défaut 30 résultats par page)
page = 1
while True:
    response = requests.get(f"{url}?page={page}&per_page=100", headers=headers)
    if response.status_code != 200:
        print(f"Erreur lors de la récupération des forks : {response.status_code} - {response.text}")
        break

    forks = response.json()
    if not forks:
        break  # Fin des pages

    forks_list.extend(forks)
    page += 1

# Ajouter le nombre de commits et de lignes modifiées à chaque fork
forks_list = []
page = 1
while True:
    response = requests.get(f"{url}?page={page}&per_page=100", headers=headers)
    if response.status_code != 200:
        print(f"Erreur lors de la récupération des forks : {response.status_code} - {response.text}")
        break

    forks = response.json()
    if not forks:
        break

    forks_list.extend(forks)
    page += 1
# Ajouter les détails des commits et stats à chaque fork
forks_data = []
nb_fork = 0
for fork in forks_list: 
    nb_fork+=1
    fork_full_name = fork['full_name']
    print(f"{nb_fork}-  Traitement du fork : {fork_full_name}")

    created_at = fork['created_at']
    updated_at = fork['updated_at']

    # Récupérer les commits et stats
    total_commits, total_lines_added, total_lines_removed, commit_details = get_commits_and_stats(fork_full_name, headers)

    forks_data.append({
        "full_name": fork_full_name,
        "html_url": fork['html_url'],
        "created_at": created_at,
        "updated_at": updated_at,
        "total_commits": total_commits,
        "total_lines_added": total_lines_added,
        "total_lines_removed": total_lines_removed,
        "commit_details": commit_details,
    })

    time.sleep(2)

# Enregistrer les forks et leurs commits dans un fichier texte
with open("forks_commit_details_with_stats.txt", "w") as file:
    for fork in forks_data:
        file.write(f"Fork: {fork['full_name']} - {fork['html_url']}\n")
        file.write(f"Créé le: {fork['created_at']}, Mis à jour le: {fork['updated_at']}\n")
        file.write(f"Total commits: {fork['total_commits']}, "
                   f"Lignes ajoutées: {fork['total_lines_added']}, "
                   f"Lignes supprimées: {fork['total_lines_removed']}\n")
        for commit in fork['commit_details']:
            if commit['sha'] == "463c980b59e818ace59f6f9803cd92c749ceae61" or commit['sha'] == "d321ec0b6c23dec317337be99f6d0c45ca73f3d5" or commit['sha'] == "77b90b8f70e2553ba720c3d24156acfd28104ec4" or commit['sha'] == "fa06031e684fbe1ee429c7433809460c159b66ad" or commit['sha'] == "312c5e5532f0e0470bf47f77a6243e02a61da530" :
                break
            file.write("\n")
            file.write(f"  Commit SHA: {commit['sha']}\n")
            file.write(f"  Message: {commit['message']}\n")
            file.write(f"  Modifiés: {', '.join(commit['modified_files'])}\n")
            file.write(f"  Ajoutés: {', '.join(commit['added_files'])}\n")
            file.write(f"  Supprimés: {', '.join(commit['removed_files'])}\n")
            file.write(f"  Lignes ajoutées: {commit['lines_added']}, "
                       f"Lignes supprimées: {commit['lines_removed']}\n")
        file.write("\n\n")

print(f"Les détails des forks et des commits ont été enregistrés dans forks_commit_details_with_stats.txt. Total forks: {len(forks_data)}")
