import requests
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

GITHUB_API_URL = "https://api.github.com"
GITHUB_TOKEN = "your_github_token"


repos = [
    {"owner": "jozu-ai", "name": "kitops"},
    {"owner": "clearml", "name": "clearml"},
    {"owner": "mlflow", "name": "mlflow"},
    {"owner": "Netflix", "name": "metaflow"},
]

def get_first_commit_date(repo_owner, repo_name, github_token):
    url = f"{GITHUB_API_URL}/repos/{repo_owner}/{repo_name}/commits"
    headers = {"Authorization": f"token {github_token}"}
    params = {"per_page": 1, "page": 709}  
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code != 200:
        print(f"Erreur : {response.status_code}, {response.json().get('message', '')}")
        return None
    
    commits = response.json()
    if commits:
        first_commit = commits[-1] 
        date = first_commit["commit"]["author"]["date"]
        return date
    else:
        print(f"Aucun commit trouvé.")
        return None

def get_contributors_count(repo_owner, repo_name, github_token):
    url = f"{GITHUB_API_URL}/repos/{repo_owner}/{repo_name}/contributors"
    headers = {"Authorization": f"token {github_token}"}
    contributors = []
    page = 1

    while True:
        params = {"per_page": 100, "page": page}
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Erreur : {response.status_code}, {response.json().get('message', '')}")
            return None
        
        page_contributors = response.json()
        if not page_contributors:
            break
        
        contributors.extend(page_contributors)
        page += 1
    
    return len(contributors)

def get_first_commit_dates_and_contributors_for_repos():
    repo_info = {}
    for repo in repos:
        first_commit_date = get_first_commit_date(repo["owner"], repo["name"], GITHUB_TOKEN)
        contributors_count = get_contributors_count(repo["owner"], repo["name"], GITHUB_TOKEN)
        if first_commit_date and contributors_count is not None:
            repo_info[repo["name"]] = {"first_commit": first_commit_date, "contributors": contributors_count}
    return repo_info

def save_summary_table_as_image(repo_info, filename="../../images/repo_summary.svg"):
    df = pd.DataFrame(repo_info).T
    df = df.reset_index().rename(columns={"index": "Repository", "first_commit": "First Commit Date", "contributors": "Number of Contributors"})

    fig, ax = plt.subplots(figsize=(10, 5)) 
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', colColours=["#2196F3", "#FFC107", "#FF5722"])
    
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_fontsize(14)
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#2196F3')
        elif j == 0:
            cell.set_fontsize(12)
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f9f9f9')
        else:
            cell.set_fontsize(12)
            cell.set_text_props(weight='normal')
            cell.set_facecolor('#ffffff')
        
        cell.set_height(0.1)
        cell.set_width(0.2)
        cell.set_edgecolor('black')
        cell.set_linewidth(1)
    
    plt.savefig(filename, bbox_inches="tight", dpi=300)  
    print(f"Le tableau a été enregistré sous {filename}.")

def main():
    repo_info = get_first_commit_dates_and_contributors_for_repos()
    
    if repo_info:
        save_summary_table_as_image(repo_info)
    else:
        print("Impossible de récupérer les informations des dépôts.")

if __name__ == "__main__":
    main()
