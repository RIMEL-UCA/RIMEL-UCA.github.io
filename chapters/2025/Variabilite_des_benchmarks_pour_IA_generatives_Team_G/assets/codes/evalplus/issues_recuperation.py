import requests
import json
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

def get_issues(repo_full_name, headers, issue):
    ''' Récupère les issues à partir de leur numéro'''

    issues = []
    for i in issue:
        issue_url = f"https://api.github.com/repos/{repo_full_name}/issues/{i}"
        response = requests.get(issue_url, headers=headers)
        if response.status_code == 200:
            issue_data = response.json()
            issues.append(issue_data)

    return issues


def get_issue_number(file):
    '''Récupère les numéros d'issues à partir d'un fichier'''

    issue_pattern = re.compile(r"#(\d+)")
    issues = set()

    with open(file, "r", encoding="utf-8") as file:
        for line in file:
            issue_match = issue_pattern.search(line)
            if issue_match:
                issues.add(issue_match.group(1))

    return issues

def save_issues(issues_data, file):
    '''Sauvegarde les données des issues dans un fichier texte'''

    with open(file, "w") as file:
        for issue in issues_data:
            file.write("Issue: " + str(issue["number"]) + "\n")
            file.write("Title: " + issue["title"] + "\n")
            file.write("State: " + issue["state"] + "\n")
            body = issue.get("body")
            if body is None:
                body = "No body"
            file.write("Body: " + body + "\n")
            file.write("\n")
    

    


if __name__ == "__main__":
    issues = get_issue_number("commit_stats.txt")
    issues_data = get_issues(f"{owner}/{repo}", headers, issues)
    save_issues(issues_data, "issues_from_commit.txt")