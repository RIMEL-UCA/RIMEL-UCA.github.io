import requests
import plotly.graph_objects as go

GITHUB_API_URL = "https://api.github.com"
GITHUB_TOKEN = "your_github_token"

repos = [
    {"owner": "jozu-ai", "name": "kitops", "target": [".github", "testing"]},
    {"owner": "clearml", "name": "clearml", "target": [".github/workflows"]},
    {"owner": "mlflow", "name": "mlflow", "target": [".circleci", ".devcontainer", ".github/actions", ".github/workflows", "tests"]},
    {"owner": "Netflix", "name": "metaflow", "target": [".github/workflows", "test", "test_config"]},
]

def get_commits_for_path(repo_owner, repo_name, path, github_token):
    url = f"{GITHUB_API_URL}/repos/{repo_owner}/{repo_name}/commits"
    headers = {"Authorization": f"token {github_token}"}
    params = {"path": path, "per_page": 100}
    
    all_commits = []
    page = 1

    while True:
        params["page"] = page
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"Erreur : {response.status_code}, {response.json().get('message', '')}")
            break
        
        commits = response.json()
        if not commits:
            break
        
        all_commits.extend(commits)
        page += 1

    return all_commits

def generate_plot(data, output_file="../../images/commits_graph.html"):
    fig = go.Figure()

    for repo_name, paths_data in data.items():
        for path, count in paths_data.items():
            fig.add_trace(go.Bar(
                x=[repo_name],
                y=[count],
                name=path,
                text=path,
                hoverinfo="text+y"
            ))
    
    fig.update_layout(
        title="Nombre de commits par chemin cible pour chaque dépôt",
        xaxis_title="Dépôts",
        yaxis_title="Nombre de commits",
        barmode="stack",  
        template="plotly_white",
    )
    
    fig.show()

    fig.write_html(output_file)

    print(f"Graphique enregistré sous : {output_file}")

def main():
    repo_data = {}

    for repo in repos:
        repo_name = repo["name"]
        repo_data[repo_name] = {}
        for target_path in repo["target"]:
            commits = get_commits_for_path(repo["owner"], repo_name, target_path, GITHUB_TOKEN)
            repo_data[repo_name][target_path] = len(commits)
            print(f"Repo: {repo_name}")
            print(f"Path: {target_path}")
            print(f"Nombre de commits : {len(commits)}\n")

    generate_plot(repo_data)

if __name__ == "__main__":
    main()
