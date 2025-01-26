import requests
import plotly.express as px
from collections import defaultdict
import datetime

def fetch_github_actions(repo_owner, repo_name, token):
    """
    Fetch all GitHub Actions workflow run events for a repository using pagination with query parameters.
    
    :param repo_owner: Owner of the repository.
    :param repo_name: Name of the repository.
    :param token: GitHub Personal Access Token for authentication. (with repo scope)
    :return: A list of workflow run events with timestamps.
    """
    base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/runs"
    headers = {"Authorization": f"token {token}"}
    params = {"per_page": 100, "page": 1}
    runs = []

    while True:
        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Failed to fetch data: {response.status_code} - {response.json().get('message')}")
            break

        data = response.json()
        workflow_runs = data.get("workflow_runs", [])
        runs.extend(workflow_runs)

        if len(workflow_runs) < params["per_page"]:
            break

        params["page"] += 1

    return runs

def analyze_workflows(workflow_runs):
    """
    Analyze the frequency of GitHub Actions workflow triggers by date and workflow name.
    
    :param workflow_runs: List of workflow run events with timestamps.
    :return: 
        - freq_by_date: Dictionary mapping dates to the count of triggered workflows.
        - workflow_summary: Dictionary mapping workflow names to their run counts.
        - workflow_data: List of dictionaries containing detailed workflow information.
    """
    freq_by_date = defaultdict(int)
    workflow_summary = defaultdict(int)
    workflow_data = []

    for run in workflow_runs:
        created_at = run.get("created_at")
        workflow_name = run.get("name", "Unknown")
        if created_at:
            date = datetime.datetime.fromisoformat(created_at[:-1]).date()
            freq_by_date[date] += 1
            workflow_summary[workflow_name] += 1
            workflow_data.append({"date": date, "workflow_name": workflow_name, "count": freq_by_date[date]})

    return dict(sorted(freq_by_date.items())), dict(sorted(workflow_summary.items())), workflow_data

def plot_interactive(frequency, workflow_data):
    """
    Create an interactive plot for GitHub Actions trigger frequency with workflow names.
    
    :param frequency: Dictionary mapping dates to the count of triggered workflows.
    :param workflow_data: List of dictionaries containing workflow information.
    """
    workflow_totals = defaultdict(int)
    for wd in workflow_data:
        workflow_totals[wd["workflow_name"]] += 1
    
    sorted_workflows = sorted(workflow_totals.items(), key=lambda x: x[1])
    workflow_order = [w[0] for w in sorted_workflows]

    df = [
        {"date": date, "workflow_name": wd["workflow_name"], "count": wd["count"]}
        for date, wd in zip(frequency.keys(), workflow_data)
    ]
    
    fig = px.bar(
        df,
        x="date",
        y="count",
        color="workflow_name",
        title="Plot of GitHub Actions Trigger Frequency",
        labels={"date": "Date", "count": "Number of Triggers"},
        hover_data=["workflow_name", "count"],
        category_orders={"workflow_name": workflow_order},
    )
    
    fig.for_each_trace(lambda t: t.update(
        name=f"{t.name} ({workflow_totals[t.name]} runs)"
    ))
    
    fig.update_layout(
        bargap=0.2,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.05
        )
    )
    
    output_file = "github_actions_frequency.html"
    fig.write_html(output_file)
    print(f"Visualization saved to {output_file}")
    fig.show()

if __name__ == "__main__":
    repo_owner = input("Enter the repository owner (e.g., 'octocat'): ").strip()
    repo_name = input("Enter the repository name (e.g., 'Hello-World'): ").strip()
    token = input("Enter your GitHub Personal Access Token: ").strip()

    print("Fetching GitHub Actions data...")
    workflow_runs = fetch_github_actions(repo_owner, repo_name, token)

    if workflow_runs:
        print(f"Fetched {len(workflow_runs)} workflow runs.")
        freq_by_date, workflow_summary, workflow_data = analyze_workflows(workflow_runs)

        print("\nWorkflow Run Counts:")
        for name, count in workflow_summary.items():
            print(f"Workflow: {name} | Runs: {count}")

        plot_interactive(freq_by_date, workflow_data)
        print("You can find the HTML file in:", os.path.abspath("github_actions_frequency.html"))
    else:
        print("No workflow runs found or failed to fetch data.")
