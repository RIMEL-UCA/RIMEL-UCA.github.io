import os
from git import Repo
import datetime
from collections import defaultdict
import plotly.graph_objects as go

def track_file_changes(repo_path, file_path):
    """
    Track changes of a specific file over time in a Git repository.
    
    :param repo_path: Path to the Git repository
    :param file_path: Path to the file to track (relative to repo root)
    :return: Dictionary with dates as keys and number of changes as values
    """
    repo = Repo(repo_path)
    assert not repo.bare, "Repository is not valid or bare."

    changes_by_date = defaultdict(int)
    
    for commit in repo.iter_commits(paths=file_path):
        date = datetime.datetime.fromtimestamp(commit.committed_date).date()
        
        if commit.stats.files.get(file_path):
            stats = commit.stats.files[file_path]
            changes_by_date[date] += stats['insertions'] + stats['deletions']

    return dict(sorted(changes_by_date.items()))

def plot_file_changes(changes_by_date, file_path):
    """
    Create an interactive plot showing the evolution of changes for a specific file.
    
    :param changes_by_date: Dictionary with dates as keys and number of changes as values
    :param file_path: Path to the file (for the title)
    """
    dates = list(changes_by_date.keys())
    changes = list(changes_by_date.values())
    
    cumulative_changes = []
    total = 0
    for change in changes:
        total += change
        cumulative_changes.append(total)

    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=dates,
        y=changes,
        name='Daily Changes',
        hovertemplate='Date: %{x}<br>Changes: %{y}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=cumulative_changes,
        name='Cumulative Changes',
        yaxis='y2',
        hovertemplate='Date: %{x}<br>Total Changes: %{y}<extra></extra>'
    ))

    fig.update_layout(
        title=f'File Evolution: {os.path.basename(file_path)}',
        xaxis_title='Date',
        yaxis_title='Number of Changes',
        yaxis2=dict(
            title='Cumulative Changes',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        barmode='relative'
    )
    
    file_name = os.path.basename(file_path)
    output_file = "file_evolution_" + file_name.replace(".", "_") + ".html"
    fig.write_html(output_file)
    print(f"Visualization saved to {output_file}")
    fig.show()

if __name__ == "__main__":
    repo_path = input("Enter the path to your Git repository: ").strip()
    file_path = input("Enter the path to the file (relative to repo root): ").strip()

    if not os.path.exists(repo_path) or not os.path.isdir(repo_path):
        print(f"Invalid repository path: {repo_path}")
    else:
        print("Analyzing file changes...")
        changes = track_file_changes(repo_path, file_path)
        
        if changes:
            print(f"\nFound changes across {len(changes)} different dates")
            total_changes = sum(changes.values())
            print(f"Total number of changes: {total_changes}")
            
            plot_file_changes(changes, file_path)
            print("You can find the HTML file in:", os.path.abspath("file_evolution.html"))
        else:
            print("No changes found for the specified file.")