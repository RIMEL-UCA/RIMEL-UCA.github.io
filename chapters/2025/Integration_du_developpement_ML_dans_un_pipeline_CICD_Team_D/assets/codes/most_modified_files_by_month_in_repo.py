import os
import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from git import Repo
import pandas as pd
import plotly.graph_objects as go


def get_monthly_file_changes(repo_path, top_n=5):
    """
    Analyze the evolution of the most modified files by month in a Git repository.

    :param repo_path: Path to the Git repository.
    :param top_n: Number of top modified files to consider for visualization.
    :return: Dictionary with months as keys and file change counts as values.
    """
    repo = Repo(repo_path)
    assert not repo.bare, "Repository is not valid or bare."

    file_changes = defaultdict(lambda: defaultdict(int))

    for commit in repo.iter_commits():
        commit_date = datetime.datetime.fromtimestamp(commit.committed_date)
        month = commit_date.strftime("%Y-%m")

        for file_stat in commit.stats.files.keys():
            file_changes[month][file_stat] += commit.stats.files[file_stat]["lines"]

    monthly_top_files = {}
    for month, files in file_changes.items():
        sorted_files = sorted(files.items(), key=lambda x: x[1], reverse=True)[:top_n]
        monthly_top_files[month] = {file: changes for file, changes in sorted_files}

    return monthly_top_files

def plot_monthly_file_changes_vertical(monthly_changes):
    """
    Plot the evolution of the most modified files by month using a bar plot.

    :param monthly_changes: Dictionary with months as keys and file change counts as values.
    """
    months = sorted(monthly_changes.keys())
    all_files = set()
    for changes in monthly_changes.values():
        all_files.update(changes.keys())

    file_modifications = {file: [] for file in all_files}
    for month in months:
        changes = monthly_changes.get(month, {})
        for file in file_modifications:
            file_modifications[file].append(changes.get(file, 0))

    plt.figure(figsize=(14, 8))
    bottom = [0] * len(months)
    for file, modifications in file_modifications.items():
        plt.bar(months, modifications, bottom=bottom, label=file)
        for i, value in enumerate(modifications):
            if value > 0:
                plt.text(
                    i, 
                    bottom[i] + value / 2, 
                    f"{file}\n({value})", 
                    ha='center', 
                    va='center', 
                    fontsize=8, 
                    rotation=90
                )
        bottom = [b + m for b, m in zip(bottom, modifications)]

    plt.title("Most Modified Files by Month", fontsize=16)
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Number of Changes", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(title="Files", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()

def plot_monthly_file_changes_horizontal(monthly_changes):
    """
    Plot the evolution of the most modified files by month using a horizontal stacked bar plot.

    :param monthly_changes: Dictionary with months as keys and file change counts as values.
    """
    months = sorted(monthly_changes.keys())
    all_files = set()
    for changes in monthly_changes.values():
        all_files.update(changes.keys())

    file_modifications = {file: [] for file in all_files}
    for month in months:
        changes = monthly_changes.get(month, {})
        for file in file_modifications:
            file_modifications[file].append(changes.get(file, 0))

    plt.figure(figsize=(10, 10))
    left = [0] * len(months)
    for file, modifications in file_modifications.items():
        plt.barh(months, modifications, left=left, label=file)
        for i, value in enumerate(modifications):
            if value > 0:
                plt.text(
                    left[i] + value / 2,
                    i,
                    f"{file} ({value})",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        left = [l + m for l, m in zip(left, modifications)]

    plt.title("Most Modified Files by Month", fontsize=16)
    plt.xlabel("Number of Changes", fontsize=12)
    plt.ylabel("Month", fontsize=12)
    plt.tight_layout()
    plt.legend(title="Files", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()


def plot_monthly_file_changes_interactive(monthly_changes):
    """
    Plot the evolution of the most modified files by month using an interactive bar chart.

    :param monthly_changes: Dictionary with months as keys and file change counts as values.
    """
    months = sorted(monthly_changes.keys())
    all_files = set()
    for changes in monthly_changes.values():
        all_files.update(changes.keys())

    file_totals = {}
    for file in all_files:
        total = sum(monthly_changes[month].get(file, 0) for month in months)
        file_totals[file] = total
    
    sorted_files = sorted(file_totals.items(), key=lambda x: x[1], reverse=False)

    fig = go.Figure()
    for file, total_changes in sorted_files:
        values = [
            monthly_changes[month].get(file, 0) for month in months
        ]
        fig.add_trace(
            go.Bar(
                name=f"{file} ({total_changes} changes)",
                x=months,
                y=values,
                text=[f"{file}: {v}" if v > 0 else "" for v in values],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        barmode="stack",
        title="Most Modified Files by Month",
        xaxis_title="Month",
        yaxis_title="Number of Changes",
        legend_title="Files (Total Changes)",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.05
        )
    )
    
    output_file = "monthly_file_changes.html"
    fig.write_html(output_file)
    print(f"Visualization saved to {output_file}")
    fig.show()


if __name__ == "__main__":
    repo_path = input("Enter the path to your Git repository: ").strip()
    top_n = int(input("Enter the number of top modified files to analyze (e.g., 5): ").strip())

    if not os.path.exists(repo_path) or not os.path.isdir(repo_path):
        print(f"Invalid repository path: {repo_path}")
    else:
        print("Analyzing repository...")
        monthly_changes = get_monthly_file_changes(repo_path, top_n)
        plot_monthly_file_changes_interactive(monthly_changes)
        print("You can find the HTML file in:", os.path.abspath("monthly_file_changes.html"))
