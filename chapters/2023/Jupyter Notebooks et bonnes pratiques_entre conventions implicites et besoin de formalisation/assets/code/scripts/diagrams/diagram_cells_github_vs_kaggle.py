import matplotlib.pyplot as plt
import numpy as np

from scripts.diagrams.extractor import read_files


def get_cells_stats(directory: str, source: str):
    jsons = read_files(directory)
    cells_stats = []
    for element in jsons:
        try:
            if source not in element['notebook']:
                continue
            pylint = element['metrics']['code_quality']['pylint_score']['score']
            code = element['metrics']['nb_code_cells']
            markdown = element['metrics']['nb_markdown_cells']
            if pylint > 0:
                # Exclude notebooks where pylint is equals to 0. Not relevant.
                cells_stats.append({"code": code, "markdown": markdown})
        except (TypeError, KeyError):
            pass
    return cells_stats


def plot_diagram(directory: str, title: str):
    cells_stats_github = get_cells_stats(directory, "github")
    cells_stats_kaggle = get_cells_stats(directory, "kaggle")

    code_github = [element['code'] for element in cells_stats_github]
    code_kaggle = [element['code'] for element in cells_stats_kaggle]

    markdown_github = [element['markdown'] for element in cells_stats_github]
    markdown_kaggle = [element['markdown'] for element in cells_stats_kaggle]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(code_github, markdown_github, color='r')
    if code_kaggle:
        ax.scatter(code_kaggle, markdown_kaggle, color='b')

    #calculate equation for trendline
    z_github = np.polyfit(code_github, markdown_github, 1)
    p_github = np.poly1d(z_github)
    if code_kaggle:
        z_kaggle = np.polyfit(code_kaggle, markdown_kaggle, 1)
        p_kaggle = np.poly1d(z_kaggle)

    #add trendline to plot
    plt.plot(code_github, p_github(code_github), color='r')
    if code_kaggle:
        plt.plot(code_kaggle, p_kaggle(code_kaggle), color='b')

    ax.legend(("GitHub", "Kaggle") if code_kaggle else ("Github", ), fontsize="large")
    ax.set_xlabel('Cells of code')
    ax.set_ylabel('Cells of markdown')
    ax.set_title(title)

    plt.grid()
    ax.set_axisbelow(True)

    plt.xlim(0)
    plt.ylim(0)
    plt.savefig(f"diagram_cells_{'github_vs_kaggle' if code_kaggle else 'github'}_{title}.png")
    
    plt.show()


if __name__ == '__main__':
    plot_diagram("../../results/", 'Number of markdown per code cells')
