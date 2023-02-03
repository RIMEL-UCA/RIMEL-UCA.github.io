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


def plot_diagram():
    cells_stats_popular = get_cells_stats("./results/", "github")
    cells_stats_lower = get_cells_stats("./results/lower", "github")

    code_popular = [element['code'] for element in cells_stats_popular]
    code_lower = [element['code'] for element in cells_stats_lower]

    markdown_popular = [element['markdown'] for element in cells_stats_popular]
    markdown_lower = [element['markdown'] for element in cells_stats_lower]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(code_popular, markdown_popular, color='r')
    ax.scatter(code_lower, markdown_lower, color='b')

    #calculate equation for trendline
    p_popular = np.poly1d(np.polyfit(code_popular, markdown_popular, 1))
    p_lower = np.poly1d(np.polyfit(code_lower, markdown_lower, 1))

    #add trendline to plot
    plt.plot(code_popular, p_popular(code_popular), color='r')
    plt.plot(code_lower, p_lower(code_lower), color='b')

    ax.legend(("Popular", "Lower", "Popular (trend)", "Lower (trend)"))
    ax.set_xlabel('Cells of code')
    ax.set_ylabel('Cells of markdown')
    ax.set_title('Cells of code and markdown (Popular vs Lower in Github)')

    plt.xticks(range(0, 125, 25))
    plt.yticks(range(0, 125, 25))
    plt.grid()
    ax.set_axisbelow(True)

    plt.xlim(0)
    plt.ylim(0)
    plt.savefig(f'diagram_cells_popular_vs_lower.png')

    plt.show()


if __name__ == '__main__':
    plot_diagram()
