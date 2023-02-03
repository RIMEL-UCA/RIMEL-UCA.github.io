import matplotlib.pyplot as plt
from pathlib import Path

from scripts.diagrams.extractor import read_files


def round_off_rating(number):
    """
    Round a number to the closest half integer.
    """
    return round(number * 2) / 2


def get_pylint_score(directory: str):
    jsons = read_files(directory)
    scores = []
    for element in jsons:
        try:
            element = element['metrics']['code_quality']['pylint_score']['score']
            if element > 0:
                # Exclude notebooks equal to 0. Not relevant and they break the scale.
                scores.append(round_off_rating(element))
        except (TypeError, KeyError):
            pass
    return scores


def plot_diagram(directory_a: str, directory_b: str, title: str):
    path_a, path_b = Path(directory_a), Path(directory_b)
    pylint_scores_directory_a = get_pylint_score(directory_a)
    pylint_scores_directory_b = get_pylint_score(directory_b)

    occurrences_directory_a = dict(sorted({element: pylint_scores_directory_a.count(element) for element in pylint_scores_directory_a}.items()))
    occurrences_directory_b = dict(sorted({element: pylint_scores_directory_b.count(element) for element in pylint_scores_directory_b}.items()))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_directory_a = ax.plot(occurrences_directory_a.keys(), occurrences_directory_a.values(), '.-', color='r')
    plot_directory_b = ax.plot(occurrences_directory_b.keys(), occurrences_directory_b.values(), '.-', color='b')
    ax.legend((path_a.name.capitalize(), path_b.name.capitalize()), fontsize="large")
    ax.set_xlabel('Pylint score')
    ax.set_ylabel('Number of notebooks')
    ax.set_title(title)

    plt.xlim(0)
    plt.ylim(0)

    plt.grid()
    ax.set_axisbelow(True)
    plt.savefig(f"diagram_pylint_{path_a.name.capitalize()}_vs_{path_b.name.capitalize()}_{title}.png")

    plt.show()

if __name__ == '__main__':
    plot_diagram("../../results/references", "../../results/lower", "Score of notebooks")
