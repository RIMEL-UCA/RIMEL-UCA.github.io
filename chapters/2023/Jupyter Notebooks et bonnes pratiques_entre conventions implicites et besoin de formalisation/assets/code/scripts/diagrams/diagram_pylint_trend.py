import matplotlib.pyplot as plt

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


def plot_diagram(directory: str, title: str):
    pylint_scores = get_pylint_score(directory)

    occurrences = dict(sorted({element: pylint_scores.count(element) for element in pylint_scores}.items()))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(occurrences.keys(), occurrences.values(), '.-', color='r')
    ax.set_xlabel('Pylint score')
    ax.set_ylabel('Number of notebooks')
    ax.set_title(title)

    plt.xlim(0)
    plt.ylim(0)

    plt.grid()
    ax.set_axisbelow(True)
    plt.savefig(f'diagram_pylint_trend_{title}.png')

    plt.show()


if __name__ == '__main__':
    plot_diagram("../../results/", 'Score of notebooks')
