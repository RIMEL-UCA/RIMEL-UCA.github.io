import numpy as np
from matplotlib import pyplot as plt

f = open("../output/outputAuthor.txt", "r")


def toGraph():
    data = []
    name = ""
    for line in f:
        line = line.replace('\n', '')
        if "Weblate" in line:
            data.append(float(line.split(" ")[-1]))
    print(data)
    f.close()

    plt.xlabel('Moyenne du % de commits fait par Weblate')
    plt.xticks(rotation='vertical')
    plt.ylabel("Pourcentage")
    plt.boxplot(data)
    plt.ylim(-10,100)
    plt.show()
toGraph()

