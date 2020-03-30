import numpy as np
from matplotlib import pyplot as plt

f = open("../output/outputBranches.txt", "r")


def toDict():
    data = []
    for line in f:
        line = line.replace('\n', '')
        if line != "":
            div = line.split(" ")
            name = div[0]
            rest = ' '.join(div[1:])
            dataTuple = (name, rest)
            data.append(dataTuple)
    f.close()

    elegantDict = {}

    dataCopy = data.copy()

    for couple in data:
        name = couple[0]
        listOfBranches = []
        for couple2 in dataCopy:
            if couple2[0] == name:
                listOfBranches.append(couple2[1])
        elegantDict[name] = listOfBranches
    return elegantDict


def maxOfTruc(project, branches):
    print(project)
    tempMax = 0
    maxBranchName = ""
    for branch in branches:
        data = branch.split(" ")[2].split("/")
        percentage = int(data[0]) / int(data[1]) * 100
        if percentage > tempMax:
            tempMax = percentage
            maxBranchName = branch.split(" ")[0]
    answerTuple = (tempMax, maxBranchName)
    return answerTuple


def average(project, branches):
    totalPercentage = 0
    for branch in branches:
        data = branch.split(" ")[2].split("/")
        percentage = int(data[0]) / int(data[1]) * 100
        totalPercentage += percentage
    return totalPercentage / len(branches)


def nbCommitMaster(project, branches):
    for branch in branches:
        name = branch.split(" ")[0]
        if ("master" in name):
            data = branch.split(" ")[2].split("/")
            percentage = int(data[0]) / int(data[1]) * 100
            return percentage


def run():
    dataDict = toDict()
    size = dataDict.__len__()
    allMax = []
    allMoyenne = []
    master = []
    names = []
    for (project, branches) in dataDict.items():
        names.append(project)
        maxAndBranche = maxOfTruc(project, branches)
        allMax.append(maxAndBranche)
        moyenne = average(project, branches)
        allMoyenne.append(moyenne)
        master.append(nbCommitMaster(project, branches))

    width = 0.30
    X = np.arange(size)

    plt.rcParams["figure.figsize"] = (10, 10)
    p1 = plt.bar(X, allMoyenne, width, color='lightskyblue')
    p2 = plt.bar(X + width, [i[0] for i in allMax], width, color='mediumpurple')
    p3 = plt.bar(X + (width * 2), master, width, color='olivedrab')

    plt.xlabel('Projets Java')
    plt.xticks(rotation='vertical')
    plt.ylabel("Pctage des commits lies a i10n")
    plt.xticks(X, names)
    plt.legend((p1[0], p2[0], p3[0]), ('Moyenne du nbm commit l10n', 'Max nbm commit l10n', 'Nbm commit l10n master'))
    plt.show()


run()
