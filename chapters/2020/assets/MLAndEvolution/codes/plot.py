import matplotlib.pyplot as plt
import matplotlib.dates as mdt

from matplotlib.dates import (YEARLY, DateFormatter,
                              rrulewrapper, RRuleLocator, drange)
import numpy as np
from datetime import datetime
import itertools
import collections
import statistics

splitLine = []
splitLineNext = []
title = ""
algoData = []
fileToLine = []


def splitLineFile(line):
    line = line.strip("\n")
    line = line.split("\t")
    splitLine.append(line)
    return line


def splitLineFileNext(line):
    line = line.strip("\n")
    line = line.split("\t")
    return line


def searchPyFile(file):
    tmp = []

    f = open(file, "r")
    lines = f.readlines()
    for i in range(0, len(lines)):

        line = lines[i]
        splitLineFile(line)

        line = splitLineFile(line)

        if i < len(lines) - 1:
            lineSuivante = splitLineFileNext(lines[i + 1])

            title = line[0]

            if title == lineSuivante[0]:
                tmp.append(line)
            else:
                tmp.append(line)
                algoData.append(tmp)
                tmp = []

            i += 1
        else:
            tmp.append(line)
            algoData.append(tmp)
    # print(splitLine)


def plotComplexityByDate(splitLine):
    i = 0
    for algo in algoData:

        title = algo[0][0]
        complexity = []
        date = []

        for data in algo:
            if str(data[1]) == 'None':
                data[1] = -100
            complexity.append(int(data[1]))
            date.append(datetime.datetime.strptime(data[2], "%d/%m/%Y"))
            # date.append(data[2])

        date = mdt.date2num(date)

        newDate, newComplexity = zip(*sorted(zip(date, complexity)))

        plt.plot_date(newDate, newComplexity, '+-')
        # plt.show()
        plt.savefig('plotComplexity/complexity' + str(title) + '.png')
        plt.clf()
        plt.close()
        i += 1


#################################### VARIATION COMPLEXITE ######################################

diminutionComplexite = []
deletedFile = []
restoreFile = []
allFileName = []
tabSplitLine = []


def removeDuplicate(tab):
    return list(dict.fromkeys(tab))


"""Permet de trier une liste d'abord par le nom de fichier puis par la date de modification"""


def sortByTwoFacteurs(file, tabSplitLine):
    with open(file) as fp:
        line = fp.readline()
        while line:
            tabSplitLine.append(splitLineFile(line))
            line = fp.readline()
        fp.close()
    tabSplitLine = sorted(tabSplitLine, key=lambda x: (x[0], datetime.strptime(x[2], '%d/%m/%Y')))
    return tabSplitLine


sortByTwoFacteurs('complexity.txt', tabSplitLine)

restoreFile = []
nbrRestoreFile = 0
"""Permet d'avoir le nombre de restoration de fichier
Il faut passer une tab triée par nom puis par date"""


def getNumberOfRestoration(tab):
    tabNone = []

    for element in tab:
        if element[1] == 'None':
            tabNone.append(element)

    count = 0
    for i in range(0, len(tabNone)):
        name = tabNone[i][0]
        if (i + 1) < len(tabNone) - 1 and name == tabNone[i + 1][0]:
            count += 1
            i += 1

        elif (i + 1) < len(tabNone) - 1 and name != tabNone[i + 1][0] and count > 1:
            restoreFile.append([name, count])
            count = 0

        elif (i + 1) == len(tabNone) - 1:
            if name == tabNone[i + 1][0]:
                count += 1
                restoreFile.append([name, count])
                count = 0
            elif count > 1:
                restoreFile.append([name, count])
                count = 0

    return restoreFile


# print(tabSplitLine)
# print(getNumberOfRestoration(tabSplitLine))
getNumberOfRestoration(tabSplitLine)
nbrRestoreFile = (len(restoreFile))


def plotNumberRestoreFile(restoreFile):
    dico = {}
    for couple in restoreFile:
        key = couple[1]
        if key in dico:
            dico[key] += 1
        else:
            dico[key] = 1
    orderedDico = collections.OrderedDict(sorted(dico.items()))
    # print(orderedDico)
    # print(dico)

    fig, ax = plt.subplots()
    ax.set_ylabel('Nombre restoration')
    nbrRestoration = orderedDico.keys()
    nbrFichier = orderedDico.values()
    plt.bar(nbrRestoration, nbrFichier)

    plt.savefig('../plotPie/nbrRestoration.png')
    plt.clf()
    plt.close()


plotNumberRestoreFile(restoreFile)

""" Remove a specifc char in a string """


def removeChar(string, char):
    counts = string.count(char)
    string = list(string)
    while counts:
        string.remove(char)
        counts -= 1
    string = ''.join(string)

    return (string)


# removeChar("['CPLMR.py', '20', '06/01/2010']", "'")


resSortUntab = []
"""Transforme le fichier complexitySort en tableau"""


def fileToTab(file):
    with open(file) as fp:
        line = fp.readline()
        while line:
            line = removeChar(line, "'")
            line = removeChar(line, "[")
            line = removeChar(line, "]")
            resSortUntab.append(line)
            line = fp.readline()

    file = open("complexitySortUntab.txt", "w")
    for element in resSortUntab:
        file.write(element)
    file.close()


# fileToTab("testComplexitySort.txt")


""" """


def fluctuationComplexity(file):
    f = open(file, 'r')
    lines = f.readlines()
    for i in range(0, len(lines)):

        line = lines[i]
        line = splitLineFile(line)
        title = line[0]
        allFileName.append(title)

        if line[1] == 'None':
            deletedFile.append(title)

        if i < len(lines) - 1:
            lineSuivante = splitLineFileNext(lines[i + 1])
            if title == lineSuivante[0]:
                # Si la complexite suivante est inferieur d'au moins 10

                if lineSuivante[1] != 'None' and int(line[1]) > (int(lineSuivante[1]) + 10):
                    diminutionComplexite.append(title)

            i += 1


def saveFileName(tab, strName):
    file = open(strName + ".txt", "w")
    for element in tab:
        file.write(str(element) + '\n')
    file.close()


def complexityAndPlot():
    searchPyFile("complexity.txt")
    print(algoData)
    plotComplexityByDate(algoData)
    print(splitLine)


def searchEvolve(diminutionComplexite, deletedFile):
    fluctuationComplexity("complexity.txt")
    diminutionComplexite = removeDuplicate(diminutionComplexite)
    deletedFile = removeDuplicate(deletedFile)

    print("diminutionComplexite : " + str(len(diminutionComplexite)))
    print("Suppression Fichier : " + str(len(deletedFile)))
    print("Nombre fichiers : " + str(len(removeDuplicate(allFileName))))


# searchEvolve(diminutionComplexite, deletedFile)
# saveFileName(removeDuplicate(diminutionComplexite), 'diminutionComplexite')
# saveFileName(removeDuplicate(deletedFile), 'deletedFile')
# saveFileName(sortByTwoFacteurs('complexity.txt', tabSplitLine), 'complexitySort')


######################################## PLOT PIE CHART #########################
# diminution = len(removeDuplicate(diminutionComplexite))
# suppression = len(removeDuplicate(deletedFile))
# total = len(removeDuplicate(allFileName))

"""Plot le pourcentage pour le maintien des fichiers (baisse complexite, suppresion fichier)"""


def plotPieRepartition():
    labels = 'diminution', 'suppression', 'reste'
    sizes = [diminution, suppression, (total - suppression - diminution)]

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False)
    ax1.axis('equal')  # Pour etre sur d'aovir un cercle
    plt.legend()

    plt.savefig('plotPie/complexityEvolution.png')
    plt.clf()
    plt.close()


# plotPieRepartition()

# TODO: Trouver la question de fichier restorés
# TODO: Afficher les dates de création et suppresion/maintien des fichiers

def plotPieRepartitionAndRestoration():
    labels = 'suppression', 'restoration'
    sizes = [suppression, nbrRestoreFile]

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False)
    ax1.axis('equal')  # Pour etre sur d'aovir un cercle
    plt.legend()

    plt.savefig('plotPie/SuppAndRepart.png')
    plt.clf()
    plt.close()


# plotPieRepartitionAndRestoration()


## TODO: Script pour récupérer la premiere date d'apparition d'un fichier
# et premiere date de diminution de comlpexite
# A lancer avec complexitySortUntab.txt
diminutionComplexiteDate = []
def traitementTab(line):
    line = line.replace(" ", "")
    line = line.strip("\n")
    line = line.split(",")
    return line

alreadyAppend = []

"""Permet de récupérer parmi tous les fichiers la première apparation
Puis la première date de baisse de complexité"""
def getFichierBaisseComplexity(file):
    f = open(file, 'r')
    lines = f.readlines()
    lineOrigin = traitementTab(lines[0])

    for i in range(0, len(lines)):
        line = traitementTab(lines[i])
        if i < len(lines)-1:
            lineSuivante = traitementTab(lines[i+1])

            if(lineOrigin[0] == line[0] and lineOrigin[0] not in alreadyAppend) :
                if(line[0] == lineSuivante[0]) :

                    if(line[1] != 'None' and lineSuivante[1] != 'None'):
                        if( int(line[1]) > (int(lineSuivante[1]) + 10)):
                            diminutionComplexiteDate.append(lineOrigin)
                            diminutionComplexiteDate.append(lineSuivante)
                            alreadyAppend.append(lineOrigin[0])

            elif(lineOrigin[0] != line[0]):
                lineOrigin = line
        i += 1



getFichierBaisseComplexity('complexitySortUntab.txt')
print(diminutionComplexiteDate)

"""Permet de calculer la durée en jours entre deux dates"""
def daysBetween2Date(date1, date2):
    delta = date1 - date2
    return(delta.days)


"""Récupère les dexu dates de chaque fichiers entrés et calcule le temps écoulé entre deux"""
resComplexityDiffAndDate = []
def differenceOfDays(diminutionComplexiteDate):
    i = 0
    while i < len(diminutionComplexiteDate)-1 :
        elt = diminutionComplexiteDate[i]
        eltSuivant = diminutionComplexiteDate[i+1]

        date1 = datetime.strptime(elt[2], "%d/%m/%Y")
        date2 = datetime.strptime(eltSuivant[2], "%d/%m/%Y")
        resComplexityDiffAndDate.append(daysBetween2Date(date1, date2)*(-1))
        i += 2

differenceOfDays(diminutionComplexiteDate)
resComplexityDiffAndDate.sort()
print(resComplexityDiffAndDate)


def moyenneDate(resComplexityDiffAndDate):
    return statistics.mean(resComplexityDiffAndDate)

print(moyenneDate(resComplexityDiffAndDate))
