import sys
sys.path.append('../')

from os import listdir
from os.path import isfile, isdir, join

import json

from plots import vennplot, scatterplot

#########
# utils #
#########

def getDirPaths(outputPath):
    res = []
    files = listdir(outputPath)
    for f in files:
        if isdir(join(outputPath, f)):
            res.append(join(outputPath, f))
    return res

def getFilePaths(projectOutputPath):
    res = []
    files = listdir(projectOutputPath)
    for f in files:
        if isfile(join(projectOutputPath, f)):
            res.append(join(projectOutputPath, f))
    return res

#######################
# data viz question 1 #
#######################

def getDataFromFile1(filePath):
        projectName = filePath.split('\\')[-2]
        data = []
        with open(filePath) as f:
                j = json.load(f)
                for test in j:
                        for file in test['data']:
                                filename = file['name']
                                for line in file['line']:
                                        if line['c'] == 1:
                                                data.append(projectName + '#' + filename + '#' + str(line['nr']))
        return data

def getData1(dirPaths):
        unitData = []
        funcData = []
        for d in dirPaths:
                projectName = d.split("\\")[-1]
                projectUnitData = []
                projectFuncData = []
                filePaths = getFilePaths(d)
                for f in filePaths:
                        if "matrix-tf.json" in f:
                                projectUnitData += getDataFromFile1(f)
                        elif "matrix-tu.json" in f:
                                projectFuncData += getDataFromFile1(f)
                if projectUnitData != [] and projectFuncData != []:
                    vennplot(projectUnitData, projectFuncData, "TU", "TF", "Ensembles de lignes testées par des TU et des TF\ndans le projet " + projectName, d, "vennplot.png")
                    unitData += projectUnitData
                    funcData += projectFuncData
        return {"unit": unitData, "func": funcData}
        
def script1(dirPaths, outputPath):
    data1 = getData1(dirPaths)
    vennplot(data1["unit"], data1["func"], "TU", "TF", "Ensembles de lignes testées par des TU et des TF\n(diagramme global)", outputPath, "vennplot_global.png")
    
#######################
# data viz question 2 #
#######################
    
def getDataFromFile2(filePath):
        projectName = filePath.split('\\')[-2]
        points = []
        with open(filePath) as f:
                j = json.load(f)
                for method in j:
                        points.append((method["countUnit"],method["countFunc"]))
        return points
    
def getData2(dirPaths):
        points = []
        for d in dirPaths:
                projectName = d.split("\\")[-1]
                projectPoints = []
                filePaths = getFilePaths(d)
                for f in filePaths:
                        if "matrix-tests-sum-merged.json" in f:
                                projectPoints += getDataFromFile2(f)
                                
                if projectPoints != []:
                    scatterplot(projectPoints, "nbTU", "nbTF", "Le nombre de TF en fonction du nombre de TU par méthode testée\ndans le projet " + projectName, d, "scatterplot.png")
                    points += projectPoints
        return points
    
def script2(dirPaths, outputPath):
    points = getData2(dirPaths)
    scatterplot(points, "nbTU", "nbTF", "Le nombre de TF en fonction du nombre de TU par méthode testée\n(graphique global)", outputPath, "scatterplot_global.png")

#######################
# data viz question 3 #
#######################
    
def script3(dirPaths, outputPath):
    pass
   
########
# main #
########
   
if __name__ == '__main__':
    outputPath = "../output"
    dirPaths = getDirPaths(outputPath)
    print(dirPaths)
    script1(dirPaths, outputPath)
    script2(dirPaths, outputPath)
    script3(dirPaths, outputPath)
    print('DONE')
