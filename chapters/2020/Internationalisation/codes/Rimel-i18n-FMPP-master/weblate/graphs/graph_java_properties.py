import numpy as np
from matplotlib import pyplot as plt

f = open("../output/result_properties.txt", "r")
allLines = f.read();
f.close();

javaFiles = [];
propertiesFile = [];
names = [];
allLinesClean = allLines[:(allLines.rfind(';'))]
for line in allLinesClean.split(';'):
	nameProject = line[:(line.find(','))].replace(" ","")
	names.append(nameProject)

	nbJava = (line[(line.find(','))+1:line.find('=')]).replace(" ","")
	javaFiles.append(int(nbJava))

	linesPackage = line[(line.rfind('='))+1:].replace("\n", "").replace(" ","")
	sumProperties = 0;
	for package in linesPackage.split('..'):
		if package != '':
			sumProperties += int(package[(package.find(':')+1):])
	propertiesFile.append(sumProperties);


ind = np.arange(len(names))
width = 0.50
p1 = plt.bar(ind, tuple(javaFiles), width)
p2 = plt.bar(ind, tuple(propertiesFile), width, bottom=tuple(javaFiles))

plt.ylabel('Nombre de fichier')
plt.title('')
plt.xticks(ind, tuple(names))
plt.legend((p1[0], p2[0]), ('Java File', 'Properties'))

plt.show()
