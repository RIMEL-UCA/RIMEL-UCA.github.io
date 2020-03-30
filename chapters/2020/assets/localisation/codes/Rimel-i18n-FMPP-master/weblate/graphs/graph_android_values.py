import numpy as np
from matplotlib import pyplot as plt

f = open("../output/result_properties_android.txt", "r")
allLines = f.read();
f.close();

javaFiles = [];
valuesFolder = [];
names = [];

allLinesClean = allLines[:(allLines.rfind(';'))]
for line in allLinesClean.split(';'):
	nameProject = line[:(line.find(','))].replace(" ","")
	names.append(nameProject)

	nbJava = (line[(line.find(','))+1:line.find('=')]).replace(" ","")
	javaFiles.append(int(nbJava))

	LinesValueFolders = line[(line.rfind('='))+1:].replace("\n", "").replace(" ","")
	sumProperties = 0;
	for package in LinesValueFolders.split('..'):
		if package != '':
			sumProperties += int(package[(package.find(':')+1):])
	valuesFolder.append(sumProperties);


ind = np.arange(len(names))
width = 0.50
p1 = plt.bar(ind, tuple(javaFiles), width)
p2 = plt.bar(ind, tuple(valuesFolder), width, bottom=tuple(javaFiles))

plt.ylabel('Nombre de fichier')
plt.title('Quantité de fichiers de traductions sur la quantité de fichiers java pour les projets android')
plt.xticks(ind, tuple(names), rotation='vertical')
plt.legend((p1[0], p2[0]), ('Java File', 'Values files'))
plt.savefig("android_project", bbox_inches="tight")

plt.show()
