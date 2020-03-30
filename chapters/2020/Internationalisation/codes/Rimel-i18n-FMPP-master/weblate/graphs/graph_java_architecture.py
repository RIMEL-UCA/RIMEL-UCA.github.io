import numpy as np
from matplotlib import pyplot as plt

f = open("../output/result_properties.txt", "r")
allLines = f.read();
f.close();

javaFiles = [];
propertiesFolder = [];
names = [];
allLinesClean = allLines[:(allLines.rfind(';'))]
for line in allLinesClean.split(';'):
	nameProject = line[:(line.find(','))].replace(" ","")
	names.append(nameProject)

	nbJava = (line[(line.find(','))+1:line.find('=')]).replace(" ","")
	javaFiles.append(int(nbJava))

	linesPackage = line[(line.rfind('='))+1:].replace("\n", "").replace(" ","")
	sumPackage = 0;
	for package in linesPackage.split('..'):
		if package != '':
			sumPackage += 1;
	propertiesFolder.append(sumPackage);
			

plt.title("Nombre de repertoire contenant des .properties liés à la localisation");
plt.xlabel('Projets Java');
plt.xticks(rotation='vertical');
plt.ylabel('Nombre de repertoire');
plt.bar(names, propertiesFolder);
plt.show();