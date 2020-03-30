import numpy as np
from matplotlib import pyplot as plt

f = open("../output/outputFile.txt", "r")
allLines = f.read();
f.close();

filesImpacted = [];
allLinesClean = allLines[:(allLines.rfind(';'))]
for line in allLinesClean.split('\n'):
	nameProject = line[:(line.find('='))].replace(" ","")
	nbFilesLoc = (line[(line.find('='))+1:]).replace(" ","").split(";")
	sumAllFile = 0
	for nbFileLoc in nbFilesLoc:
		if(nbFileLoc !=  ''):
			sumAllFile = int(nbFileLoc) + sumAllFile;

	print("somme ", sumAllFile, " nb files ", len(nbFilesLoc))
	pourcentage = (sumAllFile / len(nbFilesLoc) - 1)
	filesImpacted.append(pourcentage);

plt.title('Moyenne du % de fichiers impactés par commits liés à la l10n')
plt.xticks(rotation='vertical')
plt.ylabel("Pourcentage")
plt.boxplot(filesImpacted)
plt.ylim(0,5)
plt.show()
toGraph()