import matplotlib.pyplot as plt;
import os.path
import json 

oldResult = {}

if os.path.isfile("results/ohmyzsh-ohmyzsh.json"):
        with open("results/ohmyzsh-ohmyzsh.json","r") as outfile:
            oldResult = json.loads(outfile.read())

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Commits Etiquetés', 'Commits Non Etiquetés'
tailleDesCommitsEtiquete = oldResult['infosCommits']['nombreDeCommitsEtiquettesAnalyses'] * 100 / oldResult['infosCommits']['nombreTotalDeCommitsAnalyses']
tailleDesCommitsNonEtiquete = oldResult['infosCommits']['nombreDeCommitsNonEtiquettesAnalyses'] * 100 / oldResult['infosCommits']['nombreTotalDeCommitsAnalyses']
sizes = [tailleDesCommitsEtiquete, tailleDesCommitsNonEtiquete]
explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.savefig('../charts/ohmyzsh-commits.png')