import matplotlib.pyplot as plt;
import os.path
import json 

oldResult = {}
allCommits = {}

if os.path.isfile("results/all-projects.json"):
        with open("results/all-projects.json","r") as outfile:
            oldResult = json.loads(outfile.read())

for element in oldResult:
    for commit in element['infosCommits']:
        if commit not in allCommits:
            allCommits[commit] = element['infosCommits'][commit]
        else:
            allCommits[commit] += element['infosCommits'][commit]

# print(allCommits)
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Commits Etiquetés', 'Commits Non Etiquetés'
tailleDesCommitsEtiquete = allCommits['nombreDeCommitsEtiquettesAnalyses'] * 100 / allCommits['nombreTotalDeCommitsAnalyses']
tailleDesCommitsNonEtiquete = allCommits['nombreDeCommitsNonEtiquettesAnalyses'] * 100 / allCommits['nombreTotalDeCommitsAnalyses']
sizes = [tailleDesCommitsEtiquete, tailleDesCommitsNonEtiquete]
explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.xlabel('All projects')
plt.title('Commits Analysés')
plt.savefig('../charts/all-projects-commits.png')