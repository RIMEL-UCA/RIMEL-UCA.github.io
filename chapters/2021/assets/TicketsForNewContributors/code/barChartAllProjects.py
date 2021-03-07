import matplotlib.pyplot as plt;
import numpy as np
import os.path
import json 

oldResult = {}
allLabels = {}

if os.path.isfile("results/all-projects.json"):
        with open("results/all-projects.json","r") as outfile:
            oldResult = json.loads(outfile.read())

for element in oldResult:
    for label in element['labels']:
        if label not in allLabels:
            allLabels[label] = element['labels'][label]
        else:
            allLabels[label] += element['labels'][label]


sortedLabels = sorted(allLabels.items(), key = 
             lambda kv:(kv[1], kv[0]))
reversedOriginal = sortedLabels[::-1]
reversed1 = reversedOriginal[:10]

y_pos = np.arange(len(reversed1))
performance = [reversed1[i][1] for i in range (0, len(reversed1))]

plt.rc('ytick',labelsize=5)
plt.barh(y_pos, performance, align='edge', alpha=0.5, height=0.3)
plt.yticks(y_pos, (reversed1[i][0] for i in range (0, len(reversed1))))
plt.xlabel('all projects')
plt.title('Most labels used for first commits')
plt.savefig('../charts/all-projects.png')