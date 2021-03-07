import matplotlib.pyplot as plt;
import numpy as np
import os.path
import json 

oldResult = {}

if os.path.isfile("results/microsoft-vscode.json"):
        with open("results/microsoft-vscode.json","r") as outfile:
            oldResult = json.loads(outfile.read())

sortedLabels = sorted(oldResult['labels'].items(), key = 
             lambda kv:(kv[1], kv[0]))
reversedOriginal = sortedLabels[::-1]
reversed1 = reversedOriginal[:10]

y_pos = np.arange(len(reversed1))
performance = [reversed1[i][1] for i in range (0, len(reversed1))]

plt.rc('ytick',labelsize=5)
plt.barh(y_pos, performance, align='edge', alpha=0.5, height=0.3)
plt.yticks(y_pos, (reversed1[i][0] for i in range (0, len(reversed1))))
plt.xlabel('microsoft-vscode')
plt.title('Most labels used for first commits')
plt.savefig('../charts/microsoft-vscode.png')