import numpy as np
import matplotlib.pyplot as plot


N = 5
menMeans = (20, 35, 30, 35, 27)
womenMeans = (25, 32, 34, 20, 25)
menStd = (2, 3, 4, 1, 2)
womenStd = (3, 5, 2, 3, 3)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence
fig,plt = plot.subplots()
p1 = plt.bar(ind, menMeans, width, yerr=menStd)
p2 = plt.bar(ind, womenMeans, width,
             bottom=menMeans, yerr=womenStd)

plot.ylabel('Scores')
plot.title('Scores by group and gender')
plot.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
# plt.set_yscale("log")
plt.legend((p1[0], p2[0]), ('Men', 'Women'))

plot.show()
