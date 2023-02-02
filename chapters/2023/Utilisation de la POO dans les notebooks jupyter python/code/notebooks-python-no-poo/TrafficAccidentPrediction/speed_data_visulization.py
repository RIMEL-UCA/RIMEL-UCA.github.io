import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns
speed = pd.read_csv('speed_done.csv')
speed.plot.scatter(x = "lng", y = "lat", alpha = 0.4, c = speed["ACC"], label = "traffic", cmap=plt.get_cmap("jet"), figsize = (10, 7))
plt.legend()
plt.show()
speed.plot.scatter(x = "lng", y = "lat", alpha = 0.4, c = speed["SPD"], label = "traffic", cmap=plt.get_cmap("jet"), figsize = (10, 7))
plt.legend()
plt.show()
speed.plot.scatter(x = "lng", y = "lat", alpha = 0.4, c = speed['VOL'], cmap=plt.get_cmap("jet"), figsize = (10, 7))
plt.legend()
plt.show()
speed.sort_values(by=['ACC'],ascending = False).head(10)
speed.keys()
data = speed.drop(['Unnamed: 0', 'STREET', 'START', 'LOCAL', 'END_','lat', 'lng', 'intersection1', 'intersection2', 'intersection3', 'intersection4'], axis = 1)
scatter_matrix(data, figsize=(12, 8))
data.head(5)
corr = data.corr()
cmap = sns.diverging_palette(220, 5, as_cmap=True)
#g = sns.heatmap(corr,  vmax=.3, center=0,
            #square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.2f', cmap = cmap)
g = sns.heatmap(corr, annot=True,  cmap = 'coolwarm')
sns.despine()
g.figure.set_size_inches(14,10)
plt.show()
data.keys()
data = data.drop(['LEN', 'ACC', 'VOL'], axis=1)
corr = data.corr()
cmap = sns.diverging_palette(220, 5, as_cmap=True)
#g = sns.heatmap(corr,  vmax=.3, center=0,
            #square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.2f', cmap = cmap)
g = sns.heatmap(corr, annot=True, cmap = 'coolwarm')
sns.despine()
g.figure.set_size_inches(14,10)
plt.show()
