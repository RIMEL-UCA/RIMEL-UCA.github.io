import matplotlib.pyplot as plt
import numpy as np
from matplotlib_venn import venn2
from os.path import join
    
def scatterplot(points, x_label, y_label, title, filepath, filename): # nuage de points
    plt.clf()
    x_data = []
    y_data = []
    for p in points:
        x_data.append(p[0])
        y_data.append(p[1])
    plt.scatter(x_data, y_data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    z = np.polyfit(x_data, y_data, 1)
    p = np.poly1d(z)
    plt.plot(x_data,p(x_data),"r--")
    plt.savefig(join(filepath, filename))
    #plt.show()

# https://python-graph-gallery.com/170-basic-venn-diagram-with-2-groups/

def vennplot(arr1, arr2, label1, label2, title, filepath, filename):
    plt.clf()
    venn2([set(arr1), set(arr2)], set_labels = (label1, label2))
    plt.title(title)
    plt.savefig(join(filepath, filename))
    #plt.show()
