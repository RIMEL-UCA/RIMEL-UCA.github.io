from sklearn.neighbors import KernelDensity

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
#import geopandas as gpd
from mpl_toolkits.basemap import Basemap
#from shapely.geometry import Point, Polygon

#%matplotlib inline
dir_ = "../../data/database/output_ml/M1/NER_extractor/"
file = 'accident_tweets_lat_lon_3_months_bogota.tsv'

dataset = pd.read_csv(dir_+file, delimiter = "\t", quoting = 3)
df = dataset[(dataset['created_at'] >= '2018-10-01') & (dataset['created_at'] < '2018-11-01')]
geodata = df[['lat','lon']]
geodata.head(5)
latlon = np.vstack([geodata['lat'],
                    geodata['lon']]).T
latlon
kde = KernelDensity(bandwidth=0.0003, kernel='epanechnikov')
kde.fit(X = np.radians(latlon))
from sklearn.datasets import fetch_species_distributions
data = fetch_species_distributions()
data.x_left_lower_corner
data.grid_size
data.Nx
# Grid de valores dentro del rango observado (2 dimensiones)
x = np.linspace(min(geodata.lon), max(geodata.lon), 400)
y = np.linspace(min(geodata.lat), max(geodata.lat), 400)
xx, yy = np.meshgrid(x, y)
grid = np.column_stack((yy.flatten(), xx.flatten()))

log_densidad_pred = kde.score_samples(np.radians(grid))
densidad_pred = np.exp(log_densidad_pred)

zz = densidad_pred.reshape(xx.shape)
levels = np.linspace(0, zz.max(), 25)

zz.shape
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

map = Basemap(llcrnrlon=-74.253565,llcrnrlat=4.421138,urcrnrlon= -73.936334,urcrnrlat=4.874886,
             resolution='i', projection='cyl', lat_0 = 4.4863006081, lon_0 = -74.2306435108)
map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='white',lake_color='aqua')
#map.drawcoastlines()

map.readshapefile('maps/locashp/Loca', 'Loca')

#map.readshapefile('maps/munishp/Muni', 'Muni')
#map.readshapefile('maps/calzadashp/Calzada', 'Calzada')
#map.readshapefile('maps/malla/Malla_Vial_Integral_Bogota_D_C', 'Malla_Vial_Integral_Bogota_D_C')
patches   = []


"""for info, shape in zip(map.Loca_info, map.Loca):
    #if info['nombre'] == 'Selva':
    patches.append( Polygon(np.array(shape), True) )
 
ax.add_collection(PatchCollection(patches, facecolor= 'm', edgecolor='k', linewidths=1., zorder=2))
"""

#ax.scatter(geodata.lon, geodata.lat, alpha=0.5)
ax.contour(
    xx, yy, zz, levels=levels,
    #alpha=0.9,
    cmap=plt.cm.Reds,    
)
#x = np.linspace(0, map.urcrnrx, zz.shape[1])
#y = np.linspace(0, map.urcrnry, zz.shape[0])

#xx, yy = np.meshgrid(x, y)

#ax.contourf(xx, yy, zz, levels)

ax.set_title('Función de densidad estimada')
ax.set_xlabel('lon')
ax.set_ylabel('lat');

x0, y0 = m(-180, -65)
x1, y1 = m(180, 85)
plt.title("Transverse Mercator Projection")
map.imshow(zz, origin='lower',  cmap=plt.cm.Reds)
plt.show()
map.Loca
for info, shape in zip(map.Loca_info, map.Loca):
    print(info)

