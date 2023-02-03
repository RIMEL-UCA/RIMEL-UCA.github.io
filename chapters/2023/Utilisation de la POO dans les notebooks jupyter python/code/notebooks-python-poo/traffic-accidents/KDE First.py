import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot') or plt.style.use('ggplot')
dir_ = "../../data/database/output_ml/M1/NER_extractor/"
file = 'accident_tweets_lat_lon_3_months_bogota.tsv'

dataset = pd.read_csv(dir_+file, delimiter = "\t", quoting = 3)
df = dataset[(dataset['created_at'] >= '2018-10-01') & (dataset['created_at'] < '2018-11-01')]
geodata = df[['lat','lon']]
geodata.head(5)
geodata.info()
# Histograma de cada variable
# ==============================================================================
fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(10,4))

axs[0].hist(geodata.lat, bins=30, density=True, color="#3182bd", alpha=0.5)
axs[0].set_title('Distribución lat')
axs[0].set_xlabel('lat')
axs[0].set_ylabel('densidad')

axs[1].hist(geodata.lon, bins=30, density=True, color="#3182bd", alpha=0.5)
axs[1].set_title('Distribución lon')
axs[1].set_xlabel('lon')
axs[1].set_ylabel('densidad');


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.5, 4))
ax.scatter(geodata.lon, geodata.lat, color="#3182bd", alpha=0.5)
ax.set_title('Distribución geoespacial')
ax.set_xlabel('lon')
ax.set_ylabel('lat');
from scipy import stats
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
bandwidths = np.linspace(0.00001, 0.0001, 100)
bandwidths
np.linspace(0.0001, 0.001, 10)
latlon = np.vstack([geodata['lat'],
                    geodata['lon']]).T
latlon
np.radians(latlon).shape
# Validación cruzada para identificar kernel y bandwidth
# ==============================================================================

param_grid = {'kernel': ['gaussian', 'epanechnikov', 'exponential', 'linear'],
              'bandwidth' : bandwidths
             }

grid = GridSearchCV(
        estimator  = KernelDensity(),
        param_grid = param_grid,
        n_jobs     = -1,
        cv         = 10,
        verbose    = 2
      )

# Se asigna el resultado a _ para que no se imprima por pantalla
grid.fit(X = np.radians(latlon))
# Mejores hiperparámetros por validación cruzada
# ==============================================================================
print("----------------------------------------")
print("Mejores hiperparámetros encontrados (cv)")
print("----------------------------------------")
print(grid.best_params_, ":", grid.best_score_, grid.scoring)

modelo_kde_final = grid.best_estimator_
kde = KernelDensity(bandwidth=0.0003, kernel='epanechnikov')
kde.fit(X = np.radians(latlon))
# Mapa de densidad de probabilidad
# ==============================================================================
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

# Grid de valores dentro del rango observado (2 dimensiones)
x = np.linspace(min(geodata.lon), max(geodata.lon), 800)
y = np.linspace(min(geodata.lat), max(geodata.lat), 800)
xx, yy = np.meshgrid(x, y)
grid = np.column_stack((yy.flatten(), xx.flatten()))

# Densidad de probabilidad de cada valor del grid
#log_densidad_pred = modelo_kde_final.score_samples(np.radians(grid))
#densidad_pred = np.exp(log_densidad_pred)

log_densidad_pred = kde.score_samples(np.radians(grid))
densidad_pred = np.exp(log_densidad_pred)

ax.scatter(geodata.lon, geodata.lat, alpha=0.6)
ax.contour(
    xx, yy, densidad_pred.reshape(xx.shape),
    alpha =0.6,
    cmap="Reds"
)
ax.set_title('Función de densidad estimada')
ax.set_xlabel('lon')
ax.set_ylabel('lat');

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
ax.scatter(geodata.lon, geodata.lat, alpha=0.6)
zz = densidad_pred.reshape(xx.shape)
levels = np.linspace(0, zz.max(), 40)
ax.contour(
    xx, yy, zz, levels,
    alpha=0.9,
    cmap="Reds",    
)
ax.set_title('Función de densidad estimada')
ax.set_xlabel('lon')
ax.set_ylabel('lat');
# Representación como mapa de calor
#===============================================================================
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
ax.scatter(grid[:,1], grid[:,0], alpha=0.6, c=densidad_pred)
ax.set_title('Función de densidad estimada')
ax.set_xlabel('intervalo (waiting)')
ax.set_ylabel('duración (duration)');
from mpl_toolkits.mplot3d import axes3d
plt.style.use('default')
fig = plt.figure(figsize=(5, 5))
ax = plt.axes(projection='3d')
#ax.view_init(60, 35)
ax.plot_surface(xx, yy, densidad_pred.reshape(xx.shape), cmap='viridis')
ax.set_xlabel('intervalo (waiting)')
ax.set_ylabel('duración (duration)')
ax.set_zlabel('densidad')
ax.set_title('Superficie 3D densidad')
plt.show()
plt.style.use('ggplot');

import mpl_toolkits

from mpl_toolkits.basemap import Basemap
from sklearn.datasets.species_distributions import construct_grids

xgrid, ygrid = construct_grids(data)

# plot coastlines with basemap
m = Basemap(projection='cyl', resolution='c',
            llcrnrlat=ygrid.min(), urcrnrlat=ygrid.max(),
            llcrnrlon=xgrid.min(), urcrnrlon=xgrid.max())
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='#FFEEDD')
m.drawcoastlines(color='gray', zorder=2)
m.drawcountries(color='gray', zorder=2)

# plot locations
m.scatter(latlon[:, 1], latlon[:, 0], zorder=3,
          c=species, cmap='rainbow', latlon=True);



