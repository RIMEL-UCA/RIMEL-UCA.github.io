import pandas as pd
import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs

from shapely.geometry import Point

import warnings
warnings.filterwarnings('ignore')
def load_data(dirname, filename, init=None, end=None):
    dataset = pd.read_csv(dir_+file, delimiter = "\t", quoting = 3)
    if init != None and end != None:
        dataset = dataset[(dataset['created_at'] >= init) & (dataset['created_at'] < end)]        
    print("Loaded dataset form: ", dataset.shape)
    return dataset
# Twitter
dir_ = "../../data/database/output_ml/M1/NER_extractor/"
file = 'accident_tweets_lat_lon_geocord_bogota.tsv'
start_date = '2018-10-01'
final_date = '2018-11-01'

# Oficial data
#dir_ = "../../data/database/"
#file = 'historico_accidentes-oct-dic.tsv'


#accidentes = pd.read_csv(dir_+file, delimiter = "\t", quoting = 3)
#accidentes = accidentes[(accidentes['created_at'] >= '2018-10-01') & (accidentes['created_at'] < '2018-11-01')]
accidentes = load_data(dir_, file)
accidentes.head(3)
accidentes.shape
#localidades = gpd.read_file('maps/locashp/Loca.shp')
#localidades = gpd.read_file('maps/upla/UPla.shp')
localidades = gpd.read_file('maps/shp2/loca-urb.shp')
#'maps/munishp/Muni.shp'
#'maps/calzadashp/Calzada.shp'
#'maps/upla/UPla.shp'
#localidades = localidades[localidades['LocCodigo'] != '20'] ## Descartando sumapaz
localidades.head(5)
pip install adjustText
import adjustText as aT
import matplotlib.pyplot as plt
localidades['center'] = localidades['geometry'].centroid
loc_points = localidades.copy()
loc_points.set_geometry('center',inplace = True)
loc_points.geometry.x
ax = gplt.polyplot(localidades, figsize=(12, 12))
texts = []

for x, y, label in zip(loc_points.geometry.x, loc_points.geometry.y, loc_points["LocNombre"]):
    texts.append(plt.text(x, y, label, fontsize = 8))

aT.adjust_text(texts, force_points=0.3, force_text=0.8, expand_points=(1,1), expand_text=(1,1), 
               arrowprops=dict(arrowstyle="-", color='grey', lw=0.5))

points = accidentes.apply(
    lambda srs: Point(float(srs['lon']), float(srs['lat'])),
    axis='columns'
)
points
gpd_accidentes = gpd.GeoDataFrame(accidentes, geometry=points)
gpd_accidentes.head(2)
ax = gplt.polyplot(localidades,projection=gcrs.PlateCarree(),facecolor='lightgray', figsize=(12, 12))

#texts = []

#for x, y, label in zip(loc_points.geometry.x, loc_points.geometry.y, loc_points["LocNombre"]):
#    texts.append(plt.text(x, y, label, fontsize = 8))

    
gplt.pointplot(gpd_accidentes,edgecolor='lightgray', linewidth=0.5, alpha=0.2,ax=ax)
#ax.set_title('Reporte en Twitter de Accidentes en Bogotá')
#ax.set_xlabel('lon')
#ax.set_ylabel('lat');


#aT.adjust_text(texts, force_points=0.3, force_text=0.8, expand_points=(1,1), expand_text=(1,1), 
#               arrowprops=dict(arrowstyle="-", color='grey', lw=0.5))

ax.gridlines(draw_labels=True)
ax.set_xlabel('lon')
ax.set_ylabel('lat');
ax.text(-0.09, 0.55, 'latitude', va='bottom', ha='center',
        rotation='vertical', rotation_mode='anchor',fontsize='x-large',
        transform=ax.transAxes)
ax.text(0.5, -0.07,'longitude', va='bottom', ha='center',
        rotation='horizontal', rotation_mode='anchor',fontsize='x-large',
        transform=ax.transAxes)
ax.text(0.5, 1.1,'Twitter Report of Traffic Accidents in  Bogotá', va='top', ha='center',
        rotation='horizontal', rotation_mode='anchor', fontsize='xx-large',
        transform=ax.transAxes)
ax.text(0.5, 1.06,'Oct 2018 to Jul 2019', va='top', ha='center',
        rotation='horizontal', rotation_mode='anchor', fontsize='x-large',
        transform=ax.transAxes)

for x, y, label in zip(loc_points.geometry.x, loc_points.geometry.y, loc_points["LocNombre"]):
    texts.append(plt.text(x, y, label, fontsize = 8))

aT.adjust_text(texts, force_points=0.3, force_text=0.8, expand_points=(1,1), expand_text=(1,1), 
               arrowprops=dict(arrowstyle="-", color='grey', lw=0.5))

fig = ax.get_figure()
fig.savefig("hist-test.png")

ax = gplt.kdeplot(
    gpd_accidentes, 
    cmap='flare_r', 
    shade=True, 
    shade_lowest=True, 
    clip=localidades, 
    #kernel="epanechnikov",
    #bw_method='scott',
    #bw_adjust=1,
    levels=10,
    cbar=True,
    #projection=gcrs.PlateCarree(),
    figsize=(12, 12)
)

gplt.polyplot(localidades, ax=ax,zorder=1)

#ax.gridlines(draw_labels=True)
ax.text(-0.09, 0.55, 'latitude', va='bottom', ha='center',
        rotation='vertical', rotation_mode='anchor',fontsize='large',
        transform=ax.transAxes)
ax.text(0.5, -0.07,'longitude', va='bottom', ha='center',
        rotation='horizontal', rotation_mode='anchor',fontsize='large',
        transform=ax.transAxes)
ax.text(0.5, 1.1,'Intensity estimated using the Kernel method', va='top', ha='center',
        rotation='horizontal', rotation_mode='anchor', fontsize='x-large',
        transform=ax.transAxes)
ax.text(0.5, 1.06,'Oct 2018 to Jul 2019', va='top', ha='center',
        rotation='horizontal', rotation_mode='anchor', fontsize='large',
        transform=ax.transAxes)
texts = []
for x, y, label in zip(loc_points.geometry.x, loc_points.geometry.y, loc_points["LocNombre"]):
    texts.append(plt.text(x, y, label, fontsize = 8))

aT.adjust_text(texts, force_points=0.3, force_text=0.8, expand_points=(1,1), expand_text=(1,1))

fig = ax.get_figure()
fig.savefig("kde-test.png")
ax.factor
import folium
locations = accidentes[['lat','lon']]
locationlist = locations.values.tolist()
len(locationlist)
locationlist[7]
from folium import plugins

map = folium.Map(location = [4.645985, -74.097766], tiles='Stamen Terrain', zoom_start = 12)

heat_data = [[point.xy[1][0], point.xy[0][0]] for point in gpd_accidentes.geometry ]

heat_data
plugins.HeatMap(heat_data).add_to(map)

map


