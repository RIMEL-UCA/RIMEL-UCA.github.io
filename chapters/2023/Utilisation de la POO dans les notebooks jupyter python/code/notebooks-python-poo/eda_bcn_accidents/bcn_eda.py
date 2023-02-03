import gmaps
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Proj, transform
import seaborn as sns
import time

# Corrected dataset errors
# 2016 - 2016S000807 -missing comma in lat/lon
# 2016 - 2016S000750 -missing comma in lat/lon
# 2016 - 2016S001148 -missing comma in lat/
# 2016 - 2016S001379 -missing comma in lat/lon
# 2016 - 2016S001648 -missing comma in lat/lon
# 2016 - 2016S001984 -missing comma in lat/lon
# 2016 - 2016S004967 -missing comma in lat/lon
# 2016 - 2016S005568 -missing comma in lat/lon
# 2016 - 2016S006047 -missing comma in lat/lon
# 2016 - 2016S007360 -missing comma in lat/lon
# 2016 - 2016S007664 -missing comma in lat/lon

# Configure GMAPS with the Google API Key
gmaps.configure(api_key="your-key")

# Pandas - Parameters
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# The resulting plots are stored in the notebook document.
%matplotlib inline

# Seaborn - Parameters
plt.rcParams['figure.figsize']=(16,10)
plt.style.use('ggplot')


# Datasets from http://opendata-ajuntament.barcelona.cat/

# Load dataset individually
dt_2010 = pd.read_csv("Dataset/2010_accidents.csv",encoding="cp1252", na_values=[-1, "Desconegut"], decimal=',')
dt_2011 = pd.read_csv("Dataset/2011_accidents.csv",encoding="cp1252", na_values=[-1, "Desconegut"], decimal=',')
dt_2012 = pd.read_csv("Dataset/2012_accidents.csv",encoding="cp1252", na_values=[-1, "Desconegut"], decimal=',')
dt_2013 = pd.read_csv("Dataset/2013_accidents.csv",encoding="cp1252", na_values=[-1, "Desconegut"], decimal=',')
dt_2014 = pd.read_csv("Dataset/2014_accidents.csv",encoding="utf-8", na_values=[-1, "Desconegut"], decimal=',')
dt_2015 = pd.read_csv("Dataset/2015_accidents.csv",encoding="utf-8", na_values=[-1, "Desconegut"], decimal=',')
dt_2016 = pd.read_csv("Dataset/2016_accidents.csv",encoding="utf-8", na_values=[-1, "Desconegut"], decimal=',')

# List of datasets
dt = {
    "2010": dt_2010, "2011": dt_2011, "2012": dt_2012,
    "2013": dt_2013, "2014": dt_2014, "2015": dt_2015,
    "2016": dt_2016
}


# Rename - Ugly but easy :D
column_names_24 = ["AccidentID", "DistricCode", "DistrictName", "HoodCode", "HoodName", "StreetCode", "StreetName", 
                   "PostalCode", "DayName", "DayNameShort", "DayType", "Year", "Month", "MonthName", "NumberDayMonth",
                   "Hour", "VictimIsPedestrian", "VehicleType", "VictimGender", "VictimRole", 
                   "VictimAge", "VictimStatus", "UTM35Y", "UTM35X"]
column_names_25 = ["AccidentID", "DistricCode", "DistrictName", "HoodCode", "HoodName", "StreetCode", "StreetName", 
                   "PostalCode", "DayName", "DayNameShort", "DayType", "Year", "Month", "MonthName", "NumberDayMonth",
                   "PoliceWorkShift", "Hour", "VictimIsPedestrian", "VehicleType", "VictimGender", "VictimRole", 
                   "VictimAge", "VictimStatus", "UTM35Y", "UTM35X"]
column_names_28 = ["AccidentID", "DistricCode", "DistrictName", "HoodCode", "HoodName", "StreetCode", "StreetName", 
                   "PostalCode", "DayName", "DayNameShort", "DayType", "Year", "Month", "MonthName", "NumberDayMonth",
                   "PoliceWorkShift", "Hour", "VictimIsPedestrian", "VehicleType", "VictimGender", "VictimAge", 
                   "VictimRole", "AccidentDescription", "VictimStatus", "UTM35Y", "UTM35X", "Lon", "Lat"]

for key, dataset in dt.items():
    if len(dt[key].columns) == 24:
        dt[key].columns = column_names_24
    if len(dt[key].columns) == 25:
        dt[key].columns = column_names_25
    if len(dt[key].columns) == 28:
        dt[key].columns = column_names_28    
        
    # Translate VehicleType Name
    replace_vehicle_name = {
        "VehicleType": {
            'Cami\xf3n <= 3,5 Tm': u'Cami\xf3n <= 3,5 Tm',
            u'Cami\ufffdn <= 3,5 Tm': u'Cami\xf3n <= 3,5 Tm',
            u'Autob\ufffds': u'Autob\xfas',
            u'Autob\ufffds articulado': u'Autob\xfas articulado'
        }
    }
    dataset = dataset.replace(replace_vehicle_name)

    replace_vehicle_name = {
        "VehicleType": {
            "Motocicleta": "Motorbike", "Ciclomotor": "Moped", "Bicicleta": "Cycle", "Turismo": "Car",
            "Furgoneta": "Van", "Autobús": "Bus", u"Tranv\xeda o tren": "Trolley or train", u'Autob\xfas': "Bus",
            "Cuadriciclo >=75cc": "Quad >=75cc", 'Cami\xc3\xb3n <= 3,5 Tm': "Truck <= 3,5 Tm",
            u'Autob\xc3\xbas articulado': "Articulated bus","Autobús articulado": "Articulated bus", "Microbus <=17 plazas": "Microbus <=17 seats",
            u'Tractocami\xf3n': "Tractor-trailer", "Todo terreno": "4x4", "Cuadriciclo <75cc": "Quad <75cc",
            u'Otros veh\xedc. a motor': "Other", "Maquinaria de obras": "Civil engineering machinery",
            "Autocar": "Bus", u'Tranv\ufffda o tren': "Trolley or train", u'Tractocami\ufffdn': "Tractor-trailer",
            u'Otros veh\ufffdc. a motor': "Other", u'Cami\ufffdn > 3,5 Tm': "Truck > 3.5 Tm", 
            "Camión > 3.5 Tm": "Truck > 3.5 Tm", u'Cami\xf3n <= 3,5 Tm': "Truck <= 3,5 Tm",
            u'Autob\xfas articulado': "Articulated bus", u'Cami\xf3n > 3,5 Tm': "Truck > 3.5 Tm"
        }
    }
    dataset = dataset.replace(replace_vehicle_name)
    
    replace_month_name = {
    "MonthName": {
        "Gener": "January", "Febrer": "February", "Març": "March", "Abril": "April", "Maig": "May",
        "Juny": "June", "Juliol": "July", "Agost": "August", "Septembre": "September", "Octubre": "October",
        "Novembre": "November", "Desembre": "December"
        }
    }
    dt[key] = dataset.replace(replace_month_name)
# Group by AccidentID
dt_group = {}
for key, dataset in dt.items():
    dt_group[key] = dataset.groupby("AccidentID").first()

# All the datasets from 2010-2016 concatenated and group by AccidentID
dt_all = pd.concat(dt_group)

# Create a Dataframe with all the missing data for each year

# First we create a dictionary like this { year : {col1: num_nans_col1, col2: num_nans_col2, ... , coln: num_nans_n}}
nan_val = {}
for key, dataset in dt.items():
    nan_val[key] = dataset.isnull().sum().to_dict()
    
# We instanciate a DataFrame object using the previous dictionary
nan_values = pd.DataFrame().from_dict(nan_val, orient='index')
display(nan_values)


accidents_per_month = dt_all.sort_values("Month").groupby(['Month','MonthName','Year']).size()
accidents_per_month = accidents_per_month.reset_index()

ax = sns.violinplot(x="MonthName", y=0,data=accidents_per_month, palette="muted")
ax.set(ylabel='Number of accidents');
ax.set_title("Number accidents per Month")
plt.show(ax)
month_lst = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
              'August', 'September', 'October', 'November', 'December']

# Accident per month
plt.xlabel("Month")
plt.xticks(range(0,12), month_lst)
plt.ylabel("Num. of accidents")
plt.title("Accidents per month")

for key, dataset in sorted(dt_group.items()):
    accidents_per_month = dataset.sort_values("Month").groupby(['Month','MonthName']).size().tolist()
    plt.plot(range(0,12),accidents_per_month,label = '%s' % key)

plt.axis([0, 12, 0, 1000])
plt.legend()
plt.show()

sns.distplot(dt_all[dt_all["VictimGender"]=="Home"]["VictimAge"].dropna(),kde=False,label="Man")
sns.distplot(dt_all[dt_all["VictimGender"]=="Dona"]["VictimAge"].dropna(),kde=False,label="Woman")
plt.legend()
plt.ylabel("Number of victims");
plt.rcParams['figure.figsize']=(16,10)

def histogram_accidents_gender_age(year):
    actual_dt = dt_group[str(year)]
    sns.distplot(actual_dt[actual_dt["VictimGender"]=="Home"]["VictimAge"].dropna(),kde=False,label="Man")
    sns.distplot(actual_dt[actual_dt["VictimGender"]=="Dona"]["VictimAge"].dropna(),kde=False,label="Woman")
    plt.ylabel("Number of victims")
    plt.legend()
    plt.show()


interact(histogram_accidents_gender_age,
         year=widgets.IntSlider(min=2010,max=2016,value=2010,description="Year:"));
         

order = dt_all['VehicleType'].value_counts().index[:10]
fig = sns.countplot(x="VehicleType",hue="MonthName", data=dt_all, order=order)
plt.setp(fig.xaxis.get_majorticklabels(), rotation=90)
fig.set_ylabel('Number of accidents')
fig.legend(loc=1)
fig.update;
order = dt_all['VehicleType'].value_counts().index[:10]
fig = sns.countplot(x="VehicleType",hue="Year", data=dt_all, order=order)
plt.setp(fig.xaxis.get_majorticklabels(), rotation=90)
fig.set_ylabel('Number of accidents')
fig.legend(loc=1)
fig.update;
def number_accidents_by_vehicle_type(year):
    order = dt_group[str(year)]['VehicleType'].value_counts().index[:10]
    fig = sns.countplot(x="VehicleType", data=dt_group[str(year)], order=order);
    plt.setp(fig.xaxis.get_majorticklabels(), rotation=90)
    fig.set_ylabel('Number of accidents')
    fig.set_title("Number of accidents by vehicle type - Year " + str(year))
    fig.update


interact(number_accidents_by_vehicle_type,
         year=widgets.IntSlider(min=2010,max=2016,value=2010,description="Year:"));
vehicle_types = dt_all["VehicleType"].unique()
bins = range(0,105,5)

heatmap = []
for vehicle_type in vehicle_types:
    actual_data = dt_all[dt_all["VehicleType"]==vehicle_type]
    hist, bins = np.histogram(actual_data["VictimAge"].dropna(),bins=bins)
    heatmap.append(hist)
arr2d = np.array(heatmap)

ax = sns.heatmap(arr2d, yticklabels=vehicle_types, xticklabels=bins[1:], annot=True, fmt="d")
ax.set_ylabel('Vehicle Type')
ax.set_xlabel('Victim Age');
ax = dt_all["StreetName"].value_counts()[:10].plot.bar()
inProj = Proj(init='epsg:23031')
outProj = Proj(init='epsg:4326')

def utm_to_latlon(row):
    lon, lat =  transform(inProj,outProj,row["UTM35X"],row["UTM35Y"])
    return (lat, lon)

def join_latlon(row):
    return (float(row["Lat"]), float(row["Lon"]))

for key, dataset in dt.items():
    
    # If exist the lat lon columns drop the UTM
    no_nan_dataset = dataset.dropna()
    
    if "Lat" in no_nan_dataset.columns and "Lon" in no_nan_dataset.columns:
        dataset["LatLon"] = no_nan_dataset.apply(lambda row: join_latlon(row), axis=1)
        # Drop UTM columns
        dataset.drop(["UTM35X", "UTM35Y"], axis=1, inplace=True, errors="ignore")
    else:
        # Create a new column LatLon applying the utm_to_latlon method to each row
        dataset["LatLon"] = no_nan_dataset.apply(lambda row: utm_to_latlon(row), axis=1)
        # Drop UTM columns
        dataset.drop(["UTM35X", "UTM35Y"], axis=1, inplace=True, errors="ignore")
# Example of an accident with multiple injuried people
display(dt["2013"][44:46])

# Group the Accidents and convert to a list
latlon = {}
for key, dataset in dt.items():
    latlon[key] = dataset.groupby("AccidentID")["LatLon"].first().dropna().tolist()
class HeatmapUpdate(object):
    
    def __init__(self, datasets, center, zoom, default_value):
        self._datasets = datasets
        self._figure = gmaps.figure(center=center, zoom_level=zoom)
        self._heatmap = gmaps.heatmap_layer(datasets[default_value])
        self._figure.add_layer(self._heatmap)
        
    def render(self):
        return display(self._figure)
    
    def update_heatmap(self, year):
        self._heatmap.locations = self._datasets[str(year)] # update the locations drawn on the heatmap
animation = HeatmapUpdate(latlon , (41.383, 2.183), 12, "2010")


def update_heatmap_aux(year):
    animation.update_heatmap(year)
    
interact(update_heatmap_aux, year=widgets.IntSlider(min=2010,max=2016,value=2010,description="Year:"));
animation.render()
