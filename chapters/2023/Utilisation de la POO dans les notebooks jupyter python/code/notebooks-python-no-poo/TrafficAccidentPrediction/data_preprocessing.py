import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import array
speed = pd.read_csv("Speed_Pub.csv")
speed.keys()
speed = speed.drop(["OBJECTID","ROUTE","DATE","on_hold", "CD", "PD", "GlobalID","reason_for_on_hold", "Comment", "GlobalID", "Jurisdiction", "NEW", "MEAN_SPD","BGN_PACE","PRCNT_PACE","Shape__Length","RATE","TYPE"],axis=1)
speed.head(10)
speed.keys()
speed.sort_values('VOL', ascending= False).iloc[:10,:]
import requests
import googlemaps
import os

gmaps = googlemaps.Client(key='AIzaSyAbDfjZYGZhSPC60ZxAfYEm1yKxV77zQfw')

speed['lat'] = ""
speed['lng'] = ""

speed['intersection1'] = speed['STREET'] + ' & ' + speed['START'] +', San Jose'
speed['intersection2'] = speed['START']  + ' & ' + speed['STREET'] +', San Jose'
speed['intersection3'] = speed['STREET'] + ' & ' + speed['END_'] +', San Jose'
speed['intersection4'] = speed['END_'] + ' & ' + speed['STREET'] +', San Jose'

a = 0
for i in range(0,speed.shape[0]):
    item = speed.iloc[i]
    address0 = item.intersection1
    address1 = item.intersection3
    geocode_result0 = gmaps.geocode(address0)
    geocode_result1 = gmaps.geocode(address1)
    if len(geocode_result0) != 0:
        speed.at[i, 'lat'] = geocode_result0[0]["geometry"]["location"]["lat"]
        speed.at[i, 'lng']  = geocode_result0[0]["geometry"]["location"]["lng"]
        a = a + 1
    elif len(geocode_result1) != 0:
        speed.at[i, 'lat'] = geocode_result1[0]["geometry"]["location"]["lat"]
        speed.at[i, 'lng'] = geocode_result1[0]["geometry"]["location"]["lng"]
        a = a + 1

speed.head(10)
    
speed.info()
speed['lat'] = speed.lat.convert_objects(convert_numeric=True)
speed['lng'] = speed.lng.convert_objects(convert_numeric=True)
speed.info()
value85  = 1.1* speed['SPD']
value50 = speed['SPD']
speed['F85th'] = value85.where(speed['F85th'] == np.nan, speed['F85th'])
speed['F50th'] = value50.where(speed['F50th'] == np.nan, speed['F50th'])
speed[speed.isnull().any(axis=1)].head()
speed['F85th'].fillna(1.1*speed['SPD'], inplace = True)
speed['F50th'].fillna(speed['SPD'], inplace = True)
speed['SPD']
speed.info()
plt.boxplot(speed["lat"])
quartile_1, quartile_3 = np.percentile(speed["lat"], [25, 75])
print(min(speed["lat"]))
iqr = quartile_3 - quartile_1
lower_bound = quartile_1 - (iqr * 1.5)
upper_bound = quartile_3 + (iqr * 1.5)
np.where((speed["lat"] > upper_bound) | (speed["lat"] < lower_bound))
plt.boxplot(speed["lng"])
quartile_1, quartile_3 = np.percentile(speed["lng"], [25, 75])
iqr = quartile_3 - quartile_1
lower_bound = quartile_1 - (iqr * 1.5)
upper_bound = quartile_3 + (iqr * 1.5)
np.where((speed["lng"] > upper_bound) | (speed["lng"] < lower_bound))
speed = speed.dropna()
speed.info()
speed.plot.scatter(x = "lng", y = "lat", alpha = 0.4, c = speed["SPD"], label = "traffic", cmap=plt.get_cmap("jet"), figsize = (10, 7))
plt.legend()
plt.show()
speed.head(10)
plt.hist(speed.SPD, bins=50)
plt.show()
speed.to_csv(r'speed_done.csv')
accident = pd.read_csv("crash.csv")
accident.shape
accident = accident.drop(["AccidentId","ESRI_OID","GlobalID","Fatal_MajorInjuries"],axis=1)
accident = accident.rename(columns={'X': 'LONGITUDE', 'Y': 'LATITUDE','AccidentDateTime':'Date','AStreetNameAndSuffix':'AStreet','BStreetNameAndSuffix':'BStreet'})
for i in range(0,10):
    date = accident.iloc[i].Date
    accident.at[i, 'Date'] = date[0:10].replace('-','')
accident.head(10)
accident["Involving"].value_counts()
accident = accident.replace("Motorist","0")
accident = accident.replace("Pedestrian","1")
accident = accident.replace("Bicyclist","2")
accident["Involving"].value_counts()
dataset = pd.DataFrame(columns = ['LONGITUDE', 'LATITUDE', 'Date', 'AStreet', 'BStreet', 'FatalInjuries',
       'MajorInjuries', 'Involving', 'Nearest_Intersection', 'SPD', 'LEN', 'ACC', 'VOL', 'F85th', 'LOCAL',
       'F50th'])
num = 0
#for i in range(0,accident.shape[0]):
for i in range(0,accident.shape[0]):
    A = accident.iloc[i].Nearest_Intersection
    B = accident.iloc[i].AStreet
    C = accident.iloc[i].BStreet
    lat = accident.iloc[i].LATITUDE
    lng = accident.iloc[i].LONGITUDE
    i1 = speed.loc[speed["intersection1"].isin([A]) | speed["intersection2"].isin([A]) | speed["intersection3"].isin([A]) | speed["intersection4"].isin([A])]
    i2 = speed.loc[speed["STREET"].isin([B]) | speed["START"].isin([B]) | speed["END_"].isin([B])]
    i3 = speed.loc[speed["STREET"].isin([C]) | speed["START"].isin([C]) | speed["END_"].isin([C])]
    a = accident.iloc[i][0:9]
    if i1.shape[0] != 0:
        b = i1.iloc[0][3:10]
        c = pd.concat([a, b], sort=False,)
        num = num + 1
    elif i2.shape[0] != 0:
        arr = []
        for n in range(0, i2.shape[0]):
            lat1 = i2.iloc[n].lat
            lng1 = i2.iloc[n].lng
            dis = abs(lat - lat1) + abs(lng - lng1)
            arr.append(dis)
        arr.index(min(arr))
        b = i2.iloc[arr.index(min(arr))][3:10]
        c = pd.concat([a, b], sort=False,)
        num = num + 1
    elif i3.shape[0] != 0:
        arr = []
        for m in range(0, i3.shape[0]):
            lat2 = i3.iloc[m].lat
            lng2 = i3.iloc[m].lng
            dis = abs(lat - lat2) + abs(lng - lng2)
            arr.append(dis)
        arr.index(min(arr))
        b = i3.iloc[arr.index(min(arr))][3:10]
        c = pd.concat([a, b], sort=False,)
    else: c = a
    dataset = dataset.append(c, ignore_index=True)
dataset.info()
dataset = dataset.dropna()
dataset.info()
dataset.shape
dataset.head(10)
dataset.to_csv(r'accident_speed.csv')
