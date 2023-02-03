#Import dependencies
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import requests
import datetime
import json
from pprint import pprint
import time
import csv
import matplotlib.pylab as plb
from scipy import stats
csvpath ="US_Accidents_Dec19.csv"
accidentdata = pd.read_csv(csvpath)
accidentdata.head()
accidentdata.tail()
accidentdata.info()
newaccidentinfo = accidentdata.drop(["Source","TMC", "End_Lat","End_Lng", "Distance(mi)","Description", "Number", "Street", "Side",
                                     "Country","Airport_Code","Amenity","Bump","Crossing", "Give_Way", "Junction","No_Exit", "Railway",
                                     "Roundabout","Station", "Stop", "Traffic_Calming","Traffic_Signal","Turning_Loop","Sunrise_Sunset","Civil_Twilight",
                                      "Nautical_Twilight","Precipitation(in)","Wind_Chill(F)", "Humidity(%)", "Pressure(in)", "Wind_Direction",
                                      "City","Zipcode","Severity","End_Time","Start_Lat", "Start_Lng","County","Weather_Timestamp","Temperature(F)","Visibility(mi)","Wind_Speed(mph)"], axis=1)
newaccidentinfo.head(10)
newaccidentinfo1 = newaccidentinfo[newaccidentinfo.Start_Time.str.contains('2019')]
newaccidentinfo1.head(10)
newaccidentinfo1.info()
newaccidentinfo1.to_csv('usaccident2019.csv',index=False)
state2019 = newaccidentinfo1.groupby(['State'])\
                .count()\
                .reset_index()
state2019
cali = newaccidentinfo1[newaccidentinfo1.State.str.contains('CA')]
cali
cali.info()
cali.to_csv('CAaccident2019.csv',index=False)
import seaborn as sns
from matplotlib import pyplot
from datetime import datetime
import dateutil.parser
dt_object = cali["Start_Time"][1:30].apply(lambda x: dateutil.parser.parse(x))
dt_object
cali["Time (Military Zone)"] = cali["Start_Time"].apply(lambda x: dateutil.parser.parse(x))
cali["Time (Military Zone)"][0:4].apply(lambda timestamp: timestamp.hour)
a4_dims = (11.7, 8.27)
fig, ax = pyplot.subplots(figsize=a4_dims)
plt.title('CA Accidents in 2019')
plt.ylabel('Num of Accidents')

sns.countplot(x=cali["Time (Military Zone)"].apply(lambda timestamp: timestamp.hour))
caaccidentinfo = accidentdata.drop(["Source","TMC", "End_Lat","End_Lng", "Distance(mi)","Description", "Number", "Street", "Side",
                                     "Country","Airport_Code","Amenity","Bump","Crossing", "Give_Way", "Junction","No_Exit", "Railway",
                                     "Roundabout","Station", "Stop", "Traffic_Calming","Traffic_Signal","Turning_Loop","Sunrise_Sunset","Civil_Twilight",
                                      "Nautical_Twilight", "Wind_Direction","Severity",
                                      "City","Zipcode","End_Time","Start_Lat", "Start_Lng","County","Weather_Timestamp"], axis=1)
caaccidentinfo.head(10)
caaccidentinfo2019 =caaccidentinfo[caaccidentinfo.Start_Time.str.contains('2019')]
caaccidentinfo2019.head(10)
caaccident2019= caaccidentinfo2019[caaccidentinfo2019.State.str.contains('CA')]
caaccident2019
caaccidentinfo2019.describe()
caaccidentgg = accidentdata.drop(["Source","TMC", "End_Lat","End_Lng", "Distance(mi)","Description", "Number", "Street", "Side",
                                     "Country","Airport_Code","Amenity","Bump","Crossing", "Give_Way", "Junction","No_Exit", "Railway",
                                     "Roundabout","Station", "Stop", "Traffic_Calming","Traffic_Signal","Turning_Loop","Sunrise_Sunset","Civil_Twilight",
                                      "Nautical_Twilight", "Wind_Direction",
                                      "City","Zipcode","County","Weather_Timestamp"], axis=1)
caaccidentgg.head()
caplot2019=caaccidentgg[caaccidentgg.State.str.contains('CA')]
caplot2019.head()
newcaplot2019=caplot2019[caplot2019.Start_Time.str.contains('2019-12')]
newcaplot2019.head(20)
newcaplot2019.to_csv('CAaccident2019new.csv',index=False)
severityca = newcaplot2019.groupby(['Severity'])\
                .count()\
                .reset_index()
severityca
%matplotlib inline

plt.style.use('ggplot')
f, ax = plt.subplots(figsize=(7,7))
sns.scatterplot(x='Start_Lng', y='Start_Lat', data=newcaplot2019, hue = 'Severity', s = 50)
plt.xlabel('Longitude')
plt.ylabel('Latitude)')
plt.title('Accidents in CA by Severity')
plt.show()
plt.savefig('caseverity.png')
# Map of accidents
f, ax = plt.subplots(figsize=(7,7))
sns.scatterplot(x='Start_Lng', y='Start_Lat', data=newcaplot2019, hue = 'Severity', s = 50, alpha=.5, palette = sns.color_palette(palette ='deep',n_colors = 4))
plt.xlabel('Longitude')
plt.ylabel('Latitude)')
plt.title('Accidents in CA by Severity')
plt.show()


