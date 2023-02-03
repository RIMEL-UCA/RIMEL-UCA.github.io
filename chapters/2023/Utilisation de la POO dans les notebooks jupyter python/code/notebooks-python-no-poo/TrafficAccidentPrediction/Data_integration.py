import pandas as pd
import matplotlib.pyplot as plt

weather = pd.read_csv('weather_clean.csv')
acc_speed = pd.read_csv('accident_speed.csv')
print(acc_speed.head(20))
print(weather.head())
date_split = acc_speed['Date'].str.split("T",n=1,expand=True)
acc_speed['Accident_Date']= date_split[0]
acc_speed['Accident_Time']= date_split[1]
acc_speed = acc_speed.drop(columns='Date')
acc_speed = acc_speed[['LONGITUDE','LATITUDE','Accident_Date','Accident_Time','AStreet','BStreet','FatalInjuries','MajorInjuries','Involving','Nearest_Intersection','SPD','LEN','ACC','VOL','F85th','LOCAL','F50th']]
acc_speed['Accident_Time'] = acc_speed['Accident_Time'].fillna("unknown")
print(acc_speed.head())
from datetime import datetime

for x in range(10):
    date = acc_speed['Accident_Date'][x]
    datetimeobject = datetime.strptime(date,'%Y%m%d')
    newformat = datetimeobject.strftime('%Y-%m-%d')
    acc_speed['Accident_Date'][x] = newformat
print(acc_speed.head(20))
for y in range(len(weather['Date'])):
    wea = str(int(weather['Date'][y]))
    datetimeobject = datetime.strptime(wea,'%Y%m%d')
    newformat = datetimeobject.strftime('%Y-%m-%d')
    weather['Date'][y] = newformat
print(weather)
print(weather.info())
print(acc_speed.info())
dataset = pd.merge(left=weather,right=acc_speed, left_on='Date', right_on='Accident_Date')
print(dataset)
dataset = dataset.drop(columns=['Date','Time'])
print(dataset)
dataset = dataset[['Accident_Date','Accident_Time','LONGITUDE','LATITUDE','AStreet','BStreet',
                                  'FatalInjuries','MajorInjuries','Involving','Nearest_Intersection','SPD','LEN',
                                  'ACC','VOL','F85th','LOCAL','F50th','Precip','Air max','min','obs']]
print(dataset)
dataset.to_csv('dataset.csv')
import pandas as pd
dataset_v6_temp = pd.read_csv('dataset_v6_temp.csv')
print(dataset_v6_temp)
from sklearn.utils import shuffle

dataset_v6 = shuffle(dataset_v6_temp[1:])
print(dataset_v6)
dataset_v6.to_csv('dataset_v6.csv')
