import pandas as pd
from pandas import DataFrame
import numpy as np
import datetime
import random
from datetime import datetime
pd.set_option('display.max_columns', 500)
dataset = pd.read_csv('dataset.csv')
dataset.info()
dataset.shape
dataset.head(1)
dataset.keys()
data = pd.read_csv('speed_done.csv')
test = data.sample(n=3500, replace=True, random_state=1)
test.shape
test.head(1)
false_data = DataFrame(columns=['Accident_Date', 'Accident_Time', 'LONGITUDE', 'LATITUDE',
       'AStreet', 'BStreet', 'FatalInjuries', 'MajorInjuries', 'Involving',
       'Nearest_Intersection', 'SPD', 'LEN', 'ACC', 'VOL', 'F85th', 'LOCAL',
       'F50th', 'Accident'])
false_data
for i in range(0,test.shape[0] - 1):
    y = random.randint(2013, 2017)
    m = random.randint(1, 12)
    d = random.randint(1,28)
    date = str(y*10000+m*100+d)
    s_datetime = datetime.strptime(date, '%Y%m%d')
    newformat = s_datetime.strftime('%Y-%m-%d')
    time = random.randint(0,23)
    random_false = test.iloc[i]
    false_data = false_data.append({'Unnamed: 0' : i, 'Accident_Date' : newformat, 'Accident_Time' : time, 'LONGITUDE' : random_false['lng'], 'LATITUDE' : random_false['lat'],
       'AStreet' : random_false['STREET'], 'BStreet' : random_false['START'], 'FatalInjuries' : 0, 'MajorInjuries' : 0, 'Involving' : 0, 'Nearest_Intersection' : random_false['STREET'] + ' & ' + random_false['START'], 'SPD' : random_false['SPD'], 'LEN' : random_false['LEN'], 'ACC' : random_false['ACC'], 'VOL' : random_false['VOL'], 'F85th' : random_false['F85th'], 'LOCAL' : random_false['LOCAL'],
       'F50th' : random_false['F50th'], 'Accident' : 0}, ignore_index=True)
#weather = pd.read_csv('weather_clean.csv')
false_data.head(5)
#for y in range(len(weather['Date'])):
#    wea = str(int(weather['Date'][y]))
#    datetimeobject = datetime.strptime(wea,'%Y%m%d')
#    newformat = datetimeobject.strftime('%Y-%m-%d')
#    weather['Date'][y] = newformat
#weather.head(5)
#weather.to_csv('weather_clean.csv')
weather = pd.read_csv('weather_clean.csv')
weather.head(5)
weather.keys()
weather = weather.drop(['Unnamed: 0'], axis=1)
weather.keys()
false_dataset = pd.merge(left=false_data,right=weather, left_on='Accident_Date', right_on='Date')
false_dataset.info()
false_dataset.head(1)
false_dataset.keys()
false_dataset.shape
false_dataset = false_dataset.drop(columns=['Date','Time'])
false_dataset.shape
false_dataset.keys()
false_dataset.head(1)
#false_dataset_1 = false_dataset
#frame = [false_dataset, false_dataset_1, false_dataset_1,false_dataset_1,false_dataset_1]
#false_dataset = pd.concat(frame)
false_dataset.shape
false_dataset.to_csv(r'false_data.csv')
