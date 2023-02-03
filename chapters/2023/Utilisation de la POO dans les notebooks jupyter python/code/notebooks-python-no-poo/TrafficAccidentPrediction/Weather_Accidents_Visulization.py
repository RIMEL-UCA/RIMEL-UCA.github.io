import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('dataset.csv')
# print(data)
# Accident count on days
accident = {}
for row in data['Accident_Date']:
    if row in accident:
        accident[row] += 1
    else:
        accident[row] = 1
        
print(accident)
hap1 = []
hap2 = []
hap_more = []
for row in accident:
    if accident[row] == 1:
        hap1.append(row)
    elif accident[row] == 2:
        hap2.append(row)
    else:
        hap_more.append(row)
        
print("happened 1 accident a day:",len(hap1), "days")
print("happened 2 accidents a day:",len(hap2), "days")
print("happened more than 2 accidents a day:",len(hap_more), "days")
weather_accident = pd.DataFrame(columns = ["Accident_Date","Precip","obs","rainfall","feel","count"])
# Precip and Obs dict
precip = {}
obs = {}
for i in range(len(data)):
    a = data['Accident_Date'][i]
    if a not in precip:
        precip[a] = data['Precip'][i]
    if a not in obs:
        obs[a] = data['obs'][i]
#fill in Accident_data and count
weather_accident['Accident_Date'] = accident.keys()
for i in range(len(weather_accident)):
    key = weather_accident['Accident_Date'][i]
    if accident[key]:
        weather_accident['count'][i] = accident[key]
        weather_accident['Precip'][i] = precip[key]
        weather_accident['obs'][i] = obs[key]
for i in range(len(weather_accident)):
    precip = weather_accident['Precip'][i]
    if precip > 2:
        weather_accident['rainfall'][i] = "violent rain"
    elif precip > 0.3:
        weather_accident['rainfall'][i] = "heavy rain"
    elif precip > 0.098:
        weather_accident['rainfall'][i] = "moderate rain"
    else:
        weather_accident['rainfall'][i] = "light rain"
for i in range(len(weather_accident)):
    temp = weather_accident['obs'][i]
    if temp > 122 :
        weather_accident['feel'][i] = "extremely hot"
    elif temp > 98.6:
        weather_accident['feel'][i] = "very hot"
    elif temp > 77:
        weather_accident['feel'][i] = "hot"
    elif temp > 68:
        weather_accident['feel'][i] = "warm"
    elif temp > 59:
        weather_accident['feel'][i] = "cool"
    elif temp > 32:
        weather_accident['feel'][i] = "cold"
    else:
        weather_accident['feel'][i] = "ice/freezes"
print(weather_accident)
import numpy as np
import matplotlib.pyplot as plt

# create a figure and axis 
fig, ax = plt.subplots() 
# count the occurrence of each class 
data = weather_accident['rainfall'].value_counts()
# get x and y data 
points = data.index 
frequency = data.values 
# create bar chart 
ax.bar(points, frequency) 
# set title and labels 
ax.set_title('RainFall vs. Accidents') 
ax.set_xlabel('Rainfall') 
ax.set_ylabel('Frequency')
# create a figure and axis 
fig, ax = plt.subplots() 
# count the occurrence of each class 
data = weather_accident['feel'].value_counts()
# get x and y data 
points = data.index 
frequency = data.values 
# create bar chart 
ax.bar(points, frequency) 
# set title and labels 
ax.set_title('Weather vs. Accidents') 
ax.set_xlabel('Feels') 
ax.set_ylabel('Frequency')
print(weather_accident)
hap1_rainfall_list = []
hap2_rainfall_list = []
a_rainfall_list = []

for i in range(len(weather_accident)):
    count =  weather_accident['count'][i]
    if count == 1:
        hap1_rainfall_list.append(weather_accident['rainfall'][i])
    elif count == 2:
        hap2_rainfall_list.append(weather_accident['rainfall'][i])
    else:
        a_rainfall_list.append(weather_accident['rainfall'][i])
import seaborn as sns
sns.countplot(hap1_rainfall_list).set_title("Most likely to have 1 accident")
sns.countplot(hap2_rainfall_list).set_title("Most likely to have 2 accidents")
sns.countplot(a_rainfall_list).set_title("Most likely to have more than 2 accident")
hap1_weather_list = []
hap2_weather_list = []
a_weather_list = []

for i in range(len(weather_accident)):
    count =  weather_accident['count'][i]
    if count == 1:
        hap1_weather_list.append(weather_accident['feel'][i])
    elif count == 2:
        hap2_weather_list.append(weather_accident['feel'][i])
    else:
        a_weather_list.append(weather_accident['feel'][i])
sns.countplot(hap1_weather_list).set_title("Most likely to have 1 accident")
sns.countplot(hap2_weather_list).set_title("Most likely to have 2 accidents")
sns.countplot(a_weather_list).set_title("Most likely to have more than 2 accidents")





