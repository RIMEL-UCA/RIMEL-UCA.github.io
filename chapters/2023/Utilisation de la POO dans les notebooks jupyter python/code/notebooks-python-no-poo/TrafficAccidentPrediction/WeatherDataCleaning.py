import pandas as pd
import matplotlib.pyplot as plt

weather = pd.read_csv('weather.csv')

weather = weather.drop(columns = 'Station')
weather = weather.dropna(axis = 'columns', how = 'all')
avg = (weather['Air max'] + weather['min'])/2
weather['obs'] = weather['obs'].fillna(avg)
print(weather.head())
Yr2013_Date = []
Yr2013_temp = []

for x in range(4749,5114):
    Yr2013_Date.append(weather['Date'][x])
    Yr2013_temp.append(weather['obs'][x])

Yr2013 = {'Date': Yr2013_Date, 'Temperature': Yr2013_temp}
Yr2013 = pd.DataFrame(Yr2013)
print(Yr2013)

plt.scatter(Yr2013['Date'], Yr2013['Temperature'])
plt.title('Year: 2013')
plt.ylabel('Temperature(F)')
plt.xticks([])
plt.show()
Yr2014_Date = []
Yr2014_temp = []

for x in range(5114,5479):
    Yr2014_Date.append(weather['Date'][x])
    Yr2014_temp.append(weather['obs'][x])

Yr2014 = {'Date': Yr2014_Date, 'Temperature': Yr2014_temp}
Yr2014 = pd.DataFrame(Yr2014)
print(Yr2014)

plt.scatter(Yr2014['Date'], Yr2014['Temperature'])
plt.title('Year: 2014')
plt.ylabel('Temperature(F)')
plt.xticks([])
plt.show()
Yr2015_Date = []
Yr2015_temp = []

for x in range(5479,5844):
    Yr2015_Date.append(weather['Date'][x])
    Yr2015_temp.append(weather['obs'][x])

Yr2015 = {'Date': Yr2015_Date, 'Temperature': Yr2015_temp}
Yr2015 = pd.DataFrame(Yr2015)
print(Yr2015)

plt.scatter(Yr2015['Date'], Yr2015['Temperature'])
plt.title('Year: 2015')
plt.ylabel('Temperature(F)')
plt.xticks([])
plt.show()
Yr2016_Date = []
Yr2016_temp = []

for x in range(5844,6210):
    Yr2016_Date.append(weather['Date'][x])
    Yr2016_temp.append(weather['obs'][x])

Yr2016 = {'Date': Yr2016_Date, 'Temperature': Yr2016_temp}
Yr2016 = pd.DataFrame(Yr2016)
print(Yr2016)

plt.scatter(Yr2016['Date'], Yr2016['Temperature'])
plt.title('Year: 2016')
plt.ylabel('Temperature(F)')
plt.xticks([])
plt.show()
Yr2017_Date = []
Yr2017_temp = []

for x in range(6210,6575):
    Yr2017_Date.append(weather['Date'][x])
    Yr2017_temp.append(weather['obs'][x])

Yr2017 = {'Date': Yr2017_Date, 'Temperature': Yr2017_temp}
Yr2017 = pd.DataFrame(Yr2017)
print(Yr2017)

plt.scatter(Yr2017['Date'], Yr2017['Temperature'])
plt.title('Year: 2017')
plt.ylabel('Temperature(F)')
plt.xticks([])
plt.show()
Yr2018_Date = []
Yr2018_temp = []

for x in range(6575,6940):
    Yr2018_Date.append(weather['Date'][x])
    Yr2018_temp.append(weather['obs'][x])

Yr2018 = {'Date': Yr2018_Date, 'Temperature': Yr2018_temp}
Yr2018 = pd.DataFrame(Yr2018)
print(Yr2018)

plt.scatter(Yr2018['Date'], Yr2018['Temperature'])
plt.title('Year: 2018')
plt.ylabel('Temperature(F)')
plt.xticks([])
plt.show()
Yr2019_Date = []
Yr2019_temp = []

for x in range(6940,7233):
    Yr2019_Date.append(weather['Date'][x])
    Yr2019_temp.append(weather['obs'][x])

Yr2019 = {'Date': Yr2019_Date, 'Temperature': Yr2019_temp}
Yr2019 = pd.DataFrame(Yr2019)
print(Yr2019)

plt.scatter(Yr2019['Date'], Yr2019['Temperature'])
plt.title('Year: 2019')
plt.ylabel('Temperature(F)')
plt.xticks([])
plt.show()
weather.to_csv('weather_clean.csv')

