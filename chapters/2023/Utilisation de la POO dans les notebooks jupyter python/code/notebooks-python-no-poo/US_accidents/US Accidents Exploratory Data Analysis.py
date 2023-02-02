import pandas as pd
df = pd.read_csv(r'C:\Users\user\Desktop\dataset\US\US_Accident.csv')
# df.head()
df.columns
len(df.columns)
len(df)
df.info()
df.describe()
missing_percentages = df.isna().sum().sort_values(ascending=False)/len(df)
missing_percentages
list(missing_percentages)
missing_percentages[missing_percentages != 0].plot(kind='barh');
df.columns
df.City
cities = df.City.unique()
len(cities)
cities_by_accident= df.City.value_counts()
cities_by_accident
cities_by_accident[:20]
'New York' in df.City
'NY' in df.State
cities_by_accident[:20].plot(kind='barh')
import seaborn as sns
sns.set_style("darkgrid")
sns.distplot(cities_by_accident);
high_accident_cities = cities_by_accident[cities_by_accident>=1000]
low_accident_cities = cities_by_accident[cities_by_accident<1000]
len(high_accident_cities)/len(cities)
sns.distplot(high_accident_cities);
sns.distplot(low_accident_cities);
sns.histplot(cities_by_accident, log_scale=True);
cities_by_accident[cities_by_accident == 1]
df.Start_Time
df.Start_Time[0]
type(df.Start_Time[0])
df.Start_Time = pd.to_datetime(df.Start_Time)
df.Start_Time[0]
type(df.Start_Time[0])
df.Start_Time[1].hour
df.Start_Time.dt.hour
sns.distplot(df.Start_Time.dt.hour, bins=24, kde=False, norm_hist=True);
sns.distplot(df.Start_Time.dt.dayofweek, bins=7, kde=False, norm_hist=True);
sundays_start_time = df.Start_Time[df.Start_Time.dt.dayofweek == 6]
sns.distplot(sundays_start_time.dt.hour, bins=24, kde=False, norm_hist=True)
monday_start_time = df.Start_Time[df.Start_Time.dt.dayofweek == 0]
sns.distplot(monday_start_time.dt.hour, bins=24, kde=False, norm_hist=True)
df.Start_Time.dt.year
df_2019 =df[df.Start_Time.dt.year == 2017]
sns.distplot(df_2019.Start_Time.dt.month, bins=12, kde=False, norm_hist=True)
df.Start_Lat[0]
sample_df = df.sample(int(0.1 * len(df)))
sns.scatterplot(x=sample_df.Start_Lng, y=sample_df.Start_Lat, size=0.001);
!pip install folium
import folium
lat, lon = df.Start_Lat[0], df.Start_Lng[0]
lat, lon
for x in df[['Start_Lat', 'Start_Lng']].sample(100).iteritems():
    print(x[1])
from folium.plugins import HeatMap
list(zip(list(df.Start_Lat), list(df.Start_Lng)))[0]
# map = folium.Map()
# HeatMap(zip(list(df.Start_Lat), list(df.Start_Lng))).add_to(map)
# map

