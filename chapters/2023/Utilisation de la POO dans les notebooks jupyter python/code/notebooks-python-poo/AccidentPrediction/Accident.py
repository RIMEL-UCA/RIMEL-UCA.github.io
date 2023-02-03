from google.colab import drive
drive.mount('/content/gdrive')
import pandas as pd

data_sets= []
data_sets.append(pd.read_csv("/content/gdrive/My Drive/a/2013.csv", encoding="ISO-8859-1"))
data_sets.append(pd.read_csv("/content/gdrive/My Drive/a/2012.csv", encoding="ISO-8859-1"))
data_sets[1].head(4)
data_sets[1].isnull().sum()
def_rows = ['time', 'year', 'day', 'month','Road Surfce','lighting_conditions', 'Easting']
data_sets[1]['Road Surface'].unique()
data_sets[0]['Road Surface'].unique()
data_sets[1]['Lighting Conditions'].unique()
data_sets[1]['Weather Conditions'].unique()
def_rows = ['City', 'Road Surface', 'Lighting Conditions', 'Holiday', 'Weather Conditions']
data_sets[0]['Road Surface'] = data_sets[0]['Road Surface'].apply(lambda x: 'Wet / Damp' if x=='Snow' else x)
data_sets[1]['Road Surface'] = data_sets[1]['Road Surface'].apply(lambda x: 'Wet / Damp' if x=='Snow' else x)

data_sets[0]['Road Surface'] = data_sets[0]['Road Surface'].apply(lambda x: 'Dry' if x=='5' else x)
data_sets[1]['Road Surface'] = data_sets[1]['Road Surface'].apply(lambda x: 'Dry' if x=='5' else x)

data_sets[0]['Lighting Conditions'] = data_sets[0]['Lighting Conditions'].apply(lambda x: 'Darkness: no street lighting' if x=='5' else x)
data_sets[1]['Lighting Conditions'] = data_sets[1]['Lighting Conditions'].apply(lambda x: 'Daylight: street lights present' if x=='5' else x)
cities = ["Akkaraipattu", "Ambalangoda", "Ampara", "Anuradhapura", "Badulla", "Balangoda", "Bandarawela", "Batticaloa", "Beruwala", "Boralesgamuwa", "Chavakachcheri", "Chilaw", "Colombo", "Dambulla", "Dehiwala-Mount Lavinia", "Embilipitiya", "Eravur", "Galle", "Gampaha", "Gampola", "Hambantota", "Haputale", "Hatton-Dickoya", "Hikkaduwa", "Horana", "Ja-Ela", "Jaffna", "Kadugannawa", "Kaduwela", "Kalmunai", "Kalutara", "Kandy", "Kattankudy", "Katunayake", "Kegalle", "Kesbewa", "Kilinochchi", "Kinniya", "Kolonnawa", "Kuliyapitiya", "Kurunegala", "Maharagama", "Mannar", "Matale", "Matara", "Minuwangoda", "Moneragala", "Moratuwa", "Mullaitivu", "Nawalapitiya", "Negombo", "Nuwara Eliya", "Panadura", "Peliyagoda", "Point Pedro", "Polonnaruwa", "Puttalam", "Ratnapura", "Seethawakapura", "Sri Jayawardenepura", "Tangalle", "Thalawakele-Lindula", "Trincomalee", "Valvettithurai", "Vavuniya", "Wattala-Mabole", "Wattegama", "Weligama"]
import random 

def random_cities(dataset):
  k = dataset.shape[0]
  selected_cities = []
  for i in range(0, k):
    selected_cities.append(cities[random.randint(0, len(cities)-1)])
  dataset['City'] = selected_cities
  return dataset

data_sets[0] = random_cities(data_sets[0])
data_sets[1] = random_cities(data_sets[1])
data_comb = pd.concat([data_sets[0] , data_sets[1]])
df = data_comb.groupby(by=def_rows).size().to_frame('No Accidents').reset_index()
df.head()
data_comb = data_comb.drop(columns=['Easting','Northing', "Type of Vehicle"])
data_comb.to_csv("/content/gdrive/My Drive/a/DataSet_v1.0.csv", index=False)
data_sets[0].head(4)
import matplotlib.pyplot as plt
import seaborn as sns

f, ax = plt.subplots(figsize=(7, 3))
sns.countplot(y="road_surface", data=dataframe, color="c");

f, ax = plt.subplots(figsize=(7, 3))
sns.countplot(y="lighting_conditions", data=dataframe, color="c");
dataframe.groupby(by="grid_ref_easting").size()
print("grid_ref_northing bins :",len(dataframe['grid_ref_northing'].unique()))
print("grid_ref_easting bins :",len(dataframe['grid_ref_easting'].unique()))
easting_lowest = dataframe.sort_values(by='grid_ref_easting').reset_index()['grid_ref_easting'][0]
northing_lowest = dataframe.sort_values(by='grid_ref_northing').reset_index()['grid_ref_northing'][0]

easting_highest = dataframe.sort_values(by='grid_ref_easting').reset_index()['grid_ref_easting'][2432]
northing_highest = dataframe.sort_values(by='grid_ref_northing').reset_index()['grid_ref_northing'][2432]

easting_bins = [i for i in range(easting_lowest, easting_highest+4, 4)]
northing_bins = [i for i in range(northing_lowest, northing_highest+4, 4)]

dataframe['northing_binned'] = pd.cut(dataframe['grid_ref_northing'], northing_bins, include_lowest=True)
dataframe['easting_binned'] = pd.cut(dataframe['grid_ref_easting'], easting_bins, include_lowest=True)

easting_lowest
dataframe.sort_values(by='grid_ref_easting')['grid_ref_easting']
dataframe.isnull().sum()

f, ax = plt.subplots(figsize=(7, 7))
sns.heatmap(dataframe.pivot(dataframe.groupby(['grid_ref_easting', 'grid_ref_northing']).size().unstack(fill_value=0)));
pd.crosstab(dataframe.grid_ref_easting,dataframe.grid_ref_northing)

f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(dataframe.groupby(['northing_binned', 'easting_binned']).size().unstack(fill_value=0));
dataframe.sort_values(by='grid_ref_easting').reset_index()['grid_ref_easting']
easting_lowest = dataframe.sort_values(by='grid_ref_easting').reset_index()['grid_ref_easting'][0]
northing_lowest = dataframe.sort_values(by='grid_ref_northing').reset_index()['grid_ref_northing'][0]

easting_highest = dataframe.sort_values(by='grid_ref_easting').reset_index()['grid_ref_easting'][2432]
northing_highest = dataframe.sort_values(by='grid_ref_northing').reset_index()['grid_ref_northing'][2432]

easting_lowest,northing_lowest
easting_highest,northing_highest
414747- 423490
445576-449480
Latitude from 5.94851 to 9.81667 and longitude from 79.79528 to 81.84198.
def convert_northing(gri_ref_north):
  input_start = 423490
  input_end = 449480
  output_start = 5.94851
  output_end   = 9.81667
  output = output_start + ((output_end - output_start) / (input_end - input_start)) * (gri_ref_north - input_start)
  return output

def convert_easting(gri_ref_east):
  input_start = 414747
  input_end = 445576
  output_start = 79.79528
  output_end   = 81.84198
  output = output_start + ((output_end - output_start) / (input_end - input_start)) * (gri_ref_east - input_start)
  return output
dataframe['lat'] = dataframe['grid_ref_northing'].apply(convert_northing)
dataframe['long'] = dataframe['grid_ref_easting'].apply(convert_easting)
dataframe.head(4)
#Bins for lat long
7.428983-7.435636
80.601508-80.607613
lat_lowest = dataframe.sort_values(by='lat').reset_index()['lat'][0]
long_lowest = dataframe.sort_values(by='long').reset_index()['long'][0]

lat_highest = dataframe.sort_values(by='lat').reset_index()['lat'][2432]
long_highest = dataframe.sort_values(by='long').reset_index()['long'][2432]

lat_bins = []
lat = lat_lowest
while lat<=lat_highest:
  lat_bins.append(lat)
  lat+= 0.006653

long_bins = []
long = long_lowest
while long<=long_highest:
  long_bins.append(long)
  long += 0.006105

dataframe['lat_binned'] = pd.cut(dataframe['lat'], lat_bins, include_lowest=True)
dataframe['long_binned'] = pd.cut(dataframe['long'], long_bins, include_lowest=True)

dataframe.head()
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(dataframe.groupby(['lat_binned', 'long_binned']).size().unstack(fill_value=0));
dataFrame = dataframe.copy()
from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 


