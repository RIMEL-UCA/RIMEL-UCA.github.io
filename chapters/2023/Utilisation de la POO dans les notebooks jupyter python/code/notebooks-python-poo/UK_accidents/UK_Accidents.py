import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from itertools import chain, combinations
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.feature_selection import f_classif
from functools import partial
from sklearn.feature_selection import VarianceThreshold
from numpy import set_printoptions
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from pandas.plotting import scatter_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import fbeta_score, make_scorer
import scipy.stats as ss
def map_severity(sev):
    mapping = {1:'Serious', 2:'Serious', 3:'Slight'}
    return mapping[sev]
def map_speed_to_kmh(miles):
    mapping = {10: 15 , 15: 25 ,20: 30, 30:50 , 40:65 , 50:80 , 60:100 , 70: 110}
    return mapping[miles]
def map_point_of_impact(point):
    mapping = {1:'Front'}
    if (point in mapping ):
        return mapping[point]
       
    return 'Other'
def map_vehicleType(type):
    mapping = {1:'Bike', 2:'Bike', 3:'Bike', 4:'Bike', 5:'Bike', 22:'Bike', 23:'Bike', 97:'Bike'}
    if (type in mapping ):
        return mapping[type]
       
    return 'Other'
def map_day_of_week(day):
    mapping = {1: 'Sunday', 2:'Monday', 3:'Tuesday', 4:'Wednesday', 5:'Thursday', 6:'Friday',7:'Saturday'}
    return mapping[day]

def map_overturned(action):
    mapping = {2: True, 3:True, 4:True, 5:True}
    if (action in mapping ):
        return mapping[action]
    return False
def map_weather(type):
    mapping = {1:'Fine', 2:'Rain', 3:'Snow', 4:'FineWing', 5:'RainWing', 6:'SnowWing', 7:'Fog',
            8: 'Other',9: 'Other', -1: 'Other'}   
    return mapping[type]
def map_urban(type):
    mapping = {1: True, 2:False, 3:False}
    
    return mapping[type]
def map_darkness(type):
    mapping = {1: False}
    if (type in mapping ):
        return mapping[type]
    
    return True
def map_gender(type):
    mapping = {1: False, 2:True, -1:False}
        
    return mapping[type]
def combineTwoVariables(i,s2):       
    s1 = str(i)
    s3 = s1 + " " +s2
    return s3
def percentage_serious_accidents(group):
    serious_count = group[group.Severity=='Serious'].count()
    total_count = group.shape[0]
    
    result = int((serious_count/total_count) * 100)   
    return result   
def dictionaryFeatureValueToPercentageSeriuosSeverity(dataFrame, feature):

    data = dataFrame.groupby([feature])[['Severity']].apply(percentage_serious_accidents).reset_index()
    data = data.set_index([feature])
    return data.T.to_dict('list')    
def features_to_model(X):  
    
    return X[['Severity',
           'IsPedestrian','IsBike','IsFemale',
           'Number_of_Vehicles', 'Number_of_Casualties',
           'Speed_Risk', 'Road_Line_Risk',
           'Double_Front_Collision','IsOverturned',
           'Is_Urban', 'Is_Darkness']]
    
class DataPreprocessingPipeline:
    def __init__(self):
        self.accidents = {}
        
    def clean_data(self): 
        # Longitude & Latitude
        self.accidents.dropna(subset=['Longitude', 'Latitude'],inplace=True)       
       
        
    def convert_features(self):
    
        # Severity
        self.accidents['Severity'] = self.accidents['Accident_Severity'].apply(map_severity)
     
        # Pedestrians
        pedestrianData = uk_casualties[uk_casualties.Casualty_Class == 3]
        self.accidents = self.accidents.assign(IsPedestrian=self.accidents['Accident_Index'].isin(pedestrianData['Accident_Index']).astype(int))
     
        # Gender
        uk_casualties['Is_Female'] = uk_casualties.Sex_of_Casualty.apply(map_gender)
        femaleData = uk_casualties[uk_casualties.Is_Female == True]
        self.accidents = self.accidents.assign(IsFemale=self.accidents['Accident_Index'].isin(femaleData['Accident_Index']).astype(int))
        
        # Bikes
        uk_vehicles['IsBike'] =uk_vehicles['Vehicle_Type'].apply(map_vehicleType)
        bikesData = uk_vehicles[uk_vehicles.IsBike == 'Bike']
        self.accidents = self.accidents.assign(IsBike=self.accidents['Accident_Index'].isin(bikesData['Accident_Index']).astype(int))
    
        # Road Number        
        self.accidents['Road_Number'] = self.accidents['1st_Road_Number']
        self.accidents['Road_Number'] = self.accidents['Road_Number'].replace(10000, 0)

        self.accidents['LSOA'] = self.accidents['LSOA_of_Accident_Location']
        self.accidents['LSOA'].fillna('NoLSOA', inplace=True)        
       
        self.accidents['Road_Line'] = self.accidents.apply(lambda x: combineTwoVariables (x.Road_Number, x.LSOA),axis=1)
        dictRoadLineToPerc = dictionaryFeatureValueToPercentageSeriuosSeverity(self.accidents,'Road_Line')
        self.accidents['Road_Line_Risk'] = self.accidents['Road_Line'].map(dictRoadLineToPerc)
        self.accidents['Road_Line_Risk'] = self.accidents.Road_Line_Risk.apply(lambda x: x[0])  
    
        # Speed Limit        
        self.accidents['Speed_Limit_Kmh'] = self.accidents.Speed_limit.apply(map_speed_to_kmh)
        self.accidents['Speed_Risk'] = self.accidents.Speed_Limit_Kmh.apply(lambda x: x == 100)
        self.accidents['Speed_Risk']  = (self.accidents['Speed_Risk']  == True ).astype(int)
        
        # Point of impact
        uk_vehicles['Point_Of_Impact'] = uk_vehicles['1st_Point_of_Impact'].apply(map_point_of_impact)
        frontAccidenceData = uk_vehicles[uk_vehicles.Point_Of_Impact == 'Front']
        doubleFrontData = frontAccidenceData[frontAccidenceData.duplicated(['Accident_Index'])]
        self.accidents['Double_Front_Collision'] = self.accidents['Accident_Index'].isin(doubleFrontData['Accident_Index'])
         
        # Overturned
        uk_vehicles['Is_Overturned'] = uk_vehicles['Skidding_and_Overturning'].apply(map_overturned)
        overTurnData = uk_vehicles[uk_vehicles.Is_Overturned == True]
        self.accidents = self.accidents.assign(IsOverturned=self.accidents['Accident_Index'].isin(overTurnData['Accident_Index']).astype(int))
        
        # Urban or Rural area
        self.accidents['Is_Urban'] = self.accidents.Urban_or_Rural_Area.apply(map_urban)
        
        # Light conditions
        self.accidents['Is_Darkness'] = self.accidents.Light_Conditions.apply(map_darkness)
       
                
    def transform(self, X):
        
        self.accidents = X
        
        self.clean_data()
        self.convert_features()
       
        return features_to_model(self.accidents)
    
   
def print_metrics(y_tr,y_predict,true_def):
    print(confusion_matrix(y_true=y_tr,y_pred=y_predict,labels=true_def))
    tn, fp, fn, tp = confusion_matrix(y_true=y_tr,y_pred=y_predict,labels=true_def).ravel()
    print ('TN (Slight   Slight   )',tn)
    print ('FP (Slight   Serious  )',fp)
    print ('FN (Serious  Slight   )',fn)
    print ('TP (Serious  Serious  )',tp)
    print(classification_report(y_true=y_tr,y_pred=y_predict))
   
pd.set_option('display.max_columns', None)
uk_accidents = pd.read_csv('UK_Accidents.csv')
uk_accidents.shape
uk_accidents.head(2)
uk_casualties = pd.read_csv('UK_Casualties.csv')
uk_casualties.head(2)
uk_casualties.shape
uk_vehicles = pd.read_csv('UK_Vehicles.csv')
uk_vehicles.head(2)
uk_vehicles.shape
uk_accidents.info()
train, test = train_test_split(uk_accidents, test_size=0.4,random_state = 54321)
test, valid = train_test_split(test, test_size=0.5)
print ("train shape:      ", train.shape)
print ("validation shape: ", valid.shape)
print ("test shape:       ", test.shape)
accidents = train
plt.figure(figsize=(20,8))
sns.countplot(x=accidents.Accident_Severity,data=accidents)
plt.xlabel("Accident Severity")
plt.show()
# combine Fatal and Serious accidents
accidents['Severity'] = accidents['Accident_Severity'].apply(map_severity)
plt.figure(figsize=(20,8))
sns.countplot(x=accidents.Severity,data=accidents)
plt.xlabel("Accident Severity")
plt.show()
seriousData = accidents[accidents.Severity == 'Serious']
slightData = accidents[accidents.Severity == 'Slight']
seriousCount = seriousData.shape[0]
slightCount = slightData.shape[0]
print ('Number of serious accidents   : ', seriousCount)
print ('Number of slight accidents    : ', slightCount)
print ('% seriuos from total accidents : ', seriousCount/accidents.shape[0])
# Road number 1 in UK. Cross almost all country includes London
road1 = accidents[accidents['1st_Road_Number']==1]
road1.shape
road1.head(2)
fig = plt.figure(figsize=(14,8))
sns.scatterplot(x=road1.Latitude, y=road1.Longitude, data=road1, hue='Severity', legend='full',palette='bwr', alpha=0.5)
accidents.dropna(subset=['Longitude', 'Latitude'],inplace=True)
accidents['Weather'] = accidents['Weather_Conditions'].apply(map_weather)
accidents.Weather.value_counts()
pd.crosstab(index=accidents['Weather'],values=accidents['Severity'], columns=accidents['Severity'],
aggfunc='count', normalize='index').plot.bar(title='All accidents by Weather',figsize=(10,5)).fontsize=10
slightData = accidents[accidents.Severity == 'Slight']
plt.figure(figsize=(20,8))
sns.countplot(x=slightData.Weather ,data=slightData)
plt.xlabel("Slight Accidents")
plt.show()
seriousData = accidents[accidents.Severity == 'Serious']
plt.figure(figsize=(20,8))
sns.countplot(x=seriousData.Weather ,data=seriousData)
plt.xlabel("Serious Accidents")
plt.show()
# key: weather type
# value: percentage of critical accidents
print (dictionaryFeatureValueToPercentageSeriuosSeverity(accidents,'Weather'))
# Relation Darkness (True) and DayLight
accidents['Is_Darkness'] = accidents.Light_Conditions.apply(map_darkness)
accidents.Is_Darkness.value_counts()
seriousData = accidents[accidents.Severity == 'Serious']
seriousData.Is_Darkness.value_counts()
seriousData.shape
numSerAcc = seriousData[seriousData.Is_Darkness == True].shape[0]
print ("% serious accidents while darkness: " , numSerAcc / seriousData.shape[0])
# convert day of week to categorical feature
accidents['Day_Of_Week'] = accidents.Day_of_Week.apply(map_day_of_week)
# percentege of critical accidents per day
print (dictionaryFeatureValueToPercentageSeriuosSeverity(accidents,'Day_Of_Week'))
# convert speed rom miles/h to Km/h
accidents['Speed_Limit_Kmh'] = accidents.Speed_limit.apply(map_speed_to_kmh)
accidents.Speed_Limit_Kmh.value_counts()
# key: speed limit of the road
# value: percentage of critical accidents
print (dictionaryFeatureValueToPercentageSeriuosSeverity(accidents,'Speed_Limit_Kmh'))
# create a new column Speed_Risk on accidents dataframe
accidents['Speed_Risk'] = accidents.Speed_Limit_Kmh.apply(lambda x: x == 100)
accidents['Speed_Risk']  = (accidents['Speed_Risk']  == True ).astype(int)
# Relation between urban (True) and rural areas
accidents['Is_Urban'] = accidents.Urban_or_Rural_Area.apply(map_urban)
accidents.Is_Urban.value_counts()
urbanData = accidents[accidents.Is_Urban == True]
numSerAcc = urbanData[urbanData.Severity == 'Serious'].shape[0]
print ("% serious accidents in urban area: " , numSerAcc / urbanData.shape[0])
ruralData = accidents[accidents.Is_Urban == False]
numSerAcc = ruralData[ruralData.Severity == 'Serious'].shape[0]
print ("% serious accidents in rural area: " , numSerAcc / ruralData.shape[0])
# Road Number        
accidents['Road_Number'] = accidents['1st_Road_Number']
accidents.Road_Number.value_counts()
# road1 cross 110 administrative areas, so we can have 110 road lines with their own statistics
road1['District'] = road1['Local_Authority_(District)']
road1.District.value_counts()
accidents['LSOA'] = accidents['LSOA_of_Accident_Location']
road1['LSOA'] = road1['LSOA_of_Accident_Location']
# road 1 can be separated to 659 lines
road1.LSOA.value_counts()
# dont drop unknown but replace with demy info
accidents['Road_Number'].replace({0:10000},inplace = True)
accidents['LSOA'].fillna('A09999999', inplace=True)
accidents['Road_Line'] = accidents.apply(lambda x: combineTwoVariables (x.Road_Number, x.LSOA),axis=1)
dictRoadLineToPerc = dictionaryFeatureValueToPercentageSeriuosSeverity(accidents,'Road_Line')
accidents['Road_Line_Risk'] = accidents['Road_Line'].map(dictRoadLineToPerc)
accidents['Road_Line_Risk'] = accidents.Road_Line_Risk.apply(lambda x: x[0])  
plt.figure(figsize=(20,8))
sns.countplot(x=seriousData.Number_of_Casualties ,data=seriousData)
plt.xlabel("Number of casualties in serious accident")
# key: number of casualties
# value: percentage of critical accidents
print (dictionaryFeatureValueToPercentageSeriuosSeverity(accidents,'Number_of_Casualties'))
uk_casualties.head(2)
# total number of casualties in dateset (~12% of total)
uk_casualties[uk_casualties.Casualty_Class == 3].shape[0]
# create help dataframe pedestrian data
uk_casualties['Severity'] = uk_casualties['Casualty_Severity'].apply(map_severity)
pedestrianData = uk_casualties[uk_casualties.Casualty_Class == 3]
pedestrianData.Severity.value_counts()
# define a new column IsPedestrian on accidents dataframe
accidents = accidents.assign(IsPedestrian=accidents['Accident_Index'].isin(pedestrianData['Accident_Index']).astype(int))
uk_casualties['Is_Female'] = uk_casualties.Sex_of_Casualty.apply(map_gender)
# Relations between Male and Female(True) in all accidents
uk_casualties.Is_Female.value_counts()
# create help dataframe female data
femaleData = uk_casualties[uk_casualties.Is_Female == True]
femaleData.Severity.value_counts()
# define a new column IsFemale on accidents dataframe
accidents = accidents.assign(IsFemale=accidents['Accident_Index'].isin(femaleData['Accident_Index']).astype(int))
accidents.Number_of_Vehicles.value_counts()
plt.figure(figsize=(20,8))
sns.countplot(x=seriousData.Number_of_Vehicles ,data=seriousData)
plt.xlabel("Number of vehicles in serious accident")
uk_vehicles.head()
uk_vehicles['IsBike'] = uk_vehicles['Vehicle_Type'].apply(map_vehicleType)
uk_vehicles.head()
# total number of bikes
bikesData = uk_vehicles[uk_vehicles.IsBike == 'Bike']
bikesData.shape
# define a new column isBike on accidents dataframe
accidents = accidents.assign(IsBike=accidents['Accident_Index'].isin(bikesData['Accident_Index']).astype(int))
accidents.head(2)
accidents[accidents.IsBike==1].Severity.value_counts()
uk_vehicles.head(2)
uk_vehicles['Point_Of_Impact'] = uk_vehicles['1st_Point_of_Impact'].apply(map_point_of_impact)
# half of car with Front point of impact.
# but we need to find minimum two cars from same accidents
uk_vehicles['Point_Of_Impact'].value_counts()
# define data with car with front collision
frontAccidenceData = uk_vehicles[uk_vehicles.Point_Of_Impact == 'Front']
# in this data get duplicates - it will cars with double front collision (243K cars)
doubleFrontData = frontAccidenceData[frontAccidenceData.duplicated(['Accident_Index'])] 
doubleFrontData.shape

# define a new column Double_Front_Collision on accident dataframe
accidents['Double_Front_Collision'] = accidents['Accident_Index'].isin(doubleFrontData['Accident_Index'])
accidents.Double_Front_Collision.value_counts()
doubleFrontAccident = accidents[accidents.Double_Front_Collision==True]
doubleFrontAccident.Severity.value_counts()
uk_vehicles['Is_Overturned'] = uk_vehicles['Skidding_and_Overturning'].apply(map_overturned)
# Number of overturned cars
uk_vehicles['Is_Overturned'].value_counts()
# add a new column IsOverturned to accidents dataframe
overTurnData = uk_vehicles[uk_vehicles.Is_Overturned == True]
accidents = accidents.assign(IsOverturned=accidents['Accident_Index'].isin(overTurnData['Accident_Index']).astype(int))
accidents.head(2)
accidents.IsOverturned.value_counts()
overturned = accidents[accidents.IsOverturned==True]
overturned.Severity.value_counts()
# data from EDA 
accidentsToModel = features_to_model(accidents)

# Init pipeline
pipeline = DataPreprocessingPipeline()
trainToModel = pipeline.transform(train)
print("Compare data from EDA and pipeline transfomation.Result is: ", trainToModel.equals(accidentsToModel))

accidentsToModel.head()
trainToModel.head()
X_train = trainToModel.drop('Severity', axis=1)
y_train = trainToModel['Severity']
X_test = trainToModel.drop('Severity', axis=1)
y_train = trainToModel['Severity']
validToModel = pipeline.transform(valid)
X_valid = validToModel.drop('Severity', axis=1)
y_valid = validToModel['Severity']
testToModel = pipeline.transform(test)
X_test = testToModel.drop('Severity', axis=1)
y_test = testToModel['Severity']
"""data_true = trainToModel[trainToModel.Severity == 'Serious']
data_false = trainToModel[trainToModel.Severity == 'Slight'].sample(data_true.shape[0])
balanced_train = pd.concat([data_false,data_true])
X_train = balanced_train.drop('Severity', axis=1)
y_train = balanced_train['Severity']
"""

selector = SelectKBest(f_classif, k = 10)
X_new = selector.fit_transform(X_train, y_train)
names = X_train.columns.values[selector.get_support()]
scores = selector.scores_[selector.get_support()]
names_scores = list(zip(names, scores))
ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True])
print(ns_df_sorted)
"""parameters={'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}
clf_tree=DecisionTreeClassifier()
clf=GridSearchCV(clf_tree,parameters)
best_model = clf.fit(X_train,y_train)
# View best hyperparameters
print('min_samples_split:', best_model.best_estimator_.get_params()['min_samples_split'])
print('max_depth:', best_model.best_estimator_.get_params()['max_depth'])
min_samples_split: 10
max_depth: 9
    """
true_def = ["Slight","Serious"]
model = DecisionTreeClassifier(class_weight={"Serious": 2, "Slight": 1 },min_samples_split=10,max_depth=9) #2:1
model.fit(X_train, y_train)
y_predict = model.predict(X_train)
print_metrics(y_train,y_predict,true_def)
y_predict = model.predict(X_valid)
print_metrics(y_valid,y_predict,true_def)
y_predict = model.predict(X_test)
print_metrics(y_test,y_predict,true_def)
modelRF = RandomForestClassifier(class_weight={'Serious': 2, 'Slight': 1}, n_estimators=100, min_samples_split=10,max_depth=9)

modelRF.fit(X_train, y_train)
y_predict = modelRF.predict(X_train)
print_metrics(y_train,y_predict,true_def)
y_predict = modelRF.predict(X_test)
print_metrics(y_test,y_predict,true_def)
modelCust = DecisionTreeClassifier(class_weight={"Serious": 5, "Slight": 9 },min_samples_split=10,max_depth=9) #2:1
modelCust.fit(X_test, y_test)
y_predict = modelCust.predict(X_test)
print_metrics(y_test,y_predict,true_def)
