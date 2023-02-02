from google.colab import drive 
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import xlrd


import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

from datetime import datetime



train_df=pd.read_csv('/content/drive/My Drive/ML Project/train.csv');
test_df=pd.read_csv('/content/drive/My Drive/ML Project/test.csv');
submission_df=pd.read_csv('/content/drive/My Drive/ML Project/sample_submission.csv')
train_df.drop_duplicates(inplace = True)
test_df.drop_duplicates(inplace = True)
train_df

# train_df = train_df[train_df['duration']!=0]
train_df
lbl=[]
for index, row in train_df.iterrows():
  label=row['label']
  
  if(label=="correct"):
    lbl.append(1)
  elif(label=="incorrect"):
    lbl.append(0)
lbl
train_df['lb']=lbl

train_df
labels_df=pd.DataFrame(train_df['lb'])
features_df=train_df.drop(labels=['label','lb'],axis=1)
features_df

features_df
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn_pandas import CategoricalImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
# from catboost import CatBoostClassifier,Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from pprint import pprint
from math import radians, cos, sin, asin, sqrt 


def getDistance(df):
  # The math module contains a function named 
    # radians which converts from degrees to radians. 
  dist=[]
  for index, row in df.iterrows():
    lon1 = radians(row['pick_lon']) 
    lon2 = radians(row['drop_lon']) 
    lat1 = radians(row['pick_lat']) 
    lat2 = radians(row['drop_lat']) 
       
    # Haversine formula  
    dlon = lon2 - lon1  
    dlat = lat2 - lat1 
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
  
    c = 2 * asin(sqrt(a))  
     
    # Radius of earth in kilometers. Use 3956 for miles 
    r = 6371
       
    # calculate the result 
    distance=c * r
    dist.append(distance)
  
  return dist
      
      

getDistance(features_df)
features_df['dist']=getDistance(features_df)
features_df
# datetime_object = datetime.strptime(datetime_str, '%m/%d/%y %H:%M:%S')
def get_time_type(df):
  pickup_slot=[]
  both="Both"
  school="School"
  office="Office"
  free="free"
  for index, row in df.iterrows():
    pickup_datetime=row['pickup_time']
    both_start="6:00"
    both_end="9:30"
    school_start="13:30"
    school_end="14:30"
    office_start="16:30"
    office_end="19:30"

    pickup_time=datetime.strptime(pickup_datetime.split(" ")[1],'%H:%M')
    both_start_time=datetime.strptime(both_start,'%H:%M')
    both_end_time=datetime.strptime(both_end,'%H:%M')
    school_start_time=datetime.strptime(school_start,'%H:%M')
    school_end_time=datetime.strptime(school_end,'%H:%M')
    office_start_time=datetime.strptime(office_start,'%H:%M')
    office_end_time=datetime.strptime(office_end,'%H:%M')
    
    if(pickup_time> both_start_time and pickup_time<both_end_time):
      pickup_slot.append(both)
      # print(pickup_time,both)
    elif(pickup_time> school_start_time and pickup_time<school_end_time):
      pickup_slot.append(school)
      # print(pickup_time,school)
    elif(pickup_time>office_start_time and pickup_time<office_end_time):
      pickup_slot.append(office)
      # print(pickup_time,office)
    else:
      pickup_slot.append(free)
      # print(pickup_time,free)
    

    
  return pickup_slot
features_df['Time_type']=get_time_type(features_df)
features_df


def get_fare_per_distance(df):
  fare_per_dist=[]
  for index, row in df.iterrows():
    if(row['dist']==0):
     
      fare_per_dist.append(0)
    else:
      fare_per_distance=(float(row['fare'])-float(row['meter_waiting_fare']))/float(row['dist'])
      
      fare_per_dist.append(fare_per_distance)
  
  return fare_per_dist



get_fare_per_distance(features_df)
features_df['Fare_per_dist']=get_fare_per_distance(features_df)
features_df


def get_fare_per_min(df):
  fare_per_min=[]
  for index, row in df.iterrows():
    if(row['duration']==0):
     
      fare_per_min.append(0)
    else:
      fare_per_minute=(float(row['fare']))/(float(row['duration'])/60.0)

      fare_per_min.append(fare_per_minute)
   
        
    
    
  
  return fare_per_min



get_fare_per_min(features_df)
features_df['Fare_per_min']=get_fare_per_min(features_df)
features_df
features=['additional_fare','duration','meter_waiting','meter_waiting_fare','meter_waiting_till_pickup','fare','dist','Time_type','Fare_per_dist','Fare_per_min','pick_lat','pick_lon','drop_lat','drop_lon']
num_features=['additional_fare','duration','meter_waiting','meter_waiting_fare','meter_waiting_till_pickup','fare','dist','Fare_per_dist','Fare_per_min','pick_lat','pick_lon','drop_lat','drop_lon']
cat_features=['Time_type']

final_features_df=features_df[features]
final_features_df
preprocessing_steps = Pipeline([
    ('standard_scaler', StandardScaler()),
    ('simple_imputer', SimpleImputer(strategy="mean"))
])

cat_preprocessing_steps=Pipeline([
  ('cat_one_hot',OneHotEncoder())
])
preprocessor=ColumnTransformer(
    transformers = [
        ("features", preprocessing_steps, num_features),
        ("cat",cat_preprocessing_steps,cat_features)
    ],
    remainder="passthrough"
   
)
d=pd.DataFrame(preprocessor.fit_transform(final_features_df))
# d[0].std()
d
estimator=RandomForestClassifier(max_depth=67,random_state=42)

estimator
# rf_random.best_params_
full_Pipeline=Pipeline([
  ("preprocess",preprocessor),
  ("estimator",estimator)
])
labels_df
final_features_df
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 12)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]


# # Create the random grid
# random_grid = {'estimator__n_estimators': n_estimators,
#                'estimator__max_features': max_features,
#                'estimator__max_depth': max_depth,
#                'estimator__min_samples_split': min_samples_split,
#                'estimator__min_samples_leaf': min_samples_leaf,
#                'estimator__bootstrap': bootstrap}
# pprint(random_grid)
# from sklearn.model_selection import RandomizedSearchCV


# rf_random = RandomizedSearchCV(estimator = full_Pipeline, param_distributions = random_grid, n_iter = 60, cv = 3, verbose=2, random_state=42)

# rf_random.get_params()
X_train, X_eval, y_train, y_eval = train_test_split(
    final_features_df,
    labels_df,
    test_size=0.3,
    shuffle=True,
    random_state=6
)
final_features_df
np.testing.assert_array_equal(final_features_df.index.values,labels_df.index.values)
# payments_df

full_Pipeline.fit(X_train,y_train['lb'])
# rf_random.best_params_
full_Pipeline.score(X_eval,y_eval['lb'])
y_pred=full_Pipeline.predict(X_eval)
y_pred
full_Pipeline.score(X_eval,y_eval['lb'])
metrics.SCORERS.keys()
from sklearn.model_selection import cross_val_score
kfold = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
cv_results = cross_val_score(full_Pipeline, final_features_df, labels_df['lb'], cv=kfold, scoring='f1')
cv_results
cv_results.mean()
f1_score(y_eval, y_pred)
features
# predict=estimator.predict(preprocessed_test_features);
test_df['dist']=getDistance(test_df)
test_df['Time_type']=get_time_type(test_df)
test_df['Fare_per_dist']=get_fare_per_distance(test_df)
test_df['Fare_per_min']=get_fare_per_min(test_df)
test_df
# prepros_test=pd.DataFrame(preprocessor.fit_transform(test_df))
# prepros_test
t=full_Pipeline.predict(test_df[features])
t
submission_df['prediction']=t
submission_df
submission_df.to_csv("submission_7.csv",index=True)
