from google.colab import drive
drive.mount('/content/drive')
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
%matplotlib inline
df = pd.read_csv('/content/drive/My Drive/ML Project/train.csv')
df
def date_time_extractor(d):
  date_time_columns = ['drop_time', 'pickup_time']
  for column in date_time_columns:
    # d.insert(d.columns.get_loc(column), column+'_year', pd.DatetimeIndex(d[column]).year)
    # d.insert(d.columns.get_loc(column), column+'_month', pd.DatetimeIndex(d[column]).month)
    # d.insert(d.columns.get_loc(column), column+'_day', pd.DatetimeIndex(d[column]).day)
    d.insert(d.columns.get_loc(column), column+'_hour', pd.DatetimeIndex(d[column]).hour)
    d.insert(d.columns.get_loc(column), column+'_minute', pd.DatetimeIndex(d[column]).minute)
    del d[column]

date_time_extractor(df)
df
from math import sin, cos, sqrt, atan2, radians

def distance_calculator(d):
  distances = []
  for index, record in d.iterrows():
    lat1 = radians(record['pick_lat'])
    lon1 = radians(record['pick_lon'])
    lat2 = radians(record['drop_lat'])
    lon2 = radians(record['drop_lon'])
    d_lat = lat2-lat1
    d_lon = lon2-lon1

    a = sin(d_lat/2)**2 +cos(lat1)*cos(lat2)*sin(d_lon/2)**2
    c = 2*atan2(sqrt(a), sqrt(1-a))

    distances.append(round(6373*c, 2))
  del d['pick_lat']
  del d['pick_lon']
  del d['drop_lat']
  del d['drop_lon']
  return distances

distances = distance_calculator(df)
df.insert(df.columns.get_loc('fare'), 'distance', distances)
df
# df.insert(df.columns.get_loc('meter_waiting_till_pickup'), 'meter_waiting_fare_rate', df['meter_waiting_fare']/df['meter_waiting'])
# df.insert(df.columns.get_loc('distance'), 'fare_per_km', df['fare']/df['distance'])
# df.insert(df.columns.get_loc('fare_per_km'), 'speed', df['distance']*1000/df['duration'])
df.insert(df.columns.get_loc('distance'), 'fare_per_second', df['fare']/df['duration'])
df.insert(df.columns.get_loc('distance'), 'seconds_per_fare', (df['duration']-df['meter_waiting'])/df['fare'])
df
# df.insert(df.columns.get_loc('distance'), 'fare_per_dist', (df['fare']-(df['meter_waiting_fare']+df['additional_fare']))/df['distance'])
# df.insert(df.columns.get_loc('distance'), 'seconds_per_fare', (df['duration']-df['meter_waiting'])/df['fare'])
df
df.replace(to_replace=['correct', 'incorrect'], value=[1, 0], inplace=True)
df['label'].value_counts()
df = df.drop(['tripid'], axis=1)
# del df['pickup_time']
# del df['pick_lat']
# del df['pick_lon']
# del df['drop_lat']
# del df['drop_lon']
df
df=df.drop(columns=['pickup_time_hour', 'pickup_time_minute','drop_time_hour','drop_time_minute','meter_waiting_till_pickup'])  
df
# ## Correlation
# import seaborn as sns
# import matplotlib.pyplot as plt
# #get correlations of each features in dataset
# corrmat = df.corr()
# top_corr_features = corrmat.index
# plt.figure(figsize=(20,20))
# #plot heat map
# g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
X = df.columns
X = df[X[0:len(df.columns)-1]]
X
y = df['label'].values
y
# from numpy import nan
# from numpy import isnan
# from sklearn.impute import SimpleImputer

# imp_features = [['additional_fare', 'duration', 'meter_waiting', 'fare']]
# imp_features2 = [['meter_waiting_fare', 'meter_waiting_till_pickup']]

# for feature in imp_features2:
#   imputer = SimpleImputer(missing_values=nan, strategy='most_frequent')
#   X[feature] = imputer.fit_transform(X[feature].values)

# for feature in imp_features:
#   imputer = SimpleImputer(missing_values=nan, strategy='mean')
#   X[feature] = imputer.fit_transform(X[feature].values)
# X
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)
X_train
# params={
#  "learning_rate"    : [ 0.10, 0.20, 0.30, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ] ,
#  "max_depth"        : [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
#  "min_child_weight" : [ 1, 2, 3, 4, 5, 6, 7, 8, 9 ],
#  "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ],
#  "colsample_bytree" : [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]   
# }

# from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# import xgboost

# classifier=xgboost.XGBClassifier()
# random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
# random_search.fit(X,y)
# random_search.best_estimator_
# random_search.best_params_
# !pip install catboost
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

for i in range(1,20):
  l_rate= i*0.05
  model = XGBClassifier(learning_rate=l_rate)
  # model.fit(X, y)
  model.fit(X_train, y_train)
  pred=model.predict(X_test)
  print(roc_auc_score(y_test, pred))
  # 0.8367421259440911---9
model = XGBClassifier(learning_rate=0.2)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
cv_results = cross_val_score(model, X, y, cv=kfold, scoring='f1')
cv_results
cv_results.mean()
# X.columns
model = XGBClassifier(learning_rate=0.2)
model.fit(X_train, y_train)
imp=pd.DataFrame()
imp['feature']=X.columns
imp['importance']=model.feature_importances_
imp
  
# from sklearn.metrics import jaccard_similarity_score
# from sklearn.metrics import f1_score
# from sklearn.metrics import log_loss
# from sklearn.model_selection import cross_val_score
# from sklearn.neighbors import KNeighborsClassifier

# X_train, y_train = X,y

# bestScore=0.0
# accList=[]
# for k in range(1,15):
#     clf_knn = KNeighborsClassifier(n_neighbors=k,algorithm='auto')
    
#     scores = cross_val_score(clf_knn, X_train, y_train, cv=10)
#     score=scores.mean()
#     accList.append(score)
    
#     if score > bestScore:
#         bestScore=score
#         best_clf=clf_knn
#         bestK=k
# print("Best K is :",bestK,"| Cross validation Accuracy :",bestScore)
# clf_knn=best_clf
# clf_knn.fit(X_train,y_train)
# y_pred=best_clf.predict(X_train)
# trainScores={}
# trainScores['KNN-jaccard']=jaccard_similarity_score(y_train, y_pred)
# trainScores['KNN-f1-score']=f1_score(y_train, y_pred, average='weighted')
# trainScores
# plt.plot(range(1,15),accList)
# plt.xlabel('K')
# plt.ylabel('CV Accuracy')
# plt.show()
from sklearn.metrics import roc_auc_score

pred=model.predict(X_test)
roc_auc_score(y_test, pred)
test_df = pd.read_csv('/content/drive/My Drive/ML Project/test.csv')
test_X = test_df[test_df.columns]
test_X
date_time_extractor(test_X)
dist = distance_calculator(test_X)
test_X.insert(test_X.columns.get_loc('fare'), 'distance', dist)
del test_X['tripid']
# test_X.insert(test_X.columns.get_loc('meter_waiting_till_pickup'), 'meter_waiting_fare_rate', test_X['meter_waiting_fare']/test_X['meter_waiting'])

# test_X.insert(test_X.columns.get_loc('distance'), 'fare_per_km', test_X['fare']/test_X['distance'])
# test_X.insert(test_X.columns.get_loc('fare_per_km'), 'speed', test_X['distance']*1000/test_X['duration'])
test_X.insert(test_X.columns.get_loc('distance'), 'fare_per_second', test_X['fare']/test_X['duration'])
test_X.insert(test_X.columns.get_loc('distance'), 'seconds_per_fare', (test_X['duration']-test_X['meter_waiting'])/test_X['fare'])
test_X
test_X=test_X.drop(columns=['pickup_time_hour', 'pickup_time_minute','drop_time_hour','drop_time_minute','meter_waiting_till_pickup'])  
test_X
# test_X['drop_time'] = pd.to_datetime(test_X['drop_time'])
# test_X['pickup_time'] = pd.to_datetime(test_X['pickup_time'])
# imp_features = [['additional_fare','duration','meter_waiting','meter_waiting_fare','meter_waiting_till_pickup','pick_lat','pick_lon','drop_lat','drop_lon','fare']]

# for feature in imp_features:
#   imputer = SimpleImputer(missing_values=nan, strategy='mean')
#   transformed_values = imputer.fit_transform(test_df[feature].values)
# test_df
# test_df['duration'] = test_df['duration'].astype(np.float64)
# test_df['meter_waiting'] = test_df['meter_waiting'].astype(np.float64)
# test_df['meter_waiting_till_pickup'] = test_df['meter_waiting_till_pickup'].astype(np.float64)
# test_df
# test_df['label'] = pd.to_numeric(test_df['label'])
# test_df
# xgb_pred = model.predict(test_X)
# test_df['prediction'] = xgb_pred
# test_df
for i in range(1,20):
  l_rate= i*0.05
  model = XGBClassifier(learning_rate=l_rate)
  model.fit(X, y)
  xgb_pred = model.predict(test_X)
  test_df['prediction'] = xgb_pred
  print(test_df['prediction'].value_counts())
model = XGBClassifier(learning_rate=0.2)
model.fit(X, y)
xgb_pred = model.predict(test_X)
test_df['prediction'] = xgb_pred
test_df['prediction'].value_counts()
X
submission_df = test_df[['tripid','prediction']]
submission_df
submission_df.to_csv('submission_32.csv', index=True)
