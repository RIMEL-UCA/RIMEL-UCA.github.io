import pandas as pd
from pandas import DataFrame
import numpy as np
import datetime
import random
from datetime import datetime
pd.set_option('display.max_columns', 500)
false_data = pd.read_csv('false_data.csv')
false_data.info()
dataset = pd.read_csv('dataset.csv')
dataset.shape
dataset['Accident'] = 1
dataset.shape
dataset.head(10)
dataset.keys()
false_data.keys()
false_data = false_data[['Unnamed: 0', 'Accident_Date', 'Accident_Time', 'LONGITUDE', 'LATITUDE',
       'AStreet', 'BStreet', 'FatalInjuries', 'MajorInjuries', 'Involving',
       'Nearest_Intersection', 'SPD', 'LEN', 'ACC', 'VOL', 'F85th', 'LOCAL',
       'F50th', 'Precip', 'Air max', 'min', 'obs', 'Accident']]
#false_data.rename(columns={'LONGITUDE':'LATITUDE', 'LATITUDE':'LONGITUDE'},inplace=True)
false_data.head(5)
dataset.head(5)
frame = [dataset, false_data]
final_data = pd.concat(frame)
final_data.shape
final_data = final_data.reset_index()
final_data.info()
#final_data = final_data.sample(frac=1)
for i in range(0, final_data.shape[0]):
    time = final_data.iloc[i]['Accident_Date'].split('-')
    year = int(time[0])
    month = int(time[1])
    day = int(time[2])
    final_data.at[i,'year'] = year
    final_data.set_value(i, 'year', year)
    final_data.at[i,'month'] = month
    final_data.at[i,'day'] = day
    hour = final_data.iloc[i]['Accident_Time']
    if (type(hour) == int) != 1:
        if (hour != "unknown"):
            final_data.at[i, 'hour'] = int(final_data.iloc[i]['Accident_Time'][0:2])
        else:
            final_data.at[i, 'hour'] = random.randint(0,23)
    else:
        final_data.at[i, 'hour'] = hour
final_data.shape
#final_data = final_data.reset_index()
#final_data[final_data.index.duplicated()]
final_data.to_csv(r'dataset_v2.csv')
final_data.keys()
final_data = final_data.drop(['Unnamed: 0', 'Accident_Date', 'Accident_Time', 'AStreet', 'BStreet', 'Nearest_Intersection'], axis=1)
final_data.keys()
final_data.shape
final_data = final_data[['index', 'LONGITUDE', 'LATITUDE', 'FatalInjuries', 'MajorInjuries',
       'Involving', 'SPD', 'LEN', 'ACC', 'VOL', 'F85th', 'LOCAL', 'F50th',
       'Precip', 'Air max', 'min', 'obs', 'year', 'month', 'day',
       'hour', 'Accident']]

#final_data.to_csv(r'dataset_v3.csv')
final_data = final_data.drop(['FatalInjuries', 'MajorInjuries',
       'Involving'], axis=1)
final_data.to_csv(r'dataset_v4.csv')
final_data = pd.read_csv('dataset_v4.csv')
final_data = final_data.sample(frac=1)
final_data = final_data.reset_index()
final_data = final_data.drop(['level_0', 'index'], axis=1)
final_data.to_csv(r'dataset_v5.csv')
final_data.shape
final_data.head(10)
#final_data = final_data.drop([''], axis=1)
final_data.head(10)
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(final_data, test_size = 0.2, random_state = 42)
X_train = train_set.drop(['Accident'], axis=1)
Y_train = train_set['Accident']
X_test = test_set.drop(['Accident'], axis=1)
Y_test = test_set['Accident']
Y_train.value_counts()
Y_test.value_counts()
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
param_C = 50
param_gamma = 50
classifier = SVC(C=1,gamma=1)



classifier.fit(X_train, Y_train)
expected = Y_test
predicted = classifier.predict(X_test)
print("Accuracy={:.4f}".format(accuracy_score(expected, predicted)))
from sklearn.model_selection import GridSearchCV
parameters = {'kernel':['rbf'], 'C':[1,2,5,10,20,50], 'gamma':[0.01,0.1,1,5,10,20,50]}
svm_clsf = SVC()
grid_clsf = GridSearchCV(estimator=svm_clsf, param_grid=parameters, scoring='accuracy',n_jobs=16,verbose=10)
grid_clsf.fit(X_train, Y_train)
classifier = grid_clsf.best_estimator_
params = grid_clsf.best_params_
print("Best parameters set found on development set:")
print(params)
print(classifier)
y_true, y_pred = Y_test, grid_clsf.predict(X_test)
print(classification_report(y_true, y_pred))
results = grid_clsf.cv_results_
for mean, std, params in zip(results["mean_test_score"],results["std_test_score"],results["params"]):
    print("%0.3f (+-%0.03f) for %r" %(mean, std*2, params))

