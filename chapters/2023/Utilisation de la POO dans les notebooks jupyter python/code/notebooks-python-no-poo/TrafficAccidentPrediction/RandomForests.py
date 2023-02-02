import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data = pd.read_csv('final_data.csv')
data.head()
data.shape
for cols in data.columns:
    print(cols)
data.drop('Unnamed: 0', inplace=True, axis=1)
data
corr = data.corr()
corr.style.background_gradient(cmap='coolwarm')
corr_target = abs(corr["Accident"])
relevant_features = corr_target[corr_target>0]
relevant_features
for cols in data.columns:
    if cols not in relevant_features:
        print(cols)
        data.drop(f'{cols}', axis=1, inplace=True)
data.head()
from sklearn.model_selection import train_test_split
X = data.drop('Accident', axis=1)
y = data['Accident']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
lst = []
for val in y_test:
    lst.append(val)
score = 0
for i in range(len(predictions)):
    if predictions[i] == lst[i]:
        score += 1
print(score/len(predictions)*100)
def predict(data):
    X = data.drop('Accident', axis=1)
    y = data['Accident']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    predictions = dtree.predict(X_test)
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
    lst = []
    for val in y_test:
        lst.append(val)
    score = 0
    for i in range(len(predictions)):
        if predictions[i] == lst[i]:
            score += 1
    print(score/len(predictions)*100)
predict(data)
for col in data.columns:
    if col != 'Accident':
        data[col] = data[col]/data[col].mean()
data
predict(data)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
clf.fit(X_train, y_train)
def predict2(data):
    X = data.drop('Accident', axis=1)
    y = data['Accident']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = RandomForestClassifier(n_estimators=100, max_depth=3,random_state=0)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
    lst = []
    for val in y_test:
        lst.append(val)
    score = 0
    for i in range(len(predictions)):
        if predictions[i] == lst[i]:
            score += 1
    print(score/len(predictions)*100)
predict2(data)

