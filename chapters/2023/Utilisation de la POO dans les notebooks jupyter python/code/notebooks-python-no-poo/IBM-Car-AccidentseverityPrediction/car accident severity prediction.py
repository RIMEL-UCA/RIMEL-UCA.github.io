import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, validation_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import metrics


df=pd.read_csv('Data-Collisions.csv')
df.dtypes
df.shape
df.isna().sum()
print('Hello Capstone Project Course!')
df['SEVERITYCODE'].value_counts()
df['SEVERITYDESC'].value_counts()
df['SEVERITYCODE.1'].value_counts()
df['WEATHER'].value_counts()
sns.countplot(x='SEVERITYCODE',data=df) # This clearly shows the Data is imbalanced.
df.SEVERITYDESC.unique()
from sklearn.utils import resample
majordf=df[df['SEVERITYCODE']==1]
minordf=df[df['SEVERITYCODE']==2]
majordfsam=resample(majordf,replace=False,n_samples=58188,random_state=123)
majordfsam.shape
minordf.shape
balanced_df=pd.concat([majordfsam,minordf])
balanced_df.shape
sns.countplot(x='SEVERITYCODE',data=balanced_df) # The data seems balanced now.
balanced_df['UNDERINFL'].value_counts()
#Encoding Under the influence (0 = No, 1 = Yes)
balanced_df["UNDERINFL"].replace("N", 0, inplace=True)
balanced_df["UNDERINFL"].replace("Y", 1, inplace=True)
balanced_df['UNDERINFL'].value_counts()
#Encoding Speeding(0 = No, 1 = Yes)
balanced_df["SPEEDING"].replace("Y", 1, inplace=True)
balanced_df["SPEEDING"].replace(np.nan, 0, inplace=True)
balanced_df['SPEEDING'].value_counts()
#Encoding Light Conditions(0 = Light, 1 = Medium, 2 = Dark)
balanced_df["LIGHTCOND"].replace("Daylight", 0, inplace=True)
balanced_df["LIGHTCOND"].replace("Dark - Street Lights On", 1, inplace=True)
balanced_df["LIGHTCOND"].replace("Dark - No Street Lights", 2, inplace=True)
balanced_df["LIGHTCOND"].replace("Dusk", 1, inplace=True)
balanced_df["LIGHTCOND"].replace("Dawn", 1, inplace=True)
balanced_df["LIGHTCOND"].replace("Dark - Street Lights Off", 2, inplace=True)
balanced_df["LIGHTCOND"].replace("Dark - Unknown Lighting", 2, inplace=True)
balanced_df["LIGHTCOND"].replace("Other",3, inplace=True)

#Encoding Weather Conditions(0 = Clear, 1 = Overcast and Cloudy, 2 = Windy, 3 = Rain and Snow
balanced_df["WEATHER"].replace("Clear", 0, inplace=True)
balanced_df["WEATHER"].replace("Raining", 3, inplace=True)
balanced_df["WEATHER"].replace("Overcast", 1, inplace=True)
balanced_df["WEATHER"].replace("Other", "Unknown", inplace=True)
balanced_df["WEATHER"].replace("Snowing", 3, inplace=True)
balanced_df["WEATHER"].replace("Fog/Smog/Smoke", 2, inplace=True)
balanced_df["WEATHER"].replace("Sleet/Hail/Freezing Rain", 3, inplace=True)
balanced_df["WEATHER"].replace("Blowing Sand/Dirt", 2, inplace=True)
balanced_df["WEATHER"].replace("Severe Crosswind", 2, inplace=True)
balanced_df["WEATHER"].replace("Partly Cloudy", 1, inplace=True)

#Encoding Road Conditions(0 = Dry, 1 = Mushy, 2 = Wet)
balanced_df["ROADCOND"].replace("Dry", 0, inplace=True)
balanced_df["ROADCOND"].replace("Wet", 2, inplace=True)
balanced_df["ROADCOND"].replace("Ice", 2, inplace=True)
balanced_df["ROADCOND"].replace("Snow/Slush", 1, inplace=True)
balanced_df["ROADCOND"].replace("Other", "Unknown", inplace=True)
balanced_df["ROADCOND"].replace("Standing Water", 2, inplace=True)
balanced_df["ROADCOND"].replace("Sand/Mud/Dirt", 1, inplace=True)
balanced_df["ROADCOND"].replace("Oil", 2, inplace=True)
balanced_df.isna().sum()
df4=balanced_df[["X","Y","INCKEY","INATTENTIONIND","UNDERINFL","WEATHER","ROADCOND","LIGHTCOND","SPEEDING","SEVERITYCODE"]]
feature_df=df4.copy()
feature_df.dropna(axis=0,how='any',inplace=True)
feature_stats=feature_df.describe()

feature_stats
df4.dtypes
df4.replace(to_replace={'Unknown': np.nan, 
                        'Other':np.nan}, inplace=True)
df4.dropna(inplace=True)
df4.dtypes
df4['INATTENTIONIND'].replace(to_replace={'Y':1, 'N':0, '1':1, '0':0}, inplace=True)
df4.dtypes
df4.UNDERINFL.unique()
df4['UNDERINFL'].replace(to_replace={'Y':1, 'N':0, '1':1, '0':0}, inplace=True)
df4.dtypes
df4.drop(['X','Y'],axis=1)
plt.style.use('ggplot')
ax = sns.countplot(df['WEATHER'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()
ax = sns.countplot(df['ROADCOND'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()
df4.dtypes
df5=df4.drop(['X','Y'],axis=1)
df5
X=df4.drop('SEVERITYCODE',axis=1)
y=df4['SEVERITYCODE']
X.shape
y.shape
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit(X))
print(scaler.transform(X))
X.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
k=25
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
neigh
ypred_knn=neigh.predict(X_test)
neigh.score(X_test,y_test)
#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(C=6,solver='liblinear').fit(X_train,y_train)
lr
lr_pred=lr.predict_proba(X_test)
lr_pred
from sklearn.tree import DecisionTreeClassifier
CLTree=DecisionTreeClassifier(max_depth=7,criterion='entropy')
CLTree.fit(X_train,y_train)
TreePred=CLTree.predict(X_test)
TreePred
importance=CLTree.feature_importances_
df6=pd.DataFrame(importance)
df6.plot(kind='bar') # from this feature importance its evident that driver inattentive is the major cause for collisions

