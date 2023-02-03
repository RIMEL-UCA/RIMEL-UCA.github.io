import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector

from MultiColumnLabelEncoder import MultiColumnLabelEncoder

data = pd.read_csv("./accidents.zip")
X = data.drop(['ID',
               'Severity',
               'Start_Lat',
               'End_Lat',
               'Start_Lng',
               'End_Lng',
               'Distance(mi)',
               'Start_Time',
               'End_Time',
               'Description',
               'Number',
               'Side'
               ], axis=1)
y = data["Severity"]
X_labeled = MultiColumnLabelEncoder(columns = ['Street','City','County','State','Zipcode','Country','Timezone','Airport_Code','Weather_Timestamp','Wind_Direction','Weather_Condition','Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop','Turning_Loop','Sunrise_Sunset','Civil_Twilight','Nautical_Twilight','Astronomical_Twilight']).fit_transform(X)
X_labeled = X_labeled.fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X_labeled, y, test_size=0.33, random_state=42)

clf = tree.DecisionTreeClassifier()
sfs = SequentialFeatureSelector(clf, n_features_to_select=10)
sfs.fit(X_labeled, y)

indices = sfs.get_support(indices=True)

for index in indices:
    print(X_labeled.columns[index])
