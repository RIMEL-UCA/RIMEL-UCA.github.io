from sklearn import tree
import pandas as pd

data = pd.read_csv("C:/Users/shard/OneDrive/Documents/MSAI/Data Mining/US_Accidents_Dec20_Updated.csv")




X = data.drop(['ID',
               'Severity',
               'Start_Lat',
               'End_Lat',
               'Start_Lng',
               'End_Lng',
               'Distance(mi)',
               'Start_Time',
               'End_Time',
               ], axis=1)
y = data["Severity"]
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline



# from https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

X_labeled = MultiColumnLabelEncoder(columns = ['Description','Street','Side','City','County','State','Zipcode','Country','Timezone','Airport_Code','Weather_Timestamp','Wind_Direction','Weather_Condition','Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop','Turning_Loop','Sunrise_Sunset','Civil_Twilight','Nautical_Twilight','Astronomical_Twilight']).fit_transform(X)

X_labeled = X_labeled.fillna(0)

print(X_labeled.dtypes)
print(X_labeled.head(10))
from sklearn.feature_selection import SelectKBest, chi2

#print (X_labeled.head(20))



#Converting F to Kelvin because K Best doesn't accept negative numbers
X_labeled["Temperature(Kel)"] =  273.5 + ((X_labeled["Temperature(F)"] - 32.0) * (5.0/9.0))
X_labeled["Wind_Chill(Kel)"] =  273.5 + ((X_labeled["Wind_Chill(F)"] - 32.0) * (5.0/9.0))

#Dropping F values to get rid of negatives
X_labeled = X_labeled.drop(['Temperature(F)', 'Wind_Chill(F)'], axis=1)



X_new = SelectKBest(chi2, k=10).fit(X_labeled, y)

#X_new.shape

cols = X_new.get_support(indices=True)
features_df_new = X_labeled.iloc[:,cols]

print(features_df_new)
from sklearn import tree
from sklearn.model_selection import train_test_split

X_labeled = X_labeled.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X_labeled, y, test_size=0.33, random_state=42)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
clf.score(X_test, y_test)
