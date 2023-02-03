import pandas as pd

data = pd.read_csv("./accidents.zip")
print(data.groupby("Severity").size())
print(data.shape[0])

X = data[['County','State','Timezone','Airport_Code','Bump','Give_Way','Roundabout','Stop','Traffic_Signal','Turning_Loop']]
y = data['Severity']

print(X.dtypes)
from MultiColumnLabelEncoder import MultiColumnLabelEncoder

X_labeled = MultiColumnLabelEncoder(columns = ['County','State','Timezone','Airport_Code']).fit_transform(X)
print(X_labeled.dtypes)
from imblearn.over_sampling import SMOTENC

sm = SMOTENC(random_state=42, categorical_features=[4,5,6,7,8,9])

X_labeled = X_labeled.head(10000)
y = y.head(10000)

X_res, y_res = sm.fit_resample(X_labeled, y)

print(y_res.value_counts())
