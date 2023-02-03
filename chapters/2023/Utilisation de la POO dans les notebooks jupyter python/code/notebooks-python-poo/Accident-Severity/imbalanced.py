import pandas as pd

data = pd.read_csv("./accidents.zip")
print(data.groupby("Severity").size())
print(data.shape[0])

X = data.drop(['ID', 'Severity'], axis=1)
y = data["Severity"]
print(X.shape[0])

print(X.dtypes)
from imblearn.under_sampling import RandomUnderSampler 

rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)

print(y_res.value_counts())

from imblearn.over_sampling import RandomOverSampler 

ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

print(y_res.value_counts())


from imblearn.over_sampling import SMOTE 

X_float = X[["Start_Lat", 
             "Start_Lng", 
             "End_Lat", 
             "End_Lng", 
             "Distance(mi)",
             "Number",
             "Temperature(F)",
             "Wind_Chill(F)",
             "Humidity(%)",
             "Pressure(in)",
             "Visibility(mi)",
             "Wind_Speed(mph)",
             "Precipitation(in)"
             ]]
print(X_float.dtypes)

X_float = X_float.fillna(0)


sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_float, y)

print(y_res.value_counts())
from imblearn.over_sampling import SMOTEN
from sklearn.preprocessing import LabelEncoder

X_cat = X.drop(["Start_Lat", 
             "Start_Lng", 
             "End_Lat", 
             "End_Lng", 
             "Distance(mi)",
             "Number",
             "Temperature(F)",
             "Wind_Chill(F)",
             "Humidity(%)",
             "Pressure(in)",
             "Visibility(mi)",
             "Wind_Speed(mph)",
             "Precipitation(in)"
             ], axis=1)

X_cat = X_cat.fillna(0)
X_cat = X_cat.astype(str)

X_cat = X_cat.head(100)
y_cat = y.head(100)

print (X_cat)

sampler = SMOTEN(random_state=0)

X_res, y_res = sampler.fit_resample(X_cat, y_cat)
print(y_res.value_counts())
X_short = X.head(100000)
y_short = y.head(100000)

X_short = X_short.fillna(0)

X_short = X_short[["Start_Time", "Start_Lat"]]

from imblearn.over_sampling import SMOTENC

sm = SMOTENC(random_state=42, categorical_features=[0])
X_res, y_res = sm.fit_resample(X_short, y_short)

print(y_res.value_counts())
# # Now do the same but for SMOTENC

# # shorten data
# X_short = X.head(200)
# y_short = y.head(200)

# # remove NaN's
# X_short = X_short.fillna(0)

# # cast all categorical data as string

# X_short[["Start_Time", 
#          "End_Time",
#          "Description",
#          "Street",
#          "Side", 
#          "City",
#          "County",
#          "State",
#          "Zipcode", 
#          "Country",
#          "Timezone",
#          "Airport_Code",
#          "Weather_Timestamp", 
#          "Weather_Condition",
#          "Amenity",
#          "Bump",
#          "Crossing", 
#          "Give_Way",
#          "Junction",
#          "No_Exit",
#          "Railway", 
#          "Roundabout",
#          "Station",
#          "Stop",
#          "Traffic_Calming", 
#          "Traffic_Signal",
#          "Turning_Loop",
#          "Sunrise_Sunset",
#          "Civil_Twilight", 
#          "Nautical_Twilight",
#          "Astronomical_Twilight",
#          "Street", 
#          ]] = X_short[["Start_Time", 
#          "End_Time",
#          "Description",
#          "Street",
#          "Side", 
#          "City",
#          "County",
#          "State",
#          "Zipcode", 
#          "Country",
#          "Timezone",
#          "Airport_Code",
#          "Weather_Timestamp", 
#          "Weather_Condition",
#          "Amenity",
#          "Bump",
#          "Crossing", 
#          "Give_Way",
#          "Junction",
#          "No_Exit",
#          "Railway", 
#          "Roundabout",
#          "Station",
#          "Stop",
#          "Traffic_Calming", 
#          "Traffic_Signal",
#          "Turning_Loop",
#          "Sunrise_Sunset",
#          "Civil_Twilight", 
#          "Nautical_Twilight",
#          "Astronomical_Twilight",
#          "Street", 
#          ]].astype(str)

# print(X_short.dtypes)

# # print(X_short.iloc[:,44])

# from imblearn.over_sampling import SMOTENC

# sm = SMOTENC(random_state=42, categorical_features=[0,1,
#                                                     7,
#                                                     9,10,11,12,13,14,15,16,17,18,
#                                                     24,
#                                                     27,28,29,30,31,32,33,34,35,36,37,38,
#                                                     39,40,41,42,43,44
#                                                     ])


# # train!
# X_result, y_result = sm.fit_resample(X_short, y_short)
# print(y_result.value_counts())
