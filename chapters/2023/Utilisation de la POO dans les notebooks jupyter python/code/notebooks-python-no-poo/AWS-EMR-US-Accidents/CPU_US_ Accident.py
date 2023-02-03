import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# setting output format
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
plt.style.use('ggplot')
%%time
df = pd.read_csv('/Users/shih-tien/SW_PRO/Personal Projects/AWS_EMR/Dataset/US_Accidents_June20.csv')
df_original = df.copy()
df.head(3)
df.info()
df.describe(include='all')
df.isnull().sum()
# Extract year, month, day, hour, weekday, and accident duration

# Convert Start_Time and End_Time to datetypes
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

# Extract year, month, day, hour and weekday
df['Year']=df['Start_Time'].dt.year
df['Month']=df['Start_Time'].dt.strftime('%b')
df['Day']=df['Start_Time'].dt.day
df['Hour']=df['Start_Time'].dt.hour
df['Weekday']=df['Start_Time'].dt.strftime('%a')

# Extract the amount of time in the unit of minutes for each accident, round to the nearest integer
df['Time_Duration(min)']=round((df['End_Time']-df['Start_Time'])/np.timedelta64(1,'m'))
df.info()
# Deal with outliers

# 1. Drop rows with negative time_dutaion
# Check if there is any negative time_duration values
td_drop_index = df['Time_Duration(min)'][df['Time_Duration(min)']<=0].index
# Then drop TD < 0 rows
df.drop(td_drop_index, inplace=True)

# 2. Fill outliers with median values
# Remove outliers for Time_Duration(min): n * standard_deviation (n=3), backfill with median
n=3
median = df['Time_Duration(min)'].median()
std = df['Time_Duration(min)'].std()
outliers = (df['Time_Duration(min)'] - median).abs() > std*n
# Set outliers to NAN
df[outliers] = np.nan
# Fill NAN with median
df['Time_Duration(min)'].fillna(median, inplace=True)

print('Max time to clear an accident: {} minutes or {} hours or {} days\nMin time to clear an accident: {} minutes.'.format(df['Time_Duration(min)'].max(),round(df['Time_Duration(min)'].max()/60), round(df['Time_Duration(min)'].max()/60/24), df['Time_Duration(min)'].min()))
# Set the list of features to include in Machine Learning
feature_lst=['Source','TMC','Severity','Start_Lng','Start_Lat',
             'Distance(mi)','Side','City','County','State','Timezone',
             'Temperature(F)','Humidity(%)','Pressure(in)',
             'Visibility(mi)','Wind_Direction','Weather_Condition',
             'Amenity','Bump','Crossing','Give_Way','Junction','No_Exit',
             'Railway','Roundabout','Station','Stop','Traffic_Calming',
             'Traffic_Signal','Turning_Loop','Sunrise_Sunset','Hour','Weekday',
             'Time_Duration(min)']
# Select the dataset to include only the selected features
df_model=df[feature_lst].copy()
df_model.info()
# Drop rows with missing values

# Check missing values
df_model.isnull().mean()
df_model.dropna(subset=df_model.columns[df_model.isnull().mean()!=0], how='any', axis=0, inplace=True)
df_model.shape
df_model['Timezone'].value_counts()
df_model['Timezone'] = df_model['Timezone'].str[:2] + df_model['Timezone'].str[3:]
# Set state
state='TX'

# Select the state of Pennsylvania
df_state=df_model.loc[df_model.State==state].copy()
df_state.drop('State',axis=1, inplace=True)
df_state.info()
df_state.shape
# Map of accidents, color code by county

sns.scatterplot(x='Start_Lng', y='Start_Lat', data=df_state, hue='County', legend=False, s=20)
plt.show()
df_state['County'].unique()
# Set county
county='Dallas'

# Select the state of Pennsylvania
df_county=df_state.loc[df_state.County==county].copy()
df_county.drop('County',axis=1, inplace=True)
df_county.info()
# Output dataset and upload to AWS S3 bucket for PySpark computation!
df_county.to_csv('./US_Accidents_county_clean_data.csv',index=False)

# state granularity
df_state.to_csv('./US_Accidents_state_clean_data.csv',index=False)
# Map of accidents, color code by city

sns.scatterplot(x='Start_Lng', y='Start_Lat', data=df_county, hue='City', legend=False, s=20)
plt.show()
df_county.shape
# Generate dummies for categorical data
df_county_dummy = pd.get_dummies(df_county,drop_first=True)
df_county_dummy.info()
df_county_dummy.shape
df_county_dummy.head(3)
# Generate dummies for categorical data
df_state_dummy = pd.get_dummies(df_state,drop_first=True)
df_state_dummy.info()
df_state_dummy.shape
# Import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
# Assign the data

# County Granularity
#df=df_county_dummy

# State Granularity
df=df_state_dummy

# Set the target for the prediction
target='Severity'

# set X and y
y = df[target]
X = df.drop(target, axis=1)

# Split the data set into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
df.info(verbose=True)
df.head(3)
# List of classification algorithms
model_lst=['Logistic Regression', 'Random Forest']

# Initialize an empty list for the accuracy for each algorithm
accuracy_lst=[]
%%time
# Logistic regression
lr = LogisticRegression(random_state=0)
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

# Get the accuracy score
acc=accuracy_score(y_test, y_pred)

# Append to the accuracy list
accuracy_lst.append(acc)

print("[Logistic regression algorithm] accuracy_score: {:.3f}.".format(acc))
%%time
# Random Forest algorithm

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# Get the accuracy score
acc=accuracy_score(y_test, y_pred)

# Append to the accuracy list
accuracy_lst.append(acc)


# Model Accuracy, how often is the classifier correct?
print("[Randon forest algorithm] accuracy_score: {:.3f}.".format(acc))
feature_imp = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)

# Creating a bar plot, displaying only the top k features
k=10
sns.barplot(x=feature_imp[:k], y=feature_imp.index[:k])
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

# Generate a list of ticks for y-axis
y_ticks=np.arange(len(model_lst))

# Combine the list of algorithms and list of accuracy scores into a dataframe, sort the value based on accuracy score
df_acc=pd.DataFrame(list(zip(model_lst, accuracy_lst)), columns=['Algorithm','Accuracy_Score']).sort_values(by=['Accuracy_Score'],ascending = True)

# Export to a file
df_acc.to_csv('./Accuracy_scores_algorithms_{}.csv'.format(state),index=False)

# Make a plot
ax=df_acc.plot.barh('Algorithm', 'Accuracy_Score', align='center',legend=False,color='0.5')

# Add the data label on to the plot
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()+0.02, i.get_y()+0.2, str(round(i.get_width(),2)), fontsize=10)

# Set the limit, lables, ticks and title
plt.xlim(0,1.1)
plt.xlabel('Accuracy Score')
plt.yticks(y_ticks, df_acc['Algorithm'], rotation=0)
plt.title('[{}] Which algorithm is better?'.format(state))

plt.show()

