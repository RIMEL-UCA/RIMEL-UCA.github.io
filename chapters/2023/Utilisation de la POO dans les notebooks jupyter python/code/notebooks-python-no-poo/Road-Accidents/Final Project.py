import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

road_accidents=pd.read_csv('road-accidents.csv',comment='#',sep='|')
road_accidents.head(10)
miles_driven=pd.read_csv('miles-driven.csv',sep='|',comment='#')
miles_driven.head(10)
df=pd.concat([road_accidents,miles_driven['million_miles_annually']],axis=1)
df.head(10)
df.describe()

sns.pairplot(df,diag_kind='hist')
corr=df.corr(method='pearson')
corr
sns.heatmap(corr,xticklabels=corr.columns,
           yticklabels=corr.columns,
           cmap='YlGnBu',
           annot=True,
           linewidth=1.0,vmin=0.2,vmax=0.9)
df.columns
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df[['perc_fatl_speed', 'perc_fatl_alcohol',
       'perc_fatl_1st_time']],df['drvr_fatl_col_bmiles'],test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X_train,y_train)

y_pred=lin_reg.predict(X_test)
y_pred
from sklearn.preprocessing import LabelEncoder
df['state']=LabelEncoder().fit_transform(df['state'])

from sklearn.preprocessing import MinMaxScaler
x=df[['perc_fatl_speed','perc_fatl_alcohol','perc_fatl_1st_time','drvr_fatl_col_bmiles']].values
sc=MinMaxScaler()
df1=sc.fit_transform(x)
df1
from sklearn.decomposition import PCA

pca=PCA(n_components=2)
principal_comp=pca.fit_transform(df1)
pca.components_
import matplotlib.pyplot as plt
plt.scatter(df1[:,0], df1[:,1] , c=df['drvr_fatl_col_bmiles'] ,cmap ='plasma')
plt.xlabel('state')
plt.ylabel('perc_fatl_Speed')
plt.colorbar()
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=3,random_state=0)
kmeans.fit(df1)
print(kmeans.cluster_centers_)
kmeans.labels_
y_kmeans=kmeans.predict(df1)
y_kmeans
plt.scatter(df1[y_kmeans == 0, 0], df1[y_kmeans == 0, 1], c = 'red', label = 'Cluster 1')
plt.scatter(df1[y_kmeans == 1, 0], df1[y_kmeans == 1, 1], c = 'blue', label = 'Cluster 2')
plt.scatter(df1[y_kmeans == 2, 0], df1[y_kmeans == 2, 1], c = 'green', label = 'Cluster 3')
plt.title('Clusters')
plt.legend()
plt.show()
df['c']=kmeans.labels_
df.head()
df['number_of_accidents']=df['drvr_fatl_col_bmiles']*df['million_miles_annually']/1000
df.head()
sns.barplot(df['c'],df['number_of_accidents'],ci=None,estimator=sum)

