import pandas as pd
data = pd.read_csv('final_data.csv')
data.drop(columns = 'Unnamed: 0', inplace = True)
print(data)
X = data.drop('Accident',axis=1)
y = data['Accident']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
#SVM
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear',C=2,gamma=0.01)
svclassifier.fit(X_train, y_train)
print(svclassifier.score(X_test,y_test))

from sklearn.metrics import accuracy_score
y_pred = svclassifier.predict(X_test)
print(accuracy_score(y_test,y_pred))

