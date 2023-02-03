import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
data = pd.read_csv('final_data.csv')
train_set, test_set = train_test_split(data, test_size = 0.2, random_state = 42)
X_train = train_set.drop(['Accident'], axis=1)
Y_train = train_set['Accident']
X_test = test_set.drop(['Accident'], axis=1)
Y_test = test_set['Accident']
Y = data['Accident']
X = data.drop(['Accident'], axis = 1)
Y_train.value_counts()
Y_test.value_counts()
import math
import operator
def euclideaDistance(in1, in2, length1, length2):
    distance = 0
    for i in (length1, length2):
        distance += pow((in1[i] - in2[i]),2)
    return math.sqrt(distance)
data.iloc[1]
len(data.iloc[1])
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=20)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
print("Accuracy={:.4f}".format(accuracy_score(Y_test, Y_pred)))
from sklearn.model_selection import cross_val_score

k = []
accuracy = []
for i in range(1, 50, 2):
    k.append(i)
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)    
    accuracy.append(accuracy_score(Y_test, Y_pred))

plt.plot(k, accuracy)
plt.show()
max(accuracy)
k[pd.Series(accuracy).idxmax()]
k = []
accuracy = []
for i in range(1, 100, 2):
    k.append(i)
    classifier = KNeighborsClassifier(n_neighbors=i) 
    scores = cross_val_score(classifier, X, Y, cv=5, scoring='accuracy')
    accuracy.append(scores.mean())

plt.plot(k, accuracy)
plt.show()
max(accuracy)
k[pd.Series(accuracy).idxmax()]
