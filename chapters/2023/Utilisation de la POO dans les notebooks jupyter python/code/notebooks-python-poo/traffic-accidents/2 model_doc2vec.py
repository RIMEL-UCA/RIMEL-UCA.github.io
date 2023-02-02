import pandas as pd

import sys
sys.path.insert(0, '../../../')

from classes.doc2vec.preprocessing import Preprocessing as doc2vec
import numpy
print("numpy",numpy.version.version)

import gensim
print("gensim", gensim.__version__)
train = pd.read_csv("../../../data/v1/7030/train70.tsv", delimiter = "\t", quoting = 3)
train['dataset'] = 99 # train = 1
test = pd.read_csv("../../../data/v1/7030/test30.tsv", delimiter = "\t", quoting = 3)
test['dataset'] = 100 # test = 0
dataset = pd.concat([train,test])
dataset = dataset.reset_index(drop=True)
print(dataset.shape) # (3804, 3)
dataset.head(5)
#Preprocessing
#directory = "../../../data/v1/doc2vec/"
#directory = "../../../data/v1/doc2vec/v2/"
#directory = "../../../data/v1/doc2vec/v3/"
#directory = "../../../data/v1/doc2vec/v4/"
directory = "../../../data/v1/doc2vec/v5/"

file = "6_clean_lemma_dataset_propuesta1_5050"
#file = "test30"
type_clean = 6 #Tiene que ser el mismo que 'file' (prefijo)

#Model SVM
kernel='rbf'
gamma=0.2
C=7
clean = doc2vec(dataset)
clean.fit_clean(type_clean)

embendding = clean.feature_extraction_dbow100(directory, file)
import pickle
pickle.format_version
import pandas
pandas.__version__
vecs_train = embendding[embendding[:,0] == 99.0,:] #train = 99
vecs_test = embendding[embendding[:,0] == 100.0,:] #test = 100

X_train = vecs_train[:,2:]
y_train = vecs_train[:,1]
X_test = vecs_test[:,2:]
y_test = vecs_test[:,1]
X = embendding[:,2:]
y = embendding[:,1]
print("Size vecs_train", vecs_train.shape)
print("Size vecs_test", vecs_test.shape)
print("Size: \n * X_train: %s \n * y_train: %s \n * X_test: %s \n * y_test: %s" % (X_train.shape, y_train.shape, X_test.shape, y_test.shape))
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
classifier = SVC(random_state=123, kernel=kernel, gamma=gamma, C=C)
#classifier.fit(X_train, y_train)
classifier = RandomForestClassifier(n_estimators=100,random_state=100,n_jobs=-1)
"""classifier = RandomForestClassifier(n_estimators=n_estimators,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    max_features=max_features,
                                    max_depth=max_depth,
                                    bootstrap=bootstrap,
                                    random_state=100,n_jobs=-1)
"""
#classifier.fit(X_train, y_train)
skfold = StratifiedKFold(n_splits=10, random_state=100)

scores = model_selection.cross_val_score(classifier, X, y, cv=skfold)
print("Accuracy: %0.6f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = model_selection.cross_val_score(classifier, X, y, cv=skfold, scoring='f1_macro')
print("F1-score: %0.6f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = model_selection.cross_val_score(classifier, X, y, cv=skfold, scoring='recall_macro')
print("Recall: %0.6f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = model_selection.cross_val_score(classifier, X, y, cv=skfold, scoring='precision_macro')
print("Precision: %0.6f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# Predicting the Test set results
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred)

metrics_svm = []
metrics = {}
metrics['accuracy'] = accuracy_score(y_test, y_pred)
metrics['recall'] = recall_score(y_test, y_pred)
metrics['precision'] = precision_score(y_test, y_pred)
metrics['f1'] = f1_score(y_test, y_pred)
metrics_svm.append(metrics)
metrics_svm = pd.DataFrame(metrics_svm)
print(metrics_svm)
print(cm_svm)
