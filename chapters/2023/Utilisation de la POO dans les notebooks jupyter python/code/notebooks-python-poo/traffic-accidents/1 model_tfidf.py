import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../../../')

from classes.tfidf.preprocessing import Preprocessing as tfidf
def vector_to_phrase(vector, tfidf):
    c = 0
    sentence = []
    for i in vector:
        if i!=0.0:        
            sentence.append(tfidf.get_feature_names()[c])
        c+=1
    phrase = " ".join(sentence)
    return phrase
train = pd.read_csv("../../../data/v1/7030/train70.tsv", delimiter = "\t", quoting = 3)
test = pd.read_csv("../../../data/v1/7030/test30.tsv", delimiter = "\t", quoting = 3)
dataset = pd.concat([train,test])
print(train.shape, test.shape) # (3804, 3)
type_clean = 5 #Tiene que ser el mismo que 'file' (prefijo)

#TFIDF
max_df = 0.5    
max_features = 800
min_df = 0.001
ngram_range= (1, 1)

#Model SVM
C=4
gamma=0.7
kernel= 'rbf'
clean = tfidf(train)
clean.fit_clean(type_clean)
train.head()
clean_test = tfidf(test)
clean_test.fit_clean(type_clean)
test.head()
train = train[~train['clean'].isnull()] #Elimina publicaciones que estan null al eliminarlo porque no generan valor en el proceso de limpieza
test = test[~test['clean'].isnull()]
print(train.shape, test.shape) # (3804, 3)
embedding, vectorizer = clean.feature_extraction(ngram_range=ngram_range, max_df=max_df, min_df=min_df, max_features=max_features)
X_train = embedding[:,1:]
X_train=X_train.astype('float')

y_train = embedding[:,0]
y_train=y_train.astype('int')
X_test, y_test = vectorizer.transform(test.clean).toarray(), test.label
type(X_train)
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
params = {
    'n_estimators': 1600,
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'max_depth': 50,
    'bootstrap': False,
    'random_state': 100,    
    'n_jobs': -1
}
classifier = RandomForestClassifier(**params)
#dataset = pd.concat([train,test])
clean_all = tfidf(dataset)
clean_all.fit_clean(type_clean)
embedding_all, vectorizer_all = clean_all.feature_extraction(ngram_range=ngram_range, max_df=max_df, min_df=min_df, max_features=max_features)
X_all = embedding_all[:,1:]
X_all=X_all.astype('float')

y_all = embedding_all[:,0]
y_all=y_all.astype('int')
skfold = StratifiedKFold(n_splits=10, random_state=100)

scores = model_selection.cross_val_score(classifier, X_all, y_all, cv=skfold, n_jobs=-1)
print("Accuracy: %0.6f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = model_selection.cross_val_score(classifier, X_all, y_all, cv=skfold, scoring='f1_macro', n_jobs=-1)
print("F1-score: %0.6f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = model_selection.cross_val_score(classifier, X_all, y_all, cv=skfold, scoring='recall_macro', n_jobs=-1)
print("Recall: %0.6f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = model_selection.cross_val_score(classifier, X_all, y_all, cv=skfold, scoring='precision_macro', n_jobs=-1)
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
