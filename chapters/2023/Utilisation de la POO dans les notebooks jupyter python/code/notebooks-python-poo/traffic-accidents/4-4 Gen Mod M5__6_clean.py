import pandas as pd
import numpy as np

import pickle

import sys
sys.path.insert(0, '../../../')

from classes.tweet2accident.preprocessing import Preprocessing
train = pd.read_csv("../../../data/v1/7030/train70.tsv", delimiter = "\t", quoting = 3)
test = pd.read_csv("../../../data/v1/7030/test30.tsv", delimiter = "\t", quoting = 3)

print(train.shape, test.shape) # (3804, 3)
print(train.shape[0] + test.shape[0])
X, y = train.text, train.label
X_test, y_test = test.text, test.label
type_clean = 6 # The type clean in this case 5 corresponding to 5_steam

#TFIDF
max_df = 0.3
max_features = 2000
min_df = 0.001
ngram_range= (1, 2)


#TF
max_depth = 100
min_samples_leaf = 1
min_samples_split = 4
n_estimators = 2000
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
params = {
    'n_estimators': n_estimators,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'max_features': 'sqrt',
    'max_depth': max_depth,
    'bootstrap': False,
    'random_state': 100,    
    'n_jobs': -1
}
pipeline = Pipeline([
    ('transform', Preprocessing(type_clean=type_clean, njobs=4)),
    ('tfidf', TfidfVectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df, max_features=max_features)),
    ('clf', RandomForestClassifier(**params))
])
pipeline.fit(X, y)
# Predicting the Test set results
y_pred = pipeline.predict(X_test)

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
filename = 'model_m5_tfidf_rf_1_false.pkl'
with open(filename, 'wb') as model_file:
      pickle.dump(pipeline, model_file)
accident_clasification_model = pickle.load(open(filename, 'rb'))
#Probando el modelo de clasificación cargado
text_predict = accident_clasification_model.predict(["choque entre camion y carro particular"])
print(text_predict[0]) #El resultado es 1 si es accidente, y 0 si no lo es.
text_test = [
    "choque entre camion y carro particular",
    "caí por accidente en tu corazón", 
    "accidente aereo deja 100 muertos en australia",
    "hola mundo",
    "accidente carro"
]
text_predict = accident_clasification_model.predict(text_test)
print(text_predict) #El resultado es 1 si es accidente, y 0 si no lo es.
df = pd.DataFrame(data=text_test,columns=["text"])
df["label"] = text_predict
df
