import pandas as pd
import pickle

import sys
sys.path.insert(0, '../../../')

from classes.tweet2accident.preprocessing import Preprocessing
from classes.tweet2accident.doc2vec import Embedding
train = pd.read_csv("../../../data/v1/7030/train70.tsv", delimiter = "\t", quoting = 3)
test = pd.read_csv("../../../data/v1/7030/test30.tsv", delimiter = "\t", quoting = 3)
dataset = pd.concat([train,test])

print(train.shape, test.shape, dataset.shape) # (3804, 3)
print(train.shape[0] + test.shape[0], dataset.shape[0])
X, y = dataset.text, dataset.label
X_train, y_train = train.text, train.label
X_test, y_test = test.text, test.label
#Preprocessing
directory = "../../../data/v1/doc2vec/"
file = "6_clean_lemma_dataset_propuesta1_5050"
type_clean = 6 #Tiene que ser el mismo que 'file' (prefijo)

#Model SVM
kernel='rbf'
gamma=0.2
C=7
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
pipeline = Pipeline([
    ('transform', Preprocessing(type_clean=type_clean, njobs=4)),
    #Tener en cuenta, que directory apunta a '../../../data/v1/doc2vec/', 
    #si queremos que el pipeline funcione en alguna ruta en especifica hay que cambiar la ruta.
    ('dbow', Embedding(type_transform='dbow',directory=directory, file=file)),
    ('clf', SVC(random_state=123, kernel=kernel, C=C, gamma=gamma))
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
filename = 'model_m1_dbow_svm.pkl'
with open(filename, 'wb') as model_file:
      pickle.dump(pipeline, model_file)
accident_clasification_model = pickle.load(open(filename, 'rb'))
text_test = [
    "choque entre camion y carro particular",
    "caí por accidente en tu corazón", 
    "accidente aereo deja 100 muertos en australia",
    "hola mundo",
    "accidente en carro"
]
text_predict = accident_clasification_model.predict(text_test)
print(text_predict) #El resultado es 1 si es accidente, y 0 si no lo es.
