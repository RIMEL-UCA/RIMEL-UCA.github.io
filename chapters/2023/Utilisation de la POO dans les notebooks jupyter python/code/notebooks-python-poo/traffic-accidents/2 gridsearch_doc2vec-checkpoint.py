import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../../../')

from classes.doc2vec.preprocessing import Preprocessing as doc2vec
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
directory = "../../../data/v1/doc2vec/v2/"
file = "5_clean_stem_dataset_propuesta1_5050"
type_clean = 5 #Tiene que ser el mismo que 'file' (prefijo)
clean = doc2vec(dataset)
clean.fit_clean(type_clean)

embendding = clean.feature_extraction_dbow(directory, file)
vecs_train = embendding[embendding[:,0] == 99.0,:] #train = 99
vecs_test = embendding[embendding[:,0] == 100.0,:] #test = 100

X_train = vecs_train[:,2:]
y_train = vecs_train[:,1]
X_test = vecs_test[:,2:]
y_test = vecs_test[:,1]
print("Size vecs_train", vecs_train.shape)
print("Size vecs_test", vecs_test.shape)
print("Size: \n * X_train: %s \n * y_train: %s \n * X_test: %s \n * y_test: %s" % (X_train.shape, y_train.shape, X_test.shape, y_test.shape))
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from pprint import pprint
from time import time
import logging  # Setting up the loggings to monitor gensim

logger = logging.getLogger("gridsearch")
hdlr = logging.FileHandler("gridsearch_doc2vec.log")
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)
logger.info("#####Comenzando a entrenar modelo######")    
logger.info(__doc__)
pipeline = Pipeline([      
  ('clf', RandomForestClassifier(random_state=100,bootstrap=False, max_features='auto') )
])

"""parameters = {'clf__n_estimators': [500, 600, 800],
               'clf__max_features': ['log2', 'auto'],
               'clf__max_depth': [30, 40, 70, 100, None],
               'clf__min_samples_split': [2, 5, 10],
               'clf__min_samples_leaf': [1, 2, 4],
               'clf__bootstrap': [True, False],
             }  """
parameters = {'clf__n_estimators': [600,1000, 1200, 1600, 2000],                             
               'clf__max_depth': [40, 50, 100, 110],
               'clf__min_samples_split': [2, 4, 5, 10],
               'clf__min_samples_leaf': [1, 2, 4],               
             }    

scores = ['accuracy', 'f1']  
{'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
 'max_features': ['auto', 'sqrt', 'log2'],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
 'min_samples_split': [2, 5, 10],
 'min_samples_leaf': [1, 2, 4, 7],
 'bootstrap': [True, False]}
