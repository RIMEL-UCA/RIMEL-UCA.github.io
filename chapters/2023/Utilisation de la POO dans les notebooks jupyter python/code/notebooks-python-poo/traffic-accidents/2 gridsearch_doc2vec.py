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
#directory = "../../../data/v1/doc2vec/v2/"
#directory = "../../../data/v1/doc2vec/v3/"
directory = "../../../data/v1/doc2vec/v4/"
file = "6_clean_lemma_dataset_propuesta1_5050"

type_clean = 6 #Tiene que ser el mismo que 'file' (prefijo)
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
  ('clf', SVC(random_state=123) )
])
parameters = {          
        'clf__kernel': ('linear', 'poly', 'rbf'),              
        'clf__C': (0.01, 0.05, 0.1, 1, 2, 3, 4, 5, 6, 7, 8),
        'clf__gamma': (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.7, 1, 2, 3,10)            
}
"""
parameters = {          
        'clf__kernel': ('poly', 'rbf'),              
        'clf__C': (5, 6),
        'clf__gamma': (0.1, 0.2, 0.3, 0.4)            
}
"""
scores = ['accuracy', 'f1']   
try:
    for score in scores:
        logger.info("# Tuning hyper-parameters for %s" % score)
        logger.info(" ")
    
        logger.info("Performing grid search...")
        print("pipeline:", [name for name, _ in pipeline.steps])
        logger.info("parameters:")
        pprint(parameters)
        t0 = time()
        grid_search = GridSearchCV(pipeline, parameters, cv=5, scoring=score, n_jobs=-1,verbose=1)
        grid_search.fit(X_train, y_train)
        logger.info("done in %0.3fs" % (time() - t0))
        logger.info(" ")
        
        logger.info("Best parameters set found on development set:")
        logger.info(" ")
        logger.info(grid_search.best_params_)
        logger.info(" ")
        ##Old start
        logger.info("--")
        logger.info("Best score: %0.3f" % grid_search.best_score_)    
        logger.info("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()    
        for param_name in sorted(parameters.keys()):
            logger.info("\t%s: %r" % (param_name, best_parameters[param_name]))
        logger.info("--")
        logger.info(" ")
        
        logger.info("Grid scores on development set:")
        logger.info(" ")
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        for mean, std, params in sorted(zip(means, stds, grid_search.cv_results_['params']),key = lambda t: t[0],reverse=True):
            logger.info("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        logger.info(" ")
    
        logger.info("Detailed classification report:")
        logger.info(" ")
        logger.info("The model is trained on the full development set.")
        logger.info("The scores are computed on the full evaluation set.")
        logger.info(" ")
        y_true, y_pred = y_test, grid_search.predict(X_test)
        logger.info(classification_report(y_true, y_pred))
        logger.info(" ")
        
except Exception as e:
    logger.error('Unhandled exception:')
    logger.error(e)
