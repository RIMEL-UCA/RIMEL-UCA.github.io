import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../../../')

from classes.tfidf.preprocessing import Preprocessing as tfidf
train = pd.read_csv("../../../data/v1/7030/train70.tsv", delimiter = "\t", quoting = 3)
test = pd.read_csv("../../../data/v1/7030/test30.tsv", delimiter = "\t", quoting = 3)

print(train.shape, test.shape) # (3804, 3)
train.head()
test.head()
type_clean = 4 #Tiene que ser el mismo que 'file' (prefijo)
#TFIDF
max_df = 0.5    
max_features = None
min_df = 0.001    
ngram_range = (1, 2)
clean = tfidf(train)
clean.fit_clean(type_clean)
train.head()
clean = tfidf(test)
clean.fit_clean(type_clean)
test.head()
train = train[~train['clean'].isnull()] #Elimina publicaciones que estan null al eliminarlo porque no generan valor en el proceso de limpieza
test = test[~test['clean'].isnull()]
print(train.shape, test.shape) # (3804, 3)
X, y = train.clean, train.label
X_test, y_test = test.clean, test.label
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

from pprint import pprint
from time import time
embedding, vectorizer = clean.feature_extraction(ngram_range=ngram_range, max_df=max_df, min_df=min_df, max_features=max_features)
X_train = embedding[:,1:]
X_train=X_train.astype('float')

y_train = embedding[:,0]
y_train=y_train.astype('int')
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 7]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

random_grid
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=100, n_jobs = -1)# Fit the random search model
rf_random.fit(X_train, y_train)
rf_random.best_params_
