import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

import sys
sys.path.insert(0, '../../../')

from classes.tweet2accident.preprocessing import Preprocessing
train = pd.read_csv("../../../data/v1/7030/train70.tsv", delimiter = "\t", quoting = 3)
test = pd.read_csv("../../../data/v1/7030/test30.tsv", delimiter = "\t", quoting = 3)
X, y = train.text, train.label
X_test, y_test = test.text, test.label
model1 = pickle.load(open('model_m1_dbow_svm.pkl', 'rb')) 
model3 = pickle.load(open('model_m3_tfidf_svm_1.pkl', 'rb')) 
model4 = pickle.load(open('model_m4_tfidf_svm_2.pkl', 'rb'))
model5f = pickle.load(open('model_m5_tfidf_rf_1_false.pkl', 'rb'))
model5t = pickle.load(open('model_m5_tfidf_rf_1_true.pkl', 'rb'))
y_pred1 = model1.predict(X_test)
y_pred3 = model3.predict(X_test)
y_pred4 = model4.predict(X_test)
y_pred5f = model5f.predict(X_test)
y_pred5t = model5t.predict(X_test)
print(y_pred1.shape, y_pred3.shape,y_pred4.shape,y_pred5f.shape,y_pred5t.shape,)
len(y_pred1)
y_pred4[3]
predv2 = []
for i in range(len(y_pred1)):    
    score = 0
    score = (0.4*y_pred1[i]) + (0.3*y_pred3[i]) + (0.3*y_pred4[i])
    if score >= 0.4:
        predv2.append(1)
    else:
        predv2.append(0)

y_predv2 = np.array(predv2)

cm_svm = confusion_matrix(y_test, y_predv2)

metrics_svm = []
metrics = {}
metrics['accuracy'] = accuracy_score(y_test, y_predv2)
metrics['recall'] = recall_score(y_test, y_predv2)
metrics['precision'] = precision_score(y_test, y_predv2)
metrics['f1'] = f1_score(y_test, y_predv2)
metrics_svm.append(metrics)
metrics_svm = pd.DataFrame(metrics_svm)

print(metrics_svm)
print(cm_svm)
predv2 = []
for i in range(len(y_pred1)):    
    score = 0
    score = y_pred1[i] + y_pred3[i] + y_pred4[i]
    if score >= 1:
        predv2.append(1)
    else:
        predv2.append(0)

y_predv2 = np.array(predv2)

cm_svm = confusion_matrix(y_test, y_predv2)

metrics_svm = []
metrics = {}
metrics['accuracy'] = accuracy_score(y_test, y_predv2)
metrics['recall'] = recall_score(y_test, y_predv2)
metrics['precision'] = precision_score(y_test, y_predv2)
metrics['f1'] = f1_score(y_test, y_predv2)
metrics_svm.append(metrics)
metrics_svm = pd.DataFrame(metrics_svm)

print(metrics_svm)
print(cm_svm)
predv2 = []
for i in range(len(y_pred1)):    
    score = 0
    score = y_pred1[i] + y_pred3[i] + y_pred4[i]
    if score >= 2:
        predv2.append(1)
    else:
        predv2.append(0)

y_predv2 = np.array(predv2)

cm_svm = confusion_matrix(y_test, y_predv2)

metrics_svm = []
metrics = {}
metrics['accuracy'] = accuracy_score(y_test, y_predv2)
metrics['recall'] = recall_score(y_test, y_predv2)
metrics['precision'] = precision_score(y_test, y_predv2)
metrics['f1'] = f1_score(y_test, y_predv2)
metrics_svm.append(metrics)
metrics_svm = pd.DataFrame(metrics_svm)

print(metrics_svm)
print(cm_svm)
tn, fp, fn, tp = confusion_matrix(y_test, y_predv2).ravel()
(tn, fp, fn, tp)
tn = cm_svm[0,0]
fn = cm_svm[1,0]
tp = cm_svm[1,1]
fp = cm_svm[0,1]
(tn, fp, fn, tp)
predv2 = []
for i in range(len(y_pred1)):    
    score = 0
    score = y_pred1[i] + y_pred3[i] + y_pred5f[i]
    if score >= 2:
        predv2.append(1)
    else:
        predv2.append(0)

y_predv2 = np.array(predv2)

cm_svm = confusion_matrix(y_test, y_predv2)

metrics_svm = []
metrics = {}
metrics['accuracy'] = accuracy_score(y_test, y_predv2)
metrics['recall'] = recall_score(y_test, y_predv2)
metrics['precision'] = precision_score(y_test, y_predv2)
metrics['f1'] = f1_score(y_test, y_predv2)
metrics_svm.append(metrics)
metrics_svm = pd.DataFrame(metrics_svm)

print(metrics_svm)
print(cm_svm)
predv2 = []
for i in range(len(y_pred1)):    
    score = 0
    score = y_pred4[i] + y_pred5f[i]
    if score >= 1:
        predv2.append(1)
    else:
        predv2.append(0)

y_predv2 = np.array(predv2)

cm_svm = confusion_matrix(y_test, y_predv2)

metrics_svm = []
metrics = {}
metrics['accuracy'] = accuracy_score(y_test, y_predv2)
metrics['recall'] = recall_score(y_test, y_predv2)
metrics['precision'] = precision_score(y_test, y_predv2)
metrics['f1'] = f1_score(y_test, y_predv2)
metrics_svm.append(metrics)
metrics_svm = pd.DataFrame(metrics_svm)

print(metrics_svm)
print(cm_svm)
