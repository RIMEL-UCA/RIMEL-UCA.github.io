import pandas as pd
import numpy as np

import pickle

import sys
sys.path.insert(0, '../../../')

from classes.tweet2accident.preprocessing import Preprocessing
file = "1_server_bogota.tsv"
#M1
filemodel = 'model_m1_dbow_svm.pkl'
dir_ = "../../../data/database/output_ml/M1/"

#M2
#filemodel = 'model_m2_tfidf_svm.pkl'
#dir_ = "../../../data/database/output_ml/M2/"
#dataset = pd.read_csv("../../../data/database/server_bogota/"+file, delimiter = "\t", quoting = 3)
dataset = pd.read_csv("../../../data/database/"+file, delimiter = "\t", quoting = 3)
dataset.shape
model = pickle.load(open(filemodel, 'rb')) 
import time
start = time.perf_counter()
text_predict = model.predict(dataset['text'])
end = time.perf_counter()
print("Tiempo de total (seg.)")
print(end - start)
dataset['label'] = text_predict
#dataset['label_3'] = text_predict
accident = dataset[dataset['label'] == 1 ]
no_accident = dataset[dataset['label'] == 0 ]
dataset.to_csv(dir_+"clf_"+file,sep='\t')
print("Accident dataset Size:"+str(accident.shape))
print("No Accident dataset Size:"+str(no_accident.shape))
name_output = file
accident.to_csv(dir_+"accident_"+name_output,sep='\t')
no_accident.to_csv(dir_+"no_accident_"+name_output,sep='\t')
count_tweet_by_username = accident['user_name'].value_counts() # Show distribution of tweets by user
count_tweet_by_username[:10]
