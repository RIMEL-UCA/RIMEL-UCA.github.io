#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import math
import pandas as pd
from simpletransformers.model import TransformerModel
from sklearn.model_selection import train_test_split

from alibi.datasets import fetch_movie_sentiment

def prepare_data(test_size):
    # load data
    X, y = fetch_movie_sentiment(return_X_y=True)

    # prepare data
    data = pd.DataFrame()
    data['text'] = X
    data['labels'] = y

    if math.isclose(test_size, 0.0):
        return data, None
    else:
        train, test = train_test_split(data, test_size=test_size)
        return train, test


train, test = prepare_data(0.20)
model = TransformerModel('roberta', 'roberta-base', args=({'fp16': False}))
model.train_model(train)


result, model_outputs, wrong_predictions = model.eval_model(test)
    
    
    
    

# In[ ]:



