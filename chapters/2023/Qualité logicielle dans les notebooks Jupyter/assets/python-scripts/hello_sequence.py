#!/usr/bin/env python
# coding: utf-8

# Credits / Inspired by => https://github.com/mwitiderrick/stockprice

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# In[2]:


data_input = np.arange(10000)
training_set = pd.DataFrame(data_input)


# In[4]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# In[6]:


X_train = []
y_train = []
for i in range(60, 10000):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# In[8]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# In[ ]:


model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(LSTM(units = 50))
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(X_train, y_train, epochs = 10, batch_size = 32)

# In[ ]:


y_predict = model.predict(X_train)

# In[ ]:


pd.DataFrame(sc.inverse_transform(y_predict))

# In[ ]:



