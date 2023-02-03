#!/usr/bin/env python
# coding: utf-8

# !!!!!!!!!!!!!!!! DRAFT DRAFT DRAFT !!!!!!!!!!!!!!!!
# 
# !!!!!!!!!!!!!!!! work in progesss  !!!!!!!!!!!!!!!! 
# 
# !!!!!!!!!!!!!!!! DRAFT DRAFT DRAFT !!!!!!!!!!!!!!!!

# Credits / Inspired by => https://github.com/mwitiderrick/stockprice

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# In[ ]:


!pip install mido

# In[ ]:


import mido
mid = mido.MidiFile('bach.mid')

# In[ ]:


song = []
full_song_message = []
for i, track in enumerate(mid.tracks):
    if track.name == "Track 1":
        for msg in track:
            print(msg)
            if msg.dict()['type'] in ['note_on','note_off']:
                song.append((msg.dict()['note'],msg.dict()['velocity'],msg.dict()['time'],msg.dict()['type']))
            full_song_message.append(msg)

# In[ ]:


song

# In[ ]:


from mido import Message, MidiFile, MidiTrack

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)




for note, velocity, time, type in song:
    track.append(mido.Message(type, note=note, velocity=velocity, time=time))
    #track.append(message)
    
mid.save('new_song.mid')

# In[ ]:


training_set = pd.DataFrame(song)
del training_set[1]
del training_set[2]
del training_set[3]
training_set = training_set.to_numpy()

# In[ ]:


"""
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
"""
training_set_scaled = training_set
training_set_scaled

# In[ ]:


X_train = []
y_train = []
for i in range(60, 2700):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i:i+60, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))


# In[ ]:


y_train

# In[ ]:




# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# In[ ]:


model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(LSTM(units = 50, return_sequences = True))
model.add(LSTM(units = 50, return_sequences = True))
model.add(LSTM(units = 50, return_sequences = True))
model.add(LSTM(units = 1, return_sequences = True))


model.compile(optimizer = 'adam', loss = 'mean_squared_error')



# In[ ]:


model.fit(X_train, y_train, epochs = 100, batch_size = 32)

# In[ ]:


y_predict = model.predict(X_train)

# In[ ]:


y_predict

# In[ ]:


created_song = pd.DataFrame(song)

# In[ ]:


created_song

# In[ ]:



