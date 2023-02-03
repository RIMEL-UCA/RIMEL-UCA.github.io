#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 1000)

# In[2]:


df = pd.read_pickle('./UCF101_videos_prediction.pkl')

# In[3]:


df

# ### Videos with wrong predictions

# In[4]:


a = df['filename'][df['y_pred']!=df['y']].tolist()
b = df['y'][df['y_pred']!=df['y']].tolist()
c = df['y_pred'][df['y_pred']!=df['y']].tolist()

df_mistakes = pd.DataFrame(data={'filename':a, 'y':b, 'y_pred':c})
df_mistakes.to_pickle('./wrong_predictions.pkl')

# In[5]:


df_mistakes

# ### Videos with correct predictions

# In[6]:


# find video filenames
correct_predict = ~df['filename'].isin(df_mistakes['filename'])

df_correct = df[correct_predict]
# df_correct.to_pickle('./correct_predictions.pkl')

# In[7]:


df_correct

# In[8]:


N = len(df)
N_correct = len(df_correct)
N_mistake = len(df_mistakes)

print('total videos: {}'.format(N))
print('Accuracy: {:.2f}%'.format(N_correct/N * 100))
