#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np

# In[2]:


# Random seed to make results deterministic and reproducible
torch.manual_seed(0)

# In[3]:


# declare dimension
input_size = 4
hidden_size = 2

# In[4]:


# singleton example
# shape : (1, 1, 4)
# input_data_np = np.array([[[1, 0, 0, 0]]])

# sequential example
# shape : (3, 5, 4)
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]
input_data_np = np.array([[h, e, l, l, o], [e, o, l, l, l], [l, l, e, e, l]], dtype=np.float32)

# In[5]:


# transform as torch tensor
input_data = torch.Tensor(input_data_np)

# In[6]:


# declare RNN
rnn = torch.nn.RNN(input_size, hidden_size)

# In[7]:


# check output
outputs, _status = rnn(input_data)
print(outputs)
print(outputs.size())

# In[ ]:



