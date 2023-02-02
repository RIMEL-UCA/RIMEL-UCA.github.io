#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.optim as optim
import numpy as np

# In[2]:


# Random seed to make results deterministic and reproducible
torch.manual_seed(0)

# In[3]:


sample = " if you want you"

# In[11]:


# make dictionary
char_set = list(set(sample))
char_dic = {c: i for i, c in enumerate(char_set)}
print(char_dic)

# In[12]:


# hyper parameters
dic_size = len(char_dic)
hidden_size = len(char_dic)
learning_rate = 0.1

# In[13]:


# data setting
sample_idx = [char_dic[c] for c in sample]
x_data = [sample_idx[:-1]]
x_one_hot = [np.eye(dic_size)[x] for x in x_data]
y_data = [sample_idx[1:]]

# In[14]:


# transform as torch tensor variable
X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)

# In[15]:


# declare RNN
rnn = torch.nn.RNN(dic_size, hidden_size, batch_first=True)

# In[16]:


# loss & optimizer setting
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), learning_rate)

# In[17]:


# start training
for i in range(50):
    optimizer.zero_grad()
    outputs, _status = rnn(X)
    loss = criterion(outputs.view(-1, dic_size), Y.view(-1))
    loss.backward()
    optimizer.step()

    result = outputs.data.numpy().argmax(axis=2)
    result_str = ''.join([char_set[c] for c in np.squeeze(result)])
    print(i, "loss: ", loss.item(), "prediction: ", result, "true Y: ", y_data, "prediction str: ", result_str)

# In[ ]:



