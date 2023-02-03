#!/usr/bin/env python
# coding: utf-8

# # Lab 6-2: Fancy Softmax Classification

# Author: Seungjae Lee (이승재)

# ## Imports

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# In[2]:


# For reproducibility
torch.manual_seed(1)

# ## Cross-entropy Loss with `torch.nn.functional`

# PyTorch has `F.log_softmax()` function.

# In[3]:


z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)
y = torch.randint(5, (3,)).long()
y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1), 1)

# In[4]:


# Low level
torch.log(F.softmax(z, dim=1))

# In[5]:


# High level
F.log_softmax(z, dim=1)

# PyTorch also has `F.nll_loss()` function that computes the negative loss likelihood.

# In[6]:


# Low level
(y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean()

# In[7]:


# High level
F.nll_loss(F.log_softmax(z, dim=1), y.long())

# PyTorch also has `F.cross_entropy` that combines `F.log_softmax()` and `F.nll_loss()`.

# In[8]:


F.cross_entropy(z, y)

# ## Data

# In[9]:


xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)

# In[10]:


x_train = torch.FloatTensor(xy[:, 0:-1])
y_train = torch.LongTensor(xy[:, [-1]]).squeeze()

# In[11]:


print(x_train.shape) # x_train shape
print(len(x_train))  # x_train 길이
print(x_train[:5])   # 첫 다섯 개

# In[12]:


print(y_train.shape) # y_train shape
print(len(y_train))  # y_train 길이
print(y_train[:5])   # 첫 다섯 개

# In[13]:


nb_classes = 7
y_one_hot = torch.zeros((len(y_train), nb_classes))
y_one_hot = y_one_hot.scatter(1, y_train.unsqueeze(1), 1)

# ## Training with `F.cross_entropy`

# In[14]:


# 모델 초기화
W = torch.zeros((16, 7), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # Cost 계산 (2)
    z = x_train.matmul(W) + b # or .mm or @
    cost = F.cross_entropy(z, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# ## High-level Implementation with `nn.Module`

# In[15]:


class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 7)
    def forward(self, x):
        return self.linear(x)

# In[16]:


model = SoftmaxClassifierModel()

# In[17]:


# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.cross_entropy(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    # 20번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# <div class="alert alert-warning">
#     Should I display how many it got correct in the training set?
# </div>
