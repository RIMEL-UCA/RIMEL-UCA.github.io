#!/usr/bin/env python
# coding: utf-8

# # Lab 4-2: Load Data

# Author: Seungjae Lee (이승재)

# <div class="alert alert-warning">
#     We use elemental PyTorch to implement linear regression here. However, in most actual applications, abstractions such as <code>nn.Module</code> or <code>nn.Linear</code> are used.
# </div>

# ## Slicing 1D Array

# In[2]:


nums = [0, 1, 2, 3, 4]

# In[3]:


print(nums)

# index 2에서 4 전까지 가져와라. (앞 포함, 뒤 비포함)

# In[4]:


print(nums[2:4])

# index 2부터 다 가져와라.

# In[5]:


print(nums[2:])

# index 2 전까지 가져와라. (역시 뒤는 비포함)

# In[6]:


print(nums[:2])

# 전부 가져와라

# In[7]:


print(nums[:])

# 마지막 index 전까지 가져와라. (뒤는 비포함!)

# In[8]:


print(nums[:-1])

# assign 도 가능!

# In[9]:


nums[2:4] = [8, 9]

# In[10]:


print(nums)

# ## Slicing 2D Array

# In[11]:


import numpy as np

# In[12]:


b = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# In[13]:


print(b)

# In[14]:


b[:, 1]

# In[15]:


b[-1]

# In[16]:


b[-1, :]

# In[17]:


b[-1, ...]

# In[18]:


b[0:2, :]

# ## Loading Data from `.csv` file

# In[19]:


import numpy as np

# In[20]:


xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)

# In[21]:


x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# In[22]:


print(x_data.shape) # x_data shape
print(len(x_data))  # x_data 길이
print(x_data[:5])   # 첫 다섯 개

# In[23]:


print(y_data.shape) # y_data shape
print(len(y_data))  # y_data 길이
print(y_data[:5])   # 첫 다섯 개

# ## Imports

# In[24]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# In[25]:


# For reproducibility
torch.manual_seed(1)

# ## Low-level Implementation

# In[27]:


# 데이터
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
# 모델 초기화
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    
    # H(x) 계산
    hypothesis = x_train.matmul(W) + b # or .mm or @

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    print('Epoch {:4d}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, cost.item()
    ))

# ## High-level Implementation with `nn.Module`

# In[29]:


class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

# In[33]:


# 데이터
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
# 모델 초기화
model = MultivariateLinearRegressionModel()
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs+1):
    
    # H(x) 계산
    prediction = model(x_train)
    
    # cost 계산
    cost = F.mse_loss(prediction, y_train)
    
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    # 20번마다 로그 출력
    print('Epoch {:4d}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, cost.item()
    ))

# ## Dataset and DataLoader

# <div class="alert alert-warning">
#     pandas 기초지식이 필요할 것 같다
# </div>

# 너무 데이터가 크면 `x_data`, `y_data` 를 전부 다 가져오지 말고, 필요한 배치만 가져올 수 밖에 없다.

# [PyTorch Data Loading and Processing tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#iterating-through-the-dataset)
