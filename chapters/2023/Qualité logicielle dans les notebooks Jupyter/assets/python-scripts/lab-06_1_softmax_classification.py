#!/usr/bin/env python
# coding: utf-8

# # Lab 6-1: Softmax Classification

# Author: Seungjae Lee (이승재)

# <div class="alert alert-warning">
#     We use elemental PyTorch to implement linear regression here. However, in most actual applications, abstractions such as <code>nn.Module</code> or <code>nn.Linear</code> are used.
# </div>

# ## Imports

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# In[2]:


# For reproducibility
torch.manual_seed(1)

# ## Softmax

# Convert numbers to probabilities with softmax.

# $$ P(class=i) = \frac{e^i}{\sum e^i} $$

# In[3]:


z = torch.FloatTensor([1, 2, 3])

# PyTorch has a `softmax` function.

# In[4]:


hypothesis = F.softmax(z, dim=0)
print(hypothesis)

# Since they are probabilities, they should add up to 1. Let's do a sanity check.

# In[5]:


hypothesis.sum()

# ## Cross Entropy Loss (Low-level)

# For multi-class classification, we use the cross entropy loss.

# $$ L = \frac{1}{N} \sum - y \log(\hat{y}) $$

# where $\hat{y}$ is the predicted probability and $y$ is the correct probability (0 or 1).

# In[6]:


z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)
print(hypothesis)

# In[7]:


y = torch.randint(5, (3,)).long()
print(y)

# In[8]:


y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1), 1)

# In[9]:


cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)

# ## Cross-entropy Loss with `torch.nn.functional`

# PyTorch has `F.log_softmax()` function.

# In[10]:


# Low level
torch.log(F.softmax(z, dim=1))

# In[11]:


# High level
F.log_softmax(z, dim=1)

# PyTorch also has `F.nll_loss()` function that computes the negative loss likelihood.

# In[12]:


# Low level
(y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean()

# In[13]:


# High level
F.nll_loss(F.log_softmax(z, dim=1), y)

# PyTorch also has `F.cross_entropy` that combines `F.log_softmax()` and `F.nll_loss()`.

# In[14]:


F.cross_entropy(z, y)

# ## Training with Low-level Cross Entropy Loss

# In[15]:


x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

# In[16]:


# 모델 초기화
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # Cost 계산 (1)
    hypothesis = F.softmax(x_train.matmul(W) + b, dim=1) # or .mm or @
    y_one_hot = torch.zeros_like(hypothesis)
    y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
    cost = (y_one_hot * -torch.log(F.softmax(hypothesis, dim=1))).sum(dim=1).mean()

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# ## Training with `F.cross_entropy`

# In[17]:


# 모델 초기화
W = torch.zeros((4, 3), requires_grad=True)
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

# In[18]:


class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3) # Output이 3!

    def forward(self, x):
        return self.linear(x)

# In[19]:


model = SoftmaxClassifierModel()

# Let's try another new dataset.

# In[20]:


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
