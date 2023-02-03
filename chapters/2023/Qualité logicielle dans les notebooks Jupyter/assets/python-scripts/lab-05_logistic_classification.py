#!/usr/bin/env python
# coding: utf-8

# # Lab 5: Logistic Classification

# Author: Seungjae Lee (이승재)

# <div class="alert alert-warning">
#     We use elemental PyTorch to implement linear regression here. However, in most actual applications, abstractions such as <code>nn.Module</code> or <code>nn.Linear</code> are used. You can see those implementations near the end of this notebook.
# </div>

# ## Reminder: Logistic Regression

# ### Hypothesis

# $$ H(X) = \frac{1}{1+e^{-W^T X}} $$

# ### Cost

# $$ cost(W) = -\frac{1}{m} \sum y \log\left(H(x)\right) + (1-y) \left( \log(1-H(x) \right) $$

#  - If $y \simeq H(x)$, cost is near 0.
#  - If $y \neq H(x)$, cost is high.

# ### Weight Update via Gradient Descent

# $$ W := W - \alpha \frac{\partial}{\partial W} cost(W) $$

#  - $\alpha$: Learning rate

# ## Imports

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# In[2]:


# For reproducibility
torch.manual_seed(1)

# ## Training Data

# In[3]:


x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

# Consider the following classification problem: given the number of hours each student spent watching the lecture and working in the code lab, predict whether the student passed or failed a course. For example, the first (index 0) student watched the lecture for 1 hour and spent 2 hours in the lab session ([1, 2]), and ended up failing the course ([0]).

# In[4]:


x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# As always, we need these data to be in `torch.Tensor` format, so we convert them.

# In[5]:


print(x_train.shape)
print(y_train.shape)

# ## Computing the Hypothesis

# $$ H(X) = \frac{1}{1+e^{-W^T X}} $$

# PyTorch has a `torch.exp()` function that resembles the exponential function.

# In[6]:


print('e^1 equals: ', torch.exp(torch.FloatTensor([1])))

# We can use it to compute the hypothesis function conveniently.

# In[7]:


W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# In[8]:


hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))

# In[9]:


print(hypothesis)
print(hypothesis.shape)

# Or, we could use `torch.sigmoid()` function! This resembles the sigmoid function:

# In[10]:


print('1/(1+e^{-1}) equals: ', torch.sigmoid(torch.FloatTensor([1])))

# Now, the code for hypothesis function is cleaner.

# In[11]:


hypothesis = torch.sigmoid(x_train.matmul(W) + b)

# In[12]:


print(hypothesis)
print(hypothesis.shape)

# ## Computing the Cost Function (Low-level)

# $$ cost(W) = -\frac{1}{m} \sum y \log\left(H(x)\right) + (1-y) \left( \log(1-H(x) \right) $$

# We want to measure the difference between `hypothesis` and `y_train`.

# In[13]:


print(hypothesis)
print(y_train)

# For one element, the loss can be computed as follows:

# In[14]:


-(y_train[0] * torch.log(hypothesis[0]) + 
  (1 - y_train[0]) * torch.log(1 - hypothesis[0]))

# To compute the losses for the entire batch, we can simply input the entire vector.

# In[15]:


losses = -(y_train * torch.log(hypothesis) + 
           (1 - y_train) * torch.log(1 - hypothesis))
print(losses)

# Then, we just `.mean()` to take the mean of these individual losses.

# In[16]:


cost = losses.mean()
print(cost)

# ## Computing the Cost Function with `F.binary_cross_entropy`

# In reality, binary classification is used so often that PyTorch has a simple function called `F.binary_cross_entropy` implemented to lighten the burden.

# In[17]:


F.binary_cross_entropy(hypothesis, y_train)

# ## Training with Low-level Binary Cross Entropy Loss

# In[18]:


x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# In[19]:


# 모델 초기화
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # Cost 계산
    hypothesis = torch.sigmoid(x_train.matmul(W) + b) # or .mm or @
    cost = -(y_train * torch.log(hypothesis) + 
             (1 - y_train) * torch.log(1 - hypothesis)).mean()

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# ## Training with `F.binary_cross_entropy`

# In[20]:


# 모델 초기화
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # Cost 계산
    hypothesis = torch.sigmoid(x_train.matmul(W) + b) # or .mm or @
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# ## Loading Real Data

# In[21]:


import numpy as np

# In[22]:


xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# In[23]:


print(x_train[0:5])
print(y_train[0:5])

# ## Training with Real Data using low-level Binary Cross Entropy Loss

# In[24]:


# 모델 초기화
W = torch.zeros((8, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 100
for epoch in range(nb_epochs + 1):

    # Cost 계산
    hypothesis = torch.sigmoid(x_train.matmul(W) + b) # or .mm or @
    cost = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis)).mean()

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 10번마다 로그 출력
    if epoch % 10 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# ## Training with Real Data using `F.binary_cross_entropy`

# In[25]:


# 모델 초기화
W = torch.zeros((8, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 100
for epoch in range(nb_epochs + 1):

    # Cost 계산
    hypothesis = torch.sigmoid(x_train.matmul(W) + b) # or .mm or @
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 10번마다 로그 출력
    if epoch % 10 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# ## Checking the Accuracy our our Model

# After we finish training the model, we want to check how well our model fits the training set.

# In[26]:


hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis[:5])

# We can change **hypothesis** (real number from 0 to 1) to **binary predictions** (either 0 or 1) by comparing them to 0.5.

# In[27]:


prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction[:5])

# Then, we compare it with the correct labels `y_train`.

# In[28]:


print(prediction[:5])
print(y_train[:5])

# In[29]:


correct_prediction = prediction.float() == y_train
print(correct_prediction[:5])

# Finally, we can calculate the accuracy by counting the number of correct predictions and dividng by total number of predictions.

# In[30]:


accuracy = correct_prediction.sum().item() / len(correct_prediction)
print('The model has an accuracy of {:2.2f}% for the training set.'.format(accuracy * 100))

# ## Optional: High-level Implementation with `nn.Module`

# In[31]:


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

# In[32]:


model = BinaryClassifier()

# In[33]:


# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 100
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = model(x_train)

    # cost 계산
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    # 20번마다 로그 출력
    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(
            epoch, nb_epochs, cost.item(), accuracy * 100,
        ))

