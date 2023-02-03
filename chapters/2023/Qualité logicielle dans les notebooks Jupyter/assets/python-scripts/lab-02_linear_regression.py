#!/usr/bin/env python
# coding: utf-8

# # Lab 2: Linear Regression

# Author: Seungjae Lee (이승재)

# <div class="alert alert-warning">
#     We use elemental PyTorch to implement linear regression here. However, in most actual applications, abstractions such as <code>nn.Module</code> or <code>nn.Linear</code> are used.
# </div>

# ## Theoretical Overview

# $$ H(x) = Wx + b $$

# $$ cost(W, b) = \frac{1}{m} \sum^m_{i=1} \left( H(x^{(i)}) - y^{(i)} \right)^2 $$

#  - $H(x)$: 주어진 $x$ 값에 대해 예측을 어떻게 할 것인가
#  - $cost(W, b)$: $H(x)$ 가 $y$ 를 얼마나 잘 예측했는가

# ## Imports

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# In[2]:


# For reproducibility
torch.manual_seed(1)

# ## Data

# We will use fake data for this example.

# In[3]:


x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

# In[4]:


print(x_train)
print(x_train.shape)

# In[5]:


print(y_train)
print(y_train.shape)

# 기본적으로 PyTorch는 NCHW 형태이다.

# ## Weight Initialization

# In[6]:


W = torch.zeros(1, requires_grad=True)
print(W)

# In[7]:


b = torch.zeros(1, requires_grad=True)
print(b)

# ## Hypothesis

# $$ H(x) = Wx + b $$

# In[8]:


hypothesis = x_train * W + b
print(hypothesis)

# ## Cost

# $$ cost(W, b) = \frac{1}{m} \sum^m_{i=1} \left( H(x^{(i)}) - y^{(i)} \right)^2 $$

# In[9]:


print(hypothesis)

# In[10]:


print(y_train)

# In[11]:


print(hypothesis - y_train)

# In[12]:


print((hypothesis - y_train) ** 2)

# In[13]:


cost = torch.mean((hypothesis - y_train) ** 2)
print(cost)

# ## Gradient Descent

# In[14]:


optimizer = optim.SGD([W, b], lr=0.01)

# In[15]:


optimizer.zero_grad()
cost.backward()
optimizer.step()

# In[16]:


print(W)
print(b)

# Let's check if the hypothesis is now better.

# In[17]:


hypothesis = x_train * W + b
print(hypothesis)

# In[18]:


cost = torch.mean((hypothesis - y_train) ** 2)
print(cost)

# ## Training with Full Code

# In reality, we will be training on the dataset for multiple epochs. This can be done simply with loops.

# In[19]:


# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])
# 모델 초기화
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.01)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    
    # H(x) 계산
    hypothesis = x_train * W + b
    
    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))

# ## High-level Implementation with `nn.Module`

# Remember that we had this fake data.

# In[20]:


x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

# 이제 linear regression 모델을 만들면 되는데, 기본적으로 PyTorch의 모든 모델은 제공되는 `nn.Module`을 inherit 해서 만들게 됩니다.

# In[21]:


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# 모델의 `__init__`에서는 사용할 레이어들을 정의하게 됩니다. 여기서 우리는 linear regression 모델을 만들기 때문에, `nn.Linear` 를 이용할 것입니다. 그리고 `forward`에서는 이 모델이 어떻게 입력값에서 출력값을 계산하는지 알려줍니다.

# In[22]:


model = LinearRegressionModel()

# ## Hypothesis

# 이제 모델을 생성해서 예측값 $H(x)$를 구해보자

# In[23]:


hypothesis = model(x_train)

# In[24]:


print(hypothesis)

# ## Cost

# 이제 mean squared error (MSE) 로 cost를 구해보자. MSE 역시 PyTorch에서 기본적으로 제공한다.

# In[25]:


print(hypothesis)
print(y_train)

# In[26]:


cost = F.mse_loss(hypothesis, y_train)

# In[27]:


print(cost)

# ## Gradient Descent

# 마지막 주어진 cost를 이용해 $H(x)$ 의 $W, b$ 를 바꾸어서 cost를 줄여봅니다. 이때 PyTorch의 `torch.optim` 에 있는 `optimizer` 들 중 하나를 사용할 수 있습니다.

# In[28]:


optimizer = optim.SGD(model.parameters(), lr=0.01)

# In[29]:


optimizer.zero_grad()
cost.backward()
optimizer.step()

# ## Training with Full Code

# 이제 Linear Regression 코드를 이해했으니, 실제로 코드를 돌려 피팅시켜보겠습니다.

# In[30]:


# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])
# 모델 초기화
model = LinearRegressionModel()
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.01)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    
    # H(x) 계산
    prediction = model(x_train)
    
    # cost 계산
    cost = F.mse_loss(prediction, y_train)
    
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    # 100번마다 로그 출력
    if epoch % 100 == 0:
        params = list(model.parameters())
        W = params[0].item()
        b = params[1].item()
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W, b, cost.item()
        ))

# 점점 $H(x)$ 의 $W$ 와 $b$ 를 조정해서 cost가 줄어드는 것을 볼 수 있습니다.
