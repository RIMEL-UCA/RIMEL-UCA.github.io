#!/usr/bin/env python
# coding: utf-8

# # Lab 7-1: Tips

# Author: Seungjae Lee (이승재)

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# In[2]:


# For reproducibility
torch.manual_seed(1)

# ## Training and Test Datasets

# In[3]:


x_train = torch.FloatTensor([[1, 2, 1],
                             [1, 3, 2],
                             [1, 3, 4],
                             [1, 5, 5],
                             [1, 7, 5],
                             [1, 2, 5],
                             [1, 6, 6],
                             [1, 7, 7]
                            ])
y_train = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0])

# In[4]:


x_test = torch.FloatTensor([[2, 1, 1], [3, 1, 2], [3, 3, 4]])
y_test = torch.LongTensor([2, 2, 2])

# ## Model

# In[5]:


class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)
    def forward(self, x):
        return self.linear(x)

# In[6]:


model = SoftmaxClassifierModel()

# In[7]:


# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)

# In[8]:


def train(model, optimizer, x_train, y_train):
    nb_epochs = 20
    for epoch in range(nb_epochs):

        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.cross_entropy(prediction, y_train)

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# In[9]:


def test(model, optimizer, x_test, y_test):
    prediction = model(x_test)
    predicted_classes = prediction.max(1)[1]
    correct_count = (predicted_classes == y_test).sum().item()
    cost = F.cross_entropy(prediction, y_test)

    print('Accuracy: {}% Cost: {:.6f}'.format(
         correct_count / len(y_test) * 100, cost.item()
    ))

# In[10]:


train(model, optimizer, x_train, y_train)

# In[11]:


test(model, optimizer, x_test, y_test)

# ## Learning Rate

# Gradient Descent 에서의 $\alpha$ 값

# `optimizer = optim.SGD(model.parameters(), lr=0.1)` 에서 `lr=0.1` 이다

# learning rate이 너무 크면 diverge 하면서 cost 가 점점 늘어난다 (overshooting).

# In[12]:


model = SoftmaxClassifierModel()

# In[13]:


optimizer = optim.SGD(model.parameters(), lr=1e5)

# In[14]:


train(model, optimizer, x_train, y_train)

# learning rate이 너무 작으면 cost가 거의 줄어들지 않는다.

# In[15]:


model = SoftmaxClassifierModel()

# In[16]:


optimizer = optim.SGD(model.parameters(), lr=1e-10)

# In[17]:


train(model, optimizer, x_train, y_train)

# 적절한 숫자로 시작해 발산하면 작게, cost가 줄어들지 않으면 크게 조정하자.

# In[18]:


model = SoftmaxClassifierModel()

# In[19]:


optimizer = optim.SGD(model.parameters(), lr=1e-1)

# In[20]:


train(model, optimizer, x_train, y_train)

# ## Data Preprocessing (데이터 전처리)

# 데이터를 zero-center하고 normalize하자.

# In[21]:


x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# $$ x'_j = \frac{x_j - \mu_j}{\sigma_j} $$

# 여기서 $\sigma$ 는 standard deviation, $\mu$ 는 평균값 이다.

# In[22]:


mu = x_train.mean(dim=0)

# In[23]:


sigma = x_train.std(dim=0)

# In[24]:


norm_x_train = (x_train - mu) / sigma

# In[25]:


print(norm_x_train)

# Normalize와 zero center한 X로 학습해서 성능을 보자

# In[26]:


class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

# In[27]:


model = MultivariateLinearRegressionModel()

# In[28]:


optimizer = optim.SGD(model.parameters(), lr=1e-1)

# In[29]:


def train(model, optimizer, x_train, y_train):
    nb_epochs = 20
    for epoch in range(nb_epochs):

        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.mse_loss(prediction, y_train)

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# In[30]:


train(model, optimizer, norm_x_train, y_train)

# ## Overfitting

# 너무 학습 데이터에 한해 잘 학습해 테스트 데이터에 좋은 성능을 내지 못할 수도 있다.

# 이것을 방지하는 방법은 크게 세 가지인데:
# 
# 1. 더 많은 학습 데이터
# 2. 더 적은 양의 feature
# 3. **Regularization**

# Regularization: Let's not have too big numbers in the weights

# In[31]:


def train_with_regularization(model, optimizer, x_train, y_train):
    nb_epochs = 20
    for epoch in range(nb_epochs):

        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.mse_loss(prediction, y_train)
        
        # l2 norm 계산
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.norm(param)
            
        cost += l2_reg

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch+1, nb_epochs, cost.item()
        ))

# In[32]:


model = MultivariateLinearRegressionModel()

# In[33]:


optimizer = optim.SGD(model.parameters(), lr=1e-1)

# In[34]:


train_with_regularization(model, optimizer, norm_x_train, y_train)
