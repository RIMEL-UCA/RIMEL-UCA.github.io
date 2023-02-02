#!/usr/bin/env python
# coding: utf-8

# # 10-3-1 Visdom Example

# In[ ]:


import torch
import torch.nn as nn

import torchvision
import torchvision.datasets as dsets

# ## import visdom

# ![](figs/turn_on_terminal.png)
# 
# Jupyter Notebook > Terminal 를 새로 켜서 `python -m visdom.server` 를 입력하세요!

# In[ ]:


import visdom
vis = visdom.Visdom()

# ## Text

# In[ ]:


vis.text("Hello, world!",env="main")

# ## image

# In[ ]:


a=torch.randn(3,200,200)
vis.image(a)

# ## images

# In[ ]:


vis.images(torch.Tensor(3,3,28,28))

# ## example (using MNIST and CIFAR10)

# In[ ]:


# 시간이 좀 걸립니다.
MNIST = dsets.MNIST(root="./MNIST_data",train = True,transform=torchvision.transforms.ToTensor(), download=True)
cifar10 = dsets.CIFAR10(root="./cifar10",train = True, transform=torchvision.transforms.ToTensor(),download=True)

# #### CIFAR10

# In[ ]:


data = cifar10.__getitem__(0)
print(data[0].shape)
vis.images(data[0],env="main")

# #### MNIST

# In[ ]:


data = MNIST.__getitem__(0)
print(data[0].shape)
vis.images(data[0],env="main")

# #### Check dataset

# In[ ]:


data_loader = torch.utils.data.DataLoader(dataset = MNIST,
                                          batch_size = 32,
                                          shuffle = False)

# In[ ]:


for num, value in enumerate(data_loader):
    value = value[0]
    print(value.shape)
    vis.images(value)
    break

# In[ ]:


vis.close(env="main")

# ## Line Plot

# In[ ]:


Y_data = torch.randn(5)
plt = vis.line (Y=Y_data)

# In[ ]:


X_data = torch.Tensor([1,2,3,4,5])
plt = vis.line(Y=Y_data, X=X_data)

# ### Line update

# In[ ]:


Y_append = torch.randn(1)
X_append = torch.Tensor([6])

vis.line(Y=Y_append, X=X_append, win=plt, update='append')

# ### multiple Line on single windows

# In[ ]:


num = torch.Tensor(list(range(0,10)))
num = num.view(-1,1)
num = torch.cat((num,num),dim=1)

plt = vis.line(Y=torch.randn(10,2), X = num)

# ### Line info

# In[ ]:


plt = vis.line(Y=Y_data, X=X_data, opts = dict(title='Test', showlegend=True))

# In[ ]:


plt = vis.line(Y=Y_data, X=X_data, opts = dict(title='Test', legend = ['1번'],showlegend=True))

# In[ ]:


plt = vis.line(Y=torch.randn(10,2), X = num, opts=dict(title='Test', legend=['1번','2번'],showlegend=True))

# ## make function for update line

# In[ ]:


def loss_tracker(loss_plot, loss_value, num):
    '''num, loss_value, are Tensor'''
    vis.line(X=num,
             Y=loss_value,
             win = loss_plot,
             update='append'
             )

# In[ ]:


plt = vis.line(Y=torch.Tensor(1).zero_())

for i in range(500):
    loss = torch.randn(1) + i
    loss_tracker(plt, loss, torch.Tensor([i]))

# ## close the window

# In[ ]:


vis.close(env="main")

# In[ ]:



