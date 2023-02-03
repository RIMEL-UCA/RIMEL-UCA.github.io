#!/usr/bin/env python
# coding: utf-8

# # Lab 1: Tensor Manipulation

# First Author: Seungjae Ryan Lee (seungjaeryanlee at gmail dot com)
# Second Author: Ki Hyun Kim (nlp.with.deep.learning at gmail dot com)

# <div class="alert alert-warning">
#     NOTE: This corresponds to <a href="https://www.youtube.com/watch?v=ZYX0FaqUeN4&t=23s&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=25">Lab 8 of Deep Learning Zero to All Season 1 for TensorFlow</a>.
# </div>

# ## Imports

# Run `pip install -r requirements.txt` in terminal to install all required Python packages.

# In[1]:


import numpy as np
import torch

# ## NumPy Review

# We hope that you are familiar with `numpy` and basic linear algebra.

# ### 1D Array with NumPy

# In[2]:


t = np.array([0., 1., 2., 3., 4., 5., 6.])
print(t)

# In[3]:


print('Rank  of t: ', t.ndim)
print('Shape of t: ', t.shape)

# In[4]:


print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1]) # Element
print('t[2:5] t[4:-1]  = ', t[2:5], t[4:-1])   # Slicing
print('t[:2] t[3:]     = ', t[:2], t[3:])      # Slicing

# ### 2D Array with NumPy

# In[5]:


t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print(t)

# In[6]:


print('Rank  of t: ', t.ndim)
print('Shape of t: ', t.shape)

# ## PyTorch is like NumPy (but better)

# ### 1D Array with PyTorch

# In[7]:


t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)

# In[8]:


print(t.dim())  # rank
print(t.shape)  # shape
print(t.size()) # shape
print(t[0], t[1], t[-1])  # Element
print(t[2:5], t[4:-1])    # Slicing
print(t[:2], t[3:])       # Slicing

# ### 2D Array with PyTorch

# In[9]:


t = torch.FloatTensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]
                      ])
print(t)

# In[10]:


print(t.dim())  # rank
print(t.size()) # shape
print(t[:, 1])
print(t[:, 1].size())
print(t[:, :-1])

# ### Shape, Rank, Axis

# In[11]:


t = torch.FloatTensor([[[[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12]],
                       [[13, 14, 15, 16],
                        [17, 18, 19, 20],
                        [21, 22, 23, 24]]
                       ]])

# In[12]:


print(t.dim())  # rank  = 4
print(t.size()) # shape = (1, 2, 3, 4)

# ## Frequently Used Operations in PyTorch

# ### Mul vs. Matmul

# In[13]:


print()
print('-------------')
print('Mul vs Matmul')
print('-------------')
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1.matmul(m2)) # 2 x 1

m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1 * m2) # 2 x 2
print(m1.mul(m2))

# ### Broadcasting

# <div class="alert alert-warning">
#     Carelessly using broadcasting can lead to code hard to debug.
# </div>

# In[14]:


# Same shape
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)

# In[15]:


# Vector + scalar
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3]) # 3 -> [[3, 3]]
print(m1 + m2)

# In[16]:


# 2 x 1 Vector + 1 x 2 Vector
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)

# ### Mean

# In[17]:


t = torch.FloatTensor([1, 2])
print(t.mean())

# In[18]:


# Can't use mean() on integers
t = torch.LongTensor([1, 2])
try:
    print(t.mean())
except Exception as exc:
    print(exc)

# You can also use `t.mean` for higher rank tensors to get mean of all elements, or mean by particular dimension.

# In[19]:


t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

# In[20]:


print(t.mean())
print(t.mean(dim=0))
print(t.mean(dim=1))
print(t.mean(dim=-1))

# ### Sum

# In[21]:


t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

# In[22]:


print(t.sum())
print(t.sum(dim=0))
print(t.sum(dim=1))
print(t.sum(dim=-1))

# ### Max and Argmax

# In[23]:


t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

# The `max` operator returns one value if it is called without an argument.

# In[24]:


print(t.max()) # Returns one value: max

# The `max` operator returns 2 values when called with dimension specified. The first value is the maximum value, and the second value is the argmax: the index of the element with maximum value.

# In[25]:


print(t.max(dim=0)) # Returns two values: max and argmax
print('Max: ', t.max(dim=0)[0])
print('Argmax: ', t.max(dim=0)[1])

# In[26]:


print(t.max(dim=1))
print(t.max(dim=-1))

# ### View

# <div class="alert alert-warning">
#     This is a function hard to master, but is very useful!
# </div>

# In[27]:


t = np.array([[[0, 1, 2],
               [3, 4, 5]],

              [[6, 7, 8],
               [9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft.shape)

# In[28]:


print(ft.view([-1, 3]))
print(ft.view([-1, 3]).shape)

# In[29]:


print(ft.view([-1, 1, 3]))
print(ft.view([-1, 1, 3]).shape)

# ### Squeeze

# In[30]:


ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)

# In[31]:


print(ft.squeeze())
print(ft.squeeze().shape)

# ### Unsqueeze

# In[32]:


ft = torch.Tensor([0, 1, 2])
print(ft.shape)

# In[33]:


print(ft.unsqueeze(0))
print(ft.unsqueeze(0).shape)

# In[34]:


print(ft.view(1, -1))
print(ft.view(1, -1).shape)

# In[35]:


print(ft.unsqueeze(1))
print(ft.unsqueeze(1).shape)

# In[36]:


print(ft.unsqueeze(-1))
print(ft.unsqueeze(-1).shape)

# ### Scatter (for one-hot encoding)

# <div class="alert alert-warning">
#     Scatter is a very flexible function. We only discuss how to use it to get a one-hot encoding of indices.
# </div>

# In[37]:


lt = torch.LongTensor([[0], [1], [2], [0]])
print(lt)

# In[38]:


one_hot = torch.zeros(4, 3) # batch_size = 4, classes = 3
one_hot.scatter_(1, lt, 1)
print(one_hot)

# ### Casting

# In[39]:


lt = torch.LongTensor([1, 2, 3, 4])
print(lt)

# In[40]:


print(lt.float())

# In[41]:


bt = torch.ByteTensor([True, False, False, True])
print(bt)

# In[42]:


print(bt.long())
print(bt.float())

# ### Concatenation

# In[43]:


x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])

# In[44]:


print(torch.cat([x, y], dim=0))
print(torch.cat([x, y], dim=1))

# ### Stacking

# In[45]:


x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])

# In[46]:


print(torch.stack([x, y, z]))
print(torch.stack([x, y, z], dim=1))

# In[47]:


print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))

# ### Ones and Zeros Like

# In[48]:


x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)

# In[49]:


print(torch.ones_like(x))
print(torch.zeros_like(x))

# ### In-place Operation

# In[50]:


x = torch.FloatTensor([[1, 2], [3, 4]])

# In[51]:


print(x.mul(2.))
print(x)
print(x.mul_(2.))
print(x)

# ## Miscellaneous

# ### Zip

# In[52]:


for x, y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y)

# In[53]:


for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z)
