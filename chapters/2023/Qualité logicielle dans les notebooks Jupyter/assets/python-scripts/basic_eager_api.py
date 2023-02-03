#!/usr/bin/env python
# coding: utf-8

# # Basic introduction to TensorFlow's Eager API
# 
# A simple introduction to get started with TensorFlow's Eager API.
# 
# - Author: Aymeric Damien
# - Project: https://github.com/aymericdamien/TensorFlow-Examples/

# ### What is TensorFlow's Eager API ?
# 
# *Eager execution is an imperative, define-by-run interface where operations are
# executed immediately as they are called from Python. This makes it easier to
# get started with TensorFlow, and can make research and development more
# intuitive. A vast majority of the TensorFlow API remains the same whether eager
# execution is enabled or not. As a result, the exact same code that constructs
# TensorFlow graphs (e.g. using the layers API) can be executed imperatively
# by using eager execution. Conversely, most models written with Eager enabled
# can be converted to a graph that can be further optimized and/or extracted
# for deployment in production without changing code. - Rajat Monga*
# 
# More info: https://research.googleblog.com/2017/10/eager-execution-imperative-define-by.html

# In[1]:


from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

# In[2]:


# Set Eager API
print("Setting Eager mode...")
tf.enable_eager_execution()
tfe = tf.contrib.eager

# In[3]:


# Define constant tensors
print("Define constant tensors")
a = tf.constant(2)
print("a = %i" % a)
b = tf.constant(3)
print("b = %i" % b)

# In[4]:


# Run the operation without the need for tf.Session
print("Running operations, without tf.Session")
c = a + b
print("a + b = %i" % c)
d = a * b
print("a * b = %i" % d)

# In[5]:


# Full compatibility with Numpy
print("Mixing operations with Tensors and Numpy Arrays")

# Define constant tensors
a = tf.constant([[2., 1.],
                 [1., 0.]], dtype=tf.float32)
print("Tensor:\n a = %s" % a)
b = np.array([[3., 0.],
              [5., 1.]], dtype=np.float32)
print("NumpyArray:\n b = %s" % b)

# In[6]:


# Run the operation without the need for tf.Session
print("Running operations, without tf.Session")

c = a + b
print("a + b = %s" % c)

d = tf.matmul(a, b)
print("a * b = %s" % d)

# In[7]:


print("Iterate through Tensor 'a':")
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        print(a[i][j])
