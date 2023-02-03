#!/usr/bin/env python
# coding: utf-8

# # Hello World
# 
# A very simple "hello world" using TensorFlow v2 tensors.
# 
# - Author: Aymeric Damien
# - Project: https://github.com/aymericdamien/TensorFlow-Examples/

# In[1]:


import tensorflow as tf

# In[2]:


# Create a Tensor.
hello = tf.constant("hello world")
print(hello)

# In[3]:


# To access a Tensor value, call numpy().
print(hello.numpy())
