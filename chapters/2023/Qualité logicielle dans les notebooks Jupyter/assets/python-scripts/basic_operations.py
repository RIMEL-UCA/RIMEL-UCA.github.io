#!/usr/bin/env python
# coding: utf-8

# # Basic Tensor Operations
# 
# Basic tensor operations using TensorFlow v2.
# 
# - Author: Aymeric Damien
# - Project: https://github.com/aymericdamien/TensorFlow-Examples/

# In[1]:


from __future__ import print_function
import tensorflow as tf

# In[2]:


# Define tensor constants.
a = tf.constant(2)
b = tf.constant(3)
c = tf.constant(5)

# In[3]:


# Various tensor operations.
# Note: Tensors also support python operators (+, *, ...)
add = tf.add(a, b)
sub = tf.subtract(a, b)
mul = tf.multiply(a, b)
div = tf.divide(a, b)

# Access tensors value.
print("add =", add.numpy())
print("sub =", sub.numpy())
print("mul =", mul.numpy())
print("div =", div.numpy())

# In[4]:


# Some more operations.
mean = tf.reduce_mean([a, b, c])
sum = tf.reduce_sum([a, b, c])

# Access tensors value.
print("mean =", mean.numpy())
print("sum =", sum.numpy())

# In[5]:


# Matrix multiplications.
matrix1 = tf.constant([[1., 2.], [3., 4.]])
matrix2 = tf.constant([[5., 6.], [7., 8.]])

product = tf.matmul(matrix1, matrix2)

# In[6]:


# Display Tensor.
product

# In[7]:


# Convert Tensor to Numpy.
product.numpy()
