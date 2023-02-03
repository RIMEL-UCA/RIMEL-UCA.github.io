#!/usr/bin/env python
# coding: utf-8

# 
# # MNIST Dataset Introduction
# 
# Most examples are using MNIST dataset of handwritten digits. The dataset contains 60,000 examples for training and 10,000 examples for testing. The digits have been size-normalized and centered in a fixed-size image (28x28 pixels) with values from 0 to 1. For simplicity, each image has been flatten and converted to a 1-D numpy array of 784 features (28*28).
# 
# ## Overview
# 
# ![MNIST Digits](http://neuralnetworksanddeeplearning.com/images/mnist_100_digits.png)
# 
# ## Usage
# In our examples, we are using TensorFlow [input_data.py](https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/examples/tutorials/mnist/input_data.py) script to load that dataset.
# It is quite useful for managing our data, and handle:
# 
# - Dataset downloading
# 
# - Loading the entire dataset into numpy array: 
# 
# 
# 

# In[ ]:


# Import MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Load data
X_train = mnist.train.images
Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels

# - A `next_batch` function that can iterate over the whole dataset and return only the desired fraction of the dataset samples (in order to save memory and avoid to load the entire dataset).

# In[ ]:


# Get the next 64 images array and labels
batch_X, batch_Y = mnist.train.next_batch(64)

# Link: http://yann.lecun.com/exdb/mnist/
