#!/usr/bin/env python
# coding: utf-8

# # Gradient Boosted Decision Tree (GBDT)
# Implement a Gradient Boosted Decision Tree (GBDT) with TensorFlow. This example is using the Boston Housing Value dataset as training samples. The example supports both Classification (2 classes: value > $23000 or not) and Regression (raw home value as target).
# 
# - Author: Aymeric Damien
# - Project: https://github.com/aymericdamien/TensorFlow-Examples/

# ## Boston Housing Dataset
# 
# **Link:** https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
# 
# **Description:**
# 
# The dataset contains information collected by the U.S Census Service concerning housing in the area of Boston Mass. It was obtained from the StatLib archive (http://lib.stat.cmu.edu/datasets/boston), and has been used extensively throughout the literature to benchmark algorithms. However, these comparisons were primarily done outside of Delve and are thus somewhat suspect. The dataset is small in size with only 506 cases.
# 
# The data was originally published by Harrison, D. and Rubinfeld, D.L. `Hedonic prices and the demand for clean air', J. Environ. Economics & Management, vol.5, 81-102, 1978.`
# 
# *For the full features list, please see the link above*

# In[1]:


from __future__ import print_function

# Ignore all GPUs (current TF GBDT does not support GPU).
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"

import tensorflow as tf
import numpy as np
import copy

# In[2]:


# Dataset parameters.
num_classes = 2 # Total classes: greater or equal to $23,000, or not (See notes below).
num_features = 13 # data features size.

# Training parameters.
max_steps = 2000
batch_size = 256
learning_rate = 1.0
l1_regul = 0.0
l2_regul = 0.1

# GBDT parameters.
num_batches_per_layer = 1000
num_trees = 10
max_depth = 4

# In[3]:


# Prepare Boston Housing Dataset.
from tensorflow.keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# For classification purpose, we build 2 classes: price greater or lower than $23,000
def to_binary_class(y):
    for i, label in enumerate(y):
        if label >= 23.0:
            y[i] = 1
        else:
            y[i] = 0
    return y

y_train_binary = to_binary_class(copy.deepcopy(y_train))
y_test_binary = to_binary_class(copy.deepcopy(y_test))

# ### GBDT Classifier

# In[4]:


# Build the input function.
train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={'x': x_train}, y=y_train_binary,
    batch_size=batch_size, num_epochs=None, shuffle=True)
test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={'x': x_test}, y=y_test_binary,
    batch_size=batch_size, num_epochs=1, shuffle=False)
test_train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={'x': x_train}, y=y_train_binary,
    batch_size=batch_size, num_epochs=1, shuffle=False)
# GBDT Models from TF Estimator requires 'feature_column' data format.
feature_columns = [tf.feature_column.numeric_column(key='x', shape=(num_features,))]

# In[5]:


gbdt_classifier = tf.estimator.BoostedTreesClassifier(
    n_batches_per_layer=num_batches_per_layer,
    feature_columns=feature_columns, 
    n_classes=num_classes,
    learning_rate=learning_rate, 
    n_trees=num_trees,
    max_depth=max_depth,
    l1_regularization=l1_regul, 
    l2_regularization=l2_regul
)

# In[6]:


gbdt_classifier.train(train_input_fn, max_steps=max_steps)

# In[7]:


gbdt_classifier.evaluate(test_train_input_fn)

# In[8]:


gbdt_classifier.evaluate(test_input_fn)

# ### GBDT Regressor

# In[9]:


# Build the input function.
train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={'x': x_train}, y=y_train,
    batch_size=batch_size, num_epochs=None, shuffle=True)
test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={'x': x_test}, y=y_test,
    batch_size=batch_size, num_epochs=1, shuffle=False)
# GBDT Models from TF Estimator requires 'feature_column' data format.
feature_columns = [tf.feature_column.numeric_column(key='x', shape=(num_features,))]

# In[10]:


gbdt_regressor = tf.estimator.BoostedTreesRegressor(
    n_batches_per_layer=num_batches_per_layer,
    feature_columns=feature_columns, 
    learning_rate=learning_rate, 
    n_trees=num_trees,
    max_depth=max_depth,
    l1_regularization=l1_regul, 
    l2_regularization=l2_regul
)

# In[11]:


gbdt_regressor.train(train_input_fn, max_steps=max_steps)

# In[12]:


gbdt_regressor.evaluate(test_input_fn)
