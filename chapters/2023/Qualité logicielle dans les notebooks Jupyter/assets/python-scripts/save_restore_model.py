#!/usr/bin/env python
# coding: utf-8

# # Save & Restore a Model
# 
# Save and Restore a model using TensorFlow v2. In this example, we will go over both low and high-level approaches: 
# - Low-level: TF Checkpoint.
# - High-level: TF Module/Model saver.
# 
# This example is using the MNIST database of handwritten digits as toy dataset
# (http://yann.lecun.com/exdb/mnist/).
# 
# - Author: Aymeric Damien
# - Project: https://github.com/aymericdamien/TensorFlow-Examples/

# In[1]:


from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

# In[2]:


# MNIST dataset parameters.
num_classes = 10 # 0 to 9 digits
num_features = 784 # 28*28

# Training parameters.
learning_rate = 0.01
training_steps = 1000
batch_size = 256
display_step = 50

# In[3]:


# Prepare MNIST data.
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Flatten images to 1-D vector of 784 features (28*28).
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.

# In[4]:


# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

# ## 1) TF Checkpoint
# 
# Basic logistic regression

# In[5]:


# Weight of shape [784, 10], the 28*28 image features, and total number of classes.
W = tf.Variable(tf.random.normal([num_features, num_classes]), name="weight")
# Bias of shape [10], the total number of classes.
b = tf.Variable(tf.zeros([num_classes]), name="bias")

# Logistic regression (Wx + b).
def logistic_regression(x):
    # Apply softmax to normalize the logits to a probability distribution.
    return tf.nn.softmax(tf.matmul(x, W) + b)

# Cross-Entropy loss function.
def cross_entropy(y_pred, y_true):
    # Encode label to a one hot vector.
    y_true = tf.one_hot(y_true, depth=num_classes)
    # Clip prediction values to avoid log(0) error.
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    # Compute cross-entropy.
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Adam optimizer.
optimizer = tf.optimizers.Adam(learning_rate)

# In[6]:


# Optimization process. 
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = logistic_regression(x)
        loss = cross_entropy(pred, y)

        # Compute gradients.
        gradients = g.gradient(loss, [W, b])

        # Update W and b following gradients.
        optimizer.apply_gradients(zip(gradients, [W, b]))

# In[7]:


# Run training for the given number of steps.
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # Run the optimization to update W and b values.
    run_optimization(batch_x, batch_y)
    
    if step % display_step == 0:
        pred = logistic_regression(batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

# ## Save and Load with TF Checkpoint

# In[8]:


# Save weights and optimizer variables.
# Create a dict of variables to save.
vars_to_save = {"W": W, "b": b, "optimizer": optimizer}
# TF Checkpoint, pass the dict as **kwargs.
checkpoint = tf.train.Checkpoint(**vars_to_save)
# TF CheckpointManager to manage saving parameters.
saver = tf.train.CheckpointManager(
      checkpoint, directory="./tf-example", max_to_keep=5)

# In[9]:


# Save variables.
saver.save()

# In[10]:


# Check weight value.
np.mean(W.numpy())

# In[11]:


# Reset variables to test restore.
W = tf.Variable(tf.random.normal([num_features, num_classes]), name="weight")
b = tf.Variable(tf.zeros([num_classes]), name="bias")

# In[12]:


# Check resetted weight value.
np.mean(W.numpy())

# In[13]:


# Set checkpoint to load data.
vars_to_load = {"W": W, "b": b, "optimizer": optimizer}
checkpoint = tf.train.Checkpoint(**vars_to_load)
# Restore variables from latest checkpoint.
latest_ckpt = tf.train.latest_checkpoint("./tf-example")
checkpoint.restore(latest_ckpt)

# In[14]:


# Confirm that W has been correctly restored.
np.mean(W.numpy())

# ## 2) TF Model
# 
# Basic neural network with TF Model

# In[15]:


from tensorflow.keras import Model, layers

# In[16]:


# MNIST dataset parameters.
num_classes = 10 # 0 to 9 digits
num_features = 784 # 28*28

# Training parameters.
learning_rate = 0.01
training_steps = 1000
batch_size = 256
display_step = 100

# In[17]:


# Create TF Model.
class NeuralNet(Model):
    # Set layers.
    def __init__(self):
        super(NeuralNet, self).__init__(name="NeuralNet")
        # First fully-connected hidden layer.
        self.fc1 = layers.Dense(64, activation=tf.nn.relu)
        # Second fully-connected hidden layer.
        self.fc2 = layers.Dense(128, activation=tf.nn.relu)
        # Third fully-connecter hidden layer.
        self.out = layers.Dense(num_classes, activation=tf.nn.softmax)

    # Set forward pass.
    def __call__(self, x, is_training=False):
        x = self.fc1(x)
        x = self.out(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x

# Build neural network model.
neural_net = NeuralNet()

# In[18]:


# Cross-Entropy loss function.
def cross_entropy(y_pred, y_true):
    y_true = tf.cast(y_true, tf.int64)
    crossentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return tf.reduce_mean(crossentropy)

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Adam optimizer.
optimizer = tf.optimizers.Adam(learning_rate)

# In[19]:


# Optimization process. 
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = neural_net(x, is_training=True)
        loss = cross_entropy(pred, y)

        # Compute gradients.
        gradients = g.gradient(loss, neural_net.trainable_variables)

        # Update W and b following gradients.
        optimizer.apply_gradients(zip(gradients, neural_net.trainable_variables))

# In[20]:


# Run training for the given number of steps.
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # Run the optimization to update W and b values.
    run_optimization(batch_x, batch_y)
    
    if step % display_step == 0:
        pred = neural_net(batch_x, is_training=False)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

# ## Save and Load with TF Model

# In[21]:


# Save TF model.
neural_net.save_weights(filepath="./tfmodel.ckpt")

# In[22]:


# Re-build neural network model with default values.
neural_net = NeuralNet()
# Test model performance.
pred = neural_net(batch_x)
print("accuracy: %f" % accuracy(pred, batch_y))

# In[23]:


# Load saved weights.
neural_net.load_weights(filepath="./tfmodel.ckpt")

# In[24]:


# Test that weights loaded correctly.
pred = neural_net(batch_x)
print("accuracy: %f" % accuracy(pred, batch_y))
