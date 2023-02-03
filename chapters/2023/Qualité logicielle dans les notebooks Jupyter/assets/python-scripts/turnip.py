#!/usr/bin/env python
# coding: utf-8

# http://cleverowl.uk/2019/06/02/maths-problem-the-turnip-seller/

# In[1]:


!pip install setuptools --upgrade
!pip install -q tensorflow==2.0.0-beta1

# In[2]:


import tensorflow as tf
tf.__version__

# In[48]:


# Import `tensorflow` 
import tensorflow as tf
import numpy as np


@tf.function
def step(x):
    return x-(6/7)*x-1/7

#print(tf.autograph.to_code(step.python_function))


w = tf.Variable(0, trainable=True, dtype=tf.float64)
y = tf.constant([0], dtype=tf.float64)

@tf.function
def compute(x):
    return step(step(step(step(x))))

#print(tf.autograph.to_code(compute.python_function))


variables = [w]
optimizer = tf.optimizers.Adam(0.5)

loss_object = tf.keras.losses.MeanAbsoluteError()

with tf.GradientTape(persistent=True) as tape:
    for i in range(1000):
        y_pred = compute(w)
        loss = loss_object(y,y_pred)
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))
        print(w.numpy())


