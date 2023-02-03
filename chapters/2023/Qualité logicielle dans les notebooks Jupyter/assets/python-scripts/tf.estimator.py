#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
import numpy as np
tf.__version__

# In[4]:


train, _ = tf.keras.datasets.fashion_mnist.load_data()
images, labels = train
x = images/255

y=labels.astype(np.int32)

# In[5]:


feature_columns = [tf.feature_column.numeric_column("x", shape=[784])]

# In[6]:


train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(x={"x":x},y=y,num_epochs=None,shuffle=True)

# In[12]:


#classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=10)

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,hidden_units=[500, 500, 500], n_classes=10)

# In[13]:


classifier.train(input_fn=train_input_fn, steps=2000)

# In[ ]:



