#!/usr/bin/env python
# coding: utf-8

# To run this notebook in IBM Watson Studio (for free), click on the button below and then click on the same button again (top right).
# 
# In case you need a (free, never expiring, no credit card needed) IBM Cloud account, you can get one here: [ibm.biz/coursera](https://ibm.biz/coursera) (tracked URL)
# 
# [![](https://github.com/romeokienzler/TensorFlow/raw/master/images/playbutton.png)](https://dataplatform.cloud.ibm.com/analytics/notebooks/v2/d94668e2-3d68-4ee2-bc87-00a885803726/view?access_token=76adb4b998e7cb7bed5e53b0c1a99fb99a0b0a7a24498881e766c094845f0366)
# 
# 
# 

# Now it's time to install TensorFlow 2.x - as of writing of this notebook, there is only an alpha version available

# In[ ]:


!pip install tensorflow==2.0.0-alpha0 

# Now just make sure you restart the Kernel so that the changes take effect:
# 
# ![](https://github.com/romeokienzler/TensorFlow/raw/master/images/restart_kernel.png)
# 
# After the kernel has been restarted, we'll check if we are on TensorFlow 2.x

# In[1]:


import tensorflow as tf

# In[2]:


tf.__version__

# So this worked out. Now it's time to create and run a keras model. Since MNIST is getting boring, let's use the fashion MNIST dataset. If you want to learn more on the data set check out the following links
# 
# [MNIST](https://en.wikipedia.org/wiki/MNIST_database)  
# [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)
# 
# So in a nutsthell, MNIST contains 60000 28x28 pixel grey scale images of handwritten digits between 0-9 and the corresponding labels. Plus additional 10000 images for testing.
# 
# Fashing MNIST contains 60000 28x28 pixel grey scale images of fashion articles and the corresponding labels between 0-9. It also contains 10000 test images.
# 
# Luckyly, this data set is built in to Keras, so let's load it:

# In[3]:


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# As expected, we get 60000 images of 28 by 28 pixels:

# In[4]:


train_images.shape

# The labels are simply a list of 60000 elements, each one is a number (label) between 0 and 9:

# In[5]:


print(train_labels.shape)
print(train_labels)

# Let's have a look at one image:

# In[6]:


import matplotlib.pyplot as plt

plt.figure()
plt.imshow(train_images[0])
plt.show()


# So this is obviously a shoe :) - Let's normalize the data by making sure every pixel value is between 0..1; this is easy in this case:

# In[7]:


train_images = train_images / 255.0

test_images = test_images / 255.0


# For those already familiar with Keras might notice, that mainly only the import statements changed. If you are new to Keras, just check out Week 2 of my DeepLearning course on Coursera, you can get it for free in audit mode: http://coursera.org/learn/ai/. So let's start with ordinary Keras by importing some requirements:

# In[8]:


from keras import Sequential
from keras.layers import Flatten, Dense
from keras.activations import relu, softmax

# In order to write our **hello world** softmax regression model, the following code does the job. If you are familiar with Keras, this is really basic stuff. There is only one catch. The following code doesn't run since the latest stable Keras version is incompatible with the alpha release of TensorFlow 2.0. So the following code is for illustration purposes only. Don't run it, it will destroy your hard drive.

# In[9]:


model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation=relu),
    Dense(10, activation=softmax)
])

model.compile(optimizer='adam', 
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

# So from a migration and consistency perspective, it would be nice if we just could change the imports and leave our existing Keras code (which we all love) intact, so let's give it a try:

# In[11]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.nn import relu
from tensorflow.nn import softmax


    
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation=relu),
    Dense(10, activation=softmax)
])

model.compile(optimizer='adam', 
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

model.evaluate(test_images, test_labels)

# As you can see, we didn't change the Keras code at all, but now all imports are coming from the tensorflow package. I felt a bit bad when I've noticed that TensorFlow has eaten up Keras, but in reality, nobody uses a Keras runtime other then TensorFlow anyway, so it doesn't really matter. Just be aware that Keras is Google now and part of TensorFlow. In the next notebook, we'll cover the stategies for parallel training. So stay tuned.
