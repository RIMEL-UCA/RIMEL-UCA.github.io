#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

# In[2]:


tf.__version__

# In[9]:


v1 = tf.constant([1,2,3],dtype=tf.float32)
v2 = tf.constant([4,5,6],dtype=tf.float32)

# In[10]:


v1*v2

# In[11]:


2*2

# In[17]:


tf.tensordot(v1,v2,axes=1)

# In[18]:


m1 = tf.constant([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])

# In[19]:


m1

# In[20]:


tf.tensordot(v1,m1,axes=1)

# In[23]:


m2 = tf.constant([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])

# In[25]:


tf.tensordot(m1,m2,axes=1)

# In[24]:


tf.tensordot(m1,m2,axes=2)

# In[132]:


x = tf.constant([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])
x_norm = tf.linalg.normalize(x,ord='euclidean')

y = tf.constant([2.,3.,4.])
y_norm = tf.linalg.normalize(y,ord='euclidean')

w = tf.Variable([1.,2.])

# In[43]:


x_norm

# In[30]:


def lr(x):
    return tf.tensordot(w,x,1)

# In[31]:


lr(x)

# In[32]:


y = tf.constant([0,1,0])

# In[33]:


def logreg(x):
    return tf.sigmoid(lr(x))

# In[34]:


logreg(x)

# In[133]:


w2 = tf.Variable([1.,1.,1.])
def layer2(a):
    return tf.sigmoid(tf.tensordot(a,w2,1))

# In[134]:


w1 = tf.Variable([[1,1,1],[1,1,1],[1,1,1]], dtype=tf.float32)

def layer1(x):
    return tf.sigmoid(tf.tensordot(x,w1,axes=1))


# In[135]:


def nn(x):
    return layer2(layer1(x))

# In[136]:


nn(x)

# In[137]:


def mse(y_pred, y_target):
    return tf.reduce_mean(tf.square(y_pred-y_target))

# In[138]:


mse(nn(x),y_norm[0])


# In[139]:


def train(w1, w2, x, y, lr=0.12):
    with tf.GradientTape() as t:
        current_loss = mse(y, nn(x))
        lr_w1, lr_w2  = t.gradient(current_loss, [w1,w2])
        w1.assign_sub(lr * lr_w1)
        w2.assign_sub(lr * lr_w2)

# In[140]:


for i in range(1000):
    train(w1,w2,x,y_norm[0])

# In[141]:


mse(nn(x),y_norm[0])


# In[142]:


nn(x)

# In[143]:


w2

# In[125]:


y_norm[0]

# In[ ]:



