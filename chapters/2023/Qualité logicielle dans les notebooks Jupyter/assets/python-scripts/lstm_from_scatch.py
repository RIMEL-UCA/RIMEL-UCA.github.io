#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf

# In[3]:


tf.__version__

# In[97]:


w_f = tf.Variable([1.])
w_i = tf.Variable([1.])
w_o = tf.Variable([1.])
w_c = tf.Variable([1.])
u_f = tf.Variable([1.])
u_i = tf.Variable([1.])
u_o = tf.Variable([1.])
u_c = tf.Variable([1.])
b_f = tf.Variable([1.])
b_i = tf.Variable([1.])
b_o = tf.Variable([1.])
b_c = tf.Variable([1.])
x_t = tf.constant([23.])
h_t_1 = tf.constant([1.])
c_t_1 = tf.constant([1.])

# In[98]:


def lstm_cell(x_t,h_t_1, c_t_1):  
    f_t = tf.sigmoid(tf.tensordot(w_f,x_t,1) + tf.tensordot(u_f,h_t_1,1) + b_f)
    i_t = tf.sigmoid(tf.tensordot(w_i,x_t,1) + tf.tensordot(u_i,h_t_1,1) + b_i)
    o_t = tf.sigmoid(tf.tensordot(w_o,x_t,1) + tf.tensordot(u_o,h_t_1,1) + b_o)
    tf.tensordot(w_c,x_t,1)
    tf.tensordot(u_c,h_t_1,1)
    temp = tf.sigmoid(tf.tensordot(w_c,x_t,1) + tf.tensordot(u_c,h_t_1,1) + b_c)
    tf.tensordot(i_t,temp,1)
    c_t = [tf.tensordot(f_t,c_t_1,1) + tf.tensordot(i_t,temp,1)]
    h_t = [tf.tensordot(o_t, tf.sigmoid(c_t),1)]
    return (h_t, c_t)

# In[99]:


x_t_1 = tf.constant([0.5])
x_t_2 = tf.constant([0.7])
h_t_1 = tf.constant([1.])
c_t_1 = tf.constant([1.])


# In[100]:


def step():
    (h_t, c_t) = lstm_cell(x_t_1, h_t_1, c_t_1)
    (h_t_p1, c_t_p1) = lstm_cell(x_t_2, h_t, c_t)
    return (h_t, h_t_p1)

# In[101]:


def loss(x_t_1,x_t_2,h_t_1, h_t_2):
    return (tf.square(x_t_1-h_t_1)+tf.square(x_t_2-h_t_2))/2

# In[102]:


def train():
    with tf.GradientTape() as t:
        lr = 0.1
        (h_t_1, h_t_2) = step()
        current_loss = loss(x_t_1,x_t_2,h_t_1, h_t_2)
        print(f"Loss: {current_loss}")
        g_w_f,g_w_i,g_w_o,g_w_c,g_u_f,g_u_i,g_u_o,g_u_c,g_b_f,g_b_i,g_b_o,g_b_c = t.gradient(current_loss, [w_f,w_i,w_o,w_c,u_f,u_i,u_o,u_c,b_f,b_i,b_o,b_c])
        w_f.assign_sub(lr * g_w_f)
        w_i.assign_sub(lr * g_w_i)
        w_o.assign_sub(lr * g_w_o)
        w_c.assign_sub(lr * g_w_c)
        u_f.assign_sub(lr * g_u_f)
        u_i.assign_sub(lr * g_u_i)
        u_o.assign_sub(lr * g_u_o)
        u_c.assign_sub(lr * g_u_c)
        b_f.assign_sub(lr * g_b_f)
        b_i.assign_sub(lr * g_b_i)
        b_o.assign_sub(lr * g_b_o)
        b_c.assign_sub(lr * g_b_c)

# In[108]:


for i in range(10000):
    train()

# In[107]:


step()


# In[142]:


nn(x)

# In[143]:


w2

# In[125]:


y_norm[0]

# In[ ]:



