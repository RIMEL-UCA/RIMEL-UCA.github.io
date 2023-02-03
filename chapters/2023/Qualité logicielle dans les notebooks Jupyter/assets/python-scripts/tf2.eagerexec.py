#!/usr/bin/env python
# coding: utf-8

# To run this notebook in IBM Watson Studio (for free), click on the button below and then click on the same button again (top right).
# 
# In case you need a (free, never expiring, no credit card needed) IBM Cloud account, you can get one here: [ibm.biz/coursera](https://ibm.biz/coursera) (tracked URL)
# 
# [![](https://github.com/romeokienzler/TensorFlow/raw/master/images/playbutton.png)](https://dataplatform.cloud.ibm.com/analytics/notebooks/v2/12358b29-5273-48c9-8e5e-bc5c195dc0da/view?access_token=22727a749d9add9bde97ec483decacf0c267ef798d43aba9a0c860cf28287158)
# 
# 
# 

# First of all, let's install TensorFlow 1.3.0 in order to have a look into the world before Eager Execution was supported

# In[ ]:


!pip install tensorflow==1.3.0

# Now just make sure you restart the Kernel so that the changes take effect:
# 
# ![](https://github.com/romeokienzler/TensorFlow/raw/master/images/restart_kernel.png)

# Now let's double-check if we are running the correct version of TensorFlow

# In[ ]:


import tensorflow as tf
import numpy as np

# In[ ]:


tf.__version__

# If you are on Version 1.3.0, just execute the next cell. You will notice that we've created an object **a** of type __tensorflow.python.framework.ops.Tensor__

# In[ ]:


a = tf.constant(np.array([1., 2., 3.]))
print(type(a))

# Let's create another Tensor **b** and apply the dot product between them. This gives us __c__

# In[ ]:


b = tf.constant(np.array([4.,5.,6.]))
c = tf.tensordot(a, b,1)
type(c)

# Note that **c** is a __tensorflow.python.framework.ops.Tensor__ as well. So any node of the execution graph resembles a Tensor type. But so far not a single computation happened. You need to execute the graph. You can pass any graph or subgraph to the TensorFlow runtime for execution. Each TensorFlow graph runs within a TensorFlow Session, therefore we need to create it first: 

# In[ ]:


session = tf.Session()
output = session.run(c)
session.close()
print(output)

# Now you see the correct result of 32. But the problem is that debugging is pretty hard if you can only run complete graphs, so let's consider the following code:

# In[ ]:


a = tf.constant(np.array([1., 2., 3.]))
b = tf.constant(np.array([4.,5.]))
c = tf.tensordot(a, b,1)
session = tf.Session()
output = session.run(c)

# This code threw an expetion. The error is __Matrix size-incompatible: In[0]: [1,3], In[1]: [2,1]__. This is nice information, the only problem is, that the error is complaining about an operation which was executed within the exection graph and not in your code (since your code was only used to create the exection graph). This is what I mean be **hard to debug**. So let's actually switch to TensorFlow 2.0. This (as of today) is still alpha code, so in case something doesn't work please feel free to create an issue on the [GitHub](https://github.com/romeokienzler/TensorFlow/issues) page.

# In[ ]:


!pip install tensorflow==2.0.0-alpha0 

# And again, please don't forget to restart the Kernel so that the changes take effect:
# 
# ![](https://github.com/romeokienzler/TensorFlow/raw/master/images/restart_kernel.png)

# In[ ]:


import tensorflow as tf
import numpy as np

# In[ ]:


tf.__version__

# Now you should see __'2.0.0-alpha0'__ als version. So we are good to go. Let's actually run the same code again and check out if we can see any difference:

# In[ ]:


a = tf.constant(np.array([1., 2., 3.]))
print(type(a))


# So the very same code created a different type of object. So now **a** is of type __tensorflow.python.framework.ops.EagerTensor__. This is cool, because without changing code we obtain a tensor object which allows us to have a look inside, without execting a graph in a session:

# In[ ]:


print(a.numpy())

# Isn't this cool? So from now on we can thread Tensors like ordinary python object, work with then as usual, insert debug statements at any point or even use a debugger. So let's continue this example for sake of completeness:

# In[ ]:


b = tf.constant(np.array([4.,5.,6.]))
c = tf.tensordot(a, b,1)
type(c)

# Again, **c** is an __tensorflow.python.framework.ops.EagerTensor__ object which can be directly read:

# In[ ]:


print(c.numpy())

# Very cool. Without creating a session or a graph we obtained the result of the defined computation. So let's re-introduce the error we had before in order to see how debug capabilities are changing:

# In[ ]:


a = tf.constant(np.array([1., 2., 3.]))
b = tf.constant(np.array([4.,5.]))
c = tf.tensordot(a, b,1)

# As you can see, we obtained the same error message, but now we get a clear indication where the error occured. So that's it for now, stay tuned and have fun. Please contineoue to read on https://github.com/romeokienzler/TensorFlow
