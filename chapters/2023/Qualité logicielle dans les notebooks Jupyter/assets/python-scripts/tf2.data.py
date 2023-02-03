#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np

# In[ ]:


tf.__version__

# In[ ]:


dataset = tf.data.Dataset.from_tensor_slices(np.array(range(10)))


# In[ ]:


for item in dataset:
    print(item.numpy())


# In[ ]:


!rm -Rf tmp
!mkdir tmp
!echo test,label > ./tmp/chap9.csv
!for i in `seq 1000000`; do echo "$i,$i" >> ./tmp/chap9.csv; done

# In[ ]:


!ls -lahr ./tmp/chap9.csv

# In[ ]:


csv_dataset = tf.data.experimental.make_csv_dataset(
    '/Users/romeokienzler/Downloads/chap9.csv', batch_size=4,
    label_name="label")


# In[ ]:


for line in csv_dataset.take(1):
    print(line)


# In[ ]:


csv_dataset.make_one_shot_iterator()

# In[ ]:


import matplotlib.pyplot as plt
def plot_batch_sizes(ds):
    batch_sizes = [batch.shape[0] for batch in ds]
    plt.bar(range(len(batch_sizes)), batch_sizes)
    plt.xlabel('Batch number')
    plt.ylabel('Batch size')


# In[ ]:


plot_batch_sizes(csv_dataset)


# In[ ]:



