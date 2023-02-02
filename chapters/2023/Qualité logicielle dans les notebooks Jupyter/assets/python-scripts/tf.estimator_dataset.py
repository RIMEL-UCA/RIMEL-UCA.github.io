#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
tf.__version__

# # THIS NOTEBOOK IS CURRENTLY BROKEN - use tf.estimator.ipynb instead (using numpy arrays of tf.data.Dataset)

# In[11]:


train, _ = tf.keras.datasets.fashion_mnist.load_data()
images, labels = train
images = images/255

dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset

# In[10]:


iter(dataset).next()[0]

# In[111]:


def pre_process(image_tensor, label_tensor):
    #flatten
    image_tensor = tf.reshape(image_tensor,(28**2,))
    
    #tensor to tensordict
    tensor_dict = dict()
    for feature_idx in range(784):
        key = str(feature_idx)
        value = tf.slice(image_tensor,[feature_idx],[1])
        tensor_dict.update({key:value})
        
    #one hot encode labels, cast to int32
    label_tensor = tf.one_hot(label_tensor,10,dtype=tf.int32)
    
    #reshape
    #label_tensor = tf.reshape(label_tensor,[1,10])
    
    return (tensor_dict,label_tensor)

# In[52]:


features = []
for feature_idx in range(784):
    features.append(tf.feature_column.numeric_column(str(feature_idx)))

# In[115]:


estimator = tf.estimator.DNNClassifier(feature_columns=features, hidden_units = [100], n_classes=10)


# In[116]:



@tf.function
def get_data():
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset_preprocessed = dataset.map(pre_process)
    return dataset_preprocessed
    

estimator.train(input_fn=get_data, steps=2000)


# In[ ]:



