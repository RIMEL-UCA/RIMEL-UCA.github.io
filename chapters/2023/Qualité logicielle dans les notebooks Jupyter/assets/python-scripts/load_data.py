#!/usr/bin/env python
# coding: utf-8

# # Load and parse data with TensorFlow 2.0 (tf.data)
# 
# A TensorFlow 2.0 example to build input pipelines for loading data efficiently.
# 
# 
# - Numpy Arrays
# - Images
# - CSV file
# - Custom data from a Generator
# 
# For more information about creating and loading TensorFlow's `TFRecords` data format, see: [tfrecords.ipynb](tfrecords.ipynb)
# 
# - Author: Aymeric Damien
# - Project: https://github.com/aymericdamien/TensorFlow-Examples/

# In[ ]:


from __future__ import absolute_import, division, print_function

import numpy as np
import random
import requests
import string
import tarfile
import tensorflow as tf

# ### Load Numpy Arrays
# 
# Build a data pipeline over numpy arrays.

# In[ ]:


# Create a toy dataset (even and odd numbers, with respective labels of 0 and 1).
evens = np.arange(0, 100, step=2, dtype=np.int32)
evens_label = np.zeros(50, dtype=np.int32)
odds = np.arange(1, 100, step=2, dtype=np.int32)
odds_label = np.ones(50, dtype=np.int32)
# Concatenate arrays
features = np.concatenate([evens, odds])
labels = np.concatenate([evens_label, odds_label])

# Load a numpy array using tf data api with `from_tensor_slices`.
data = tf.data.Dataset.from_tensor_slices((features, labels))
# Refill data indefinitely.  
data = data.repeat()
# Shuffle data.
data = data.shuffle(buffer_size=100)
# Batch data (aggregate records together).
data = data.batch(batch_size=4)
# Prefetch batch (pre-load batch for faster consumption).
data = data.prefetch(buffer_size=1)

# In[ ]:


for batch_x, batch_y in data.take(5):
    print(batch_x, batch_y)

# In[ ]:


# Note: If you are planning on calling multiple time,
# you can user the iterator way:
ite_data = iter(data)
for i in range(5):
    batch_x, batch_y = next(ite_data)
    print(batch_x, batch_y)

for i in range(5):
    batch_x, batch_y = next(ite_data)
    print(batch_x, batch_y)

# ### Load CSV files
# 
# Build a data pipeline from features stored in a CSV file. For this example, Titanic dataset will be used as a toy dataset stored in CSV format.

# #### Titanic Dataset
# 
# 
# 
# survived|pclass|name|sex|age|sibsp|parch|ticket|fare
# --------|------|----|---|---|-----|-----|------|----
# 1|1|"Allen, Miss. Elisabeth Walton"|female|29|0|0|24160|211.3375
# 1|1|"Allison, Master. Hudson Trevor"|male|0.9167|1|2|113781|151.5500
# 0|1|"Allison, Miss. Helen Loraine"|female|2|1|2|113781|151.5500
# 0|1|"Allison, Mr. Hudson Joshua Creighton"|male|30|1|2|113781|151.5500
# ...|...|...|...|...|...|...|...|...

# In[ ]:


# Download Titanic dataset (in csv format).
d = requests.get("https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/titanic_dataset.csv")
with open("titanic_dataset.csv", "wb") as f:
    f.write(d.content)

# In[ ]:


# Load Titanic dataset.
# Original features: survived,pclass,name,sex,age,sibsp,parch,ticket,fare
# Select specific columns: survived,pclass,name,sex,age,fare
column_to_use = [0, 1, 2, 3, 4, 8]
record_defaults = [tf.int32, tf.int32, tf.string, tf.string, tf.float32, tf.float32]

# Load the whole dataset file, and slice each line.
data = tf.data.experimental.CsvDataset("titanic_dataset.csv", record_defaults, header=True, select_cols=column_to_use)
# Refill data indefinitely.
data = data.repeat()
# Shuffle data.
data = data.shuffle(buffer_size=1000)
# Batch data (aggregate records together).
data = data.batch(batch_size=2)
# Prefetch batch (pre-load batch for faster consumption).
data = data.prefetch(buffer_size=1)

# In[ ]:


for survived, pclass, name, sex, age, fare in data.take(1):
    print(survived.numpy())
    print(pclass.numpy())
    print(name.numpy())
    print(sex.numpy())
    print(age.numpy())
    print(fare.numpy())

# ### Load Images
# 
# Build a data pipeline by loading images from disk. For this example, Oxford Flowers dataset will be used.

# In[ ]:


# Download Oxford 17 flowers dataset
d = requests.get("http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz")
with open("17flowers.tgz", "wb") as f:
    f.write(d.content)
# Extract archive.
with tarfile.open("17flowers.tgz") as t:
    t.extractall()

# In[ ]:


with open('jpg/dataset.csv', 'w') as f:
    c = 0
    for i in range(1360):
        f.write("jpg/image_%04i.jpg,%i\n" % (i+1, c))
        if (i+1) % 80 == 0:
            c += 1

# In[ ]:


# Load Images
with open("jpg/dataset.csv") as f:
    dataset_file = f.read().splitlines()

# Load the whole dataset file, and slice each line.
data = tf.data.Dataset.from_tensor_slices(dataset_file)
# Refill data indefinitely.
data = data.repeat()
# Shuffle data.
data = data.shuffle(buffer_size=1000)

# Load and pre-process images.
def load_image(path):
    # Read image from path.
    image = tf.io.read_file(path)
    # Decode the jpeg image to array [0, 255].
    image = tf.image.decode_jpeg(image)
    # Resize images to a common size of 256x256.
    image = tf.image.resize(image, [256, 256])
    # Rescale values to [-1, 1].
    image = 1. - image / 127.5
    return image
# Decode each line from the dataset file.
def parse_records(line):
    # File is in csv format: "image_path,label_id".
    # TensorFlow requires a default value, but it will never be used.
    image_path, image_label = tf.io.decode_csv(line, ["", 0])
    # Apply the function to load images.
    image = load_image(image_path)
    return image, image_label
# Use 'map' to apply the above functions in parallel.
data = data.map(parse_records, num_parallel_calls=4)

# Batch data (aggregate images-array together).
data = data.batch(batch_size=2)
# Prefetch batch (pre-load batch for faster consumption).
data = data.prefetch(buffer_size=1)

# In[ ]:


for batch_x, batch_y in data.take(1):
    print(batch_x, batch_y)

# ### Load data from a Generator

# In[ ]:


# Create a dummy generator.
def generate_features():
    # Function to generate a random string.
    def random_string(length):
        return ''.join(random.choice(string.ascii_letters) for m in xrange(length))
    # Return a random string, a random vector, and a random int.
    yield random_string(4), np.random.uniform(size=4), random.randint(0, 10)

# In[ ]:


# Load a numpy array using tf data api with `from_tensor_slices`.
data = tf.data.Dataset.from_generator(generate_features, output_types=(tf.string, tf.float32, tf.int32))
# Refill data indefinitely.
data = data.repeat()
# Shuffle data.
data = data.shuffle(buffer_size=100)
# Batch data (aggregate records together).
data = data.batch(batch_size=4)
# Prefetch batch (pre-load batch for faster consumption).
data = data.prefetch(buffer_size=1)

# In[ ]:


# Display data.
for batch_str, batch_vector, batch_int in data.take(5):
    print(batch_str, batch_vector, batch_int)
