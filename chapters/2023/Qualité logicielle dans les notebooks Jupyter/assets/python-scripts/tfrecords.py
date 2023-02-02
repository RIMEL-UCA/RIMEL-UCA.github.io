#!/usr/bin/env python
# coding: utf-8

# # Create and Load TFRecords
# 
# A simple TensorFlow 2.0 example to parse a dataset into TFRecord format, and then read that dataset.
# 
# In this example, the Titanic Dataset (in CSV format) will be used as a toy dataset, for parsing all the dataset features into TFRecord format, and then building an input pipeline that can be used for training models.
# 
# - Author: Aymeric Damien
# - Project: https://github.com/aymericdamien/TensorFlow-Examples/

# ## Titanic Dataset
# 
# The titanic dataset is a popular dataset for ML that provides a list of all passengers onboard the Titanic, along with various features such as their age, sex, class (1st, 2nd, 3rd)... And if the passenger survived the disaster or not.
# 
# It can be used to see that even though some luck was involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class...
# 
# #### Overview
# survived|pclass|name|sex|age|sibsp|parch|ticket|fare
# --------|------|----|---|---|-----|-----|------|----
# 1|1|"Allen, Miss. Elisabeth Walton"|female|29|0|0|24160|211.3375
# 1|1|"Allison, Master. Hudson Trevor"|male|0.9167|1|2|113781|151.5500
# 0|1|"Allison, Miss. Helen Loraine"|female|2|1|2|113781|151.5500
# 0|1|"Allison, Mr. Hudson Joshua Creighton"|male|30|1|2|113781|151.5500
# ...|...|...|...|...|...|...|...|...
# 
# 
# #### Variable Descriptions
# ```
# survived        Survived
#                 (0 = No; 1 = Yes)
# pclass          Passenger Class
#                 (1 = 1st; 2 = 2nd; 3 = 3rd)
# name            Name
# sex             Sex
# age             Age
# sibsp           Number of Siblings/Spouses Aboard
# parch           Number of Parents/Children Aboard
# ticket          Ticket Number
# fare            Passenger Fare
# ```

# In[ ]:


from __future__ import absolute_import, division, print_function

import csv
import requests
import tensorflow as tf

# In[ ]:


# Download Titanic dataset (in csv format).
d = requests.get("https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/titanic_dataset.csv")
with open("titanic_dataset.csv", "wb") as f:
    f.write(d.content)

# ### Create TFRecords

# In[ ]:


# Generate Integer Features.
def build_int64_feature(data):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[data]))

# Generate Float Features.
def build_float_feature(data):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[data]))

# Generate String Features.
def build_string_feature(data):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))

# Generate a TF `Example`, parsing all features of the dataset.
def convert_to_tfexample(survived, pclass, name, sex, age, sibsp, parch, ticket, fare):
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'survived': build_int64_feature(survived),
                'pclass': build_int64_feature(pclass),
                'name': build_string_feature(name),
                'sex': build_string_feature(sex),
                'age': build_float_feature(age),
                'sibsp': build_int64_feature(sibsp),
                'parch': build_int64_feature(parch),
                'ticket': build_string_feature(ticket),
                'fare': build_float_feature(fare),
            })
    )

# In[ ]:


# Open dataset file.
with open("titanic_dataset.csv") as f:
    # Output TFRecord file.
    with tf.io.TFRecordWriter("titanic_dataset.tfrecord") as w:
        # Generate a TF Example for all row in our dataset.
        # CSV reader will read and parse all rows.
        reader = csv.reader(f, skipinitialspace=True)
        for i, record in enumerate(reader):
            # Skip header.
            if i == 0:
                continue
            survived, pclass, name, sex, age, sibsp, parch, ticket, fare = record
            # Parse each csv row to TF Example using the above functions.
            example = convert_to_tfexample(int(survived), int(pclass), name, sex, float(age), int(sibsp), int(parch), ticket, float(fare))
            # Serialize each TF Example to string, and write to TFRecord file.
            w.write(example.SerializeToString())

# ### Load TFRecords

# In[ ]:


# Build features template, with types.
features = {
    'survived': tf.io.FixedLenFeature([], tf.int64),
    'pclass': tf.io.FixedLenFeature([], tf.int64),
    'name': tf.io.FixedLenFeature([], tf.string),
    'sex': tf.io.FixedLenFeature([], tf.string),
    'age': tf.io.FixedLenFeature([], tf.float32),
    'sibsp': tf.io.FixedLenFeature([], tf.int64),
    'parch': tf.io.FixedLenFeature([], tf.int64),
    'ticket': tf.io.FixedLenFeature([], tf.string),
    'fare': tf.io.FixedLenFeature([], tf.float32),
}

# In[ ]:


# Load TFRecord data.
filenames = ["titanic_dataset.tfrecord"]
data = tf.data.TFRecordDataset(filenames)

# Parse features, using the above template.
def parse_record(record):
    return tf.io.parse_single_example(record, features=features)
# Apply the parsing to each record from the dataset.
data = data.map(parse_record)

# Refill data indefinitely.
data = data.repeat()
# Shuffle data.
data = data.shuffle(buffer_size=1000)
# Batch data (aggregate records together).
data = data.batch(batch_size=4)
# Prefetch batch (pre-load batch for faster consumption).
data = data.prefetch(buffer_size=1)

# In[ ]:


# Dequeue data and display.
for record in data.take(1):
    print(record['survived'].numpy())
    print(record['name'].numpy())
    print(record['fare'].numpy())
