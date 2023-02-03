#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install --upgrade pip

# In[ ]:


!pip install --upgrade tensorflow

# In[ ]:


import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Sequential

tf.__version__

# In[ ]:


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]


# In[ ]:


model = Sequential()
model.add(Conv2D(32, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10))



# In[ ]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[ ]:


model.fit(x_train, y_train, epochs=10)


# In[ ]:


tf.saved_model.save(model, "./mymodel/001234")

# In[ ]:


!tree ./mymodel

# # TensorFlow Serving using Docker

# In[ ]:


!docker pull tensorflow/serving
!docker run -t --rm -p 8501:8501 -v "$(pwd)/mymodel/:/models/mymodel" -e MODEL_NAME=mymodel tensorflow/serving

# # TensorFlow Lite

# In[ ]:


converter = tf.lite.TFLiteConverter.from_saved_model("./mymodel/001234")
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

# In[ ]:


from tflite_runtime.interpreter import Interpreter


# In[ ]:


interpreter = Interpreter("./converted_model.tflite")

# In[ ]:


interpreter.allocate_tensors()

# In[ ]:


import numpy as np
z = np.copy(x_train[0])
z.shape = (1,28, 28, 1)

# In[ ]:


tensor_index = interpreter.get_input_details()[0]['index']


# In[ ]:


input_tensor_z= tf.convert_to_tensor(z, np.float32)
interpreter.set_tensor(tensor_index, input_tensor_z)

# In[ ]:


interpreter.invoke()


# In[ ]:


output_details = interpreter.get_output_details()[0]

# In[ ]:


interpreter.get_tensor(output_details['index'])

# In[ ]:


y_train[0]

# # TensorFlow.js

# In[ ]:


!tensorflowjs_converter --input_format=tf_saved_model --output_node_names='mymodel' --saved_model_tags=serve mymodel/001234 mymodelweb

# In[ ]:


!tree mymodelweb/

# # TensorFlow Hub

# In[ ]:


import tensorflow_hub as hub

hub_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
embed = hub.KerasLayer(hub_url)
embeddings = embed(["A long sentence.", "single-word", "http://example.com"])
print(embeddings.shape, embeddings.dtype)


# In[ ]:


import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
model = keras.Sequential()
model.add(Dense(23, input_shape=(None,23)))
model.add(embed)
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.build()
model.summary()


# In[ ]:


model = tf.keras.Sequential([
    embed,
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])


# In[ ]:


model.summary()

# # TensorFlow Hub

# In[ ]:


model_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
hub_layer = hub.KerasLayer(model_url, output_shape=[128], input_shape=[], dtype=tf.string)

model = keras.Sequential()
model.add(hub_layer)
model.add(...)
...
model.summary()

