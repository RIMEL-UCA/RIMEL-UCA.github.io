#!/usr/bin/env python
# coding: utf-8

# In[1]:


import onnx
import os

# Load the ONNX model
model = onnx.load(os.path.join('model.onnx'))

# In[2]:


onnx.checker.check_model(model)  # Check that the IR is well formed
print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
   
