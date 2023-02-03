#!/usr/bin/env python
# coding: utf-8

# # Running roberta-movie-sentiment model
# 
# This tutorial shows how to run the roberta-movie-sentiment model on Onnxruntime.
# 
# To see how the roberta-movie-sentiment model was converted from tensorflow to onnx look at [roBERTatutorial.ipynb](https://github.com/SeldonIO/seldon-models/blob/master/pytorch/moviesentiment_roberta/pytorch-roberta-onnx.ipynb)

# # Step 1 - Preprocess
# 
# Extract parameters from the given input and convert it into features.

# In[1]:


import torch
import numpy as np
from simpletransformers.model import TransformerModel
from transformers import RobertaForSequenceClassification, RobertaTokenizer

text = "This film is so good"
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)  # Batch size 1


# # Step 2 - Run the ONNX model under onnxruntime
# 
# Create an onnx inference session and run the model

# In[2]:


import onnxruntime

# Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
# other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
# based on the build flags) when instantiating InferenceSession.
# For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
# onnxruntime.InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])
ort_session = onnxruntime.InferenceSession("roberta-sequence-classification-9.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_ids)}
ort_out = ort_session.run(None, ort_inputs)

# # Step 3 - Postprocessing
# 
# Print the results

# In[3]:


pred = np.argmax(ort_out)
if(pred == 0):
    print("Prediction: negative")
elif(pred == 1):
    print("Prediction: positive")

# In[ ]:



