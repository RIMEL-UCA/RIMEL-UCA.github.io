#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from onnx import numpy_helper

def f(t):
    return [f(i) for i in t] if isinstance(t, (list, tuple)) else t

def g(t, res):
    for i in t:
        res.append(i) if not isinstance(i, (list, tuple)) else g(i, res)
    return res

def SaveData(test_data_dir, prefix, data_list):
    if isinstance(data_list, torch.autograd.Variable) or isinstance(data_list, torch.Tensor):
        data_list = [data_list]
    for i, d in enumerate(data_list):
        d = d.data.cpu().numpy()
        SaveTensorProto(os.path.join(test_data_dir, '{0}_{1}.pb'.format(prefix, i)), prefix + str(i+1), d)
        
def SaveTensorProto(file_path, name, data):
    tp = numpy_helper.from_array(data)
    tp.name = name

    with open(file_path, 'wb') as f:
        f.write(tp.SerializeToString())

# In[ ]:


import torch
import re
import os
import onnxruntime as rt
from transformer_net import TransformerNet

input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    model = TransformerNet()
    model_dict = torch.load("PATH TO PYTORCH MODEL")
    for k in list(model_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del model_dict[k]
    model.load_state_dict(model_dict)
    output = model(input)
    
input_names = ['input1']
output_names = ['output1']
dir = "PATH TO CONVERTED ONNX MODEL"
if not os.path.exists(dir):
    os.makedirs(dir)
data_dir = os.path.join(dir, "data_set")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if isinstance(model, torch.jit.ScriptModule):
    torch.onnx._export(model, tuple((input,)), os.path.join(dir, 'model.onnx'), verbose=True, input_names=input_names, output_names=output_names, example_outputs=(output,))
else:
    torch.onnx.export(model, tuple((input,)), os.path.join(dir, 'model.onnx'), verbose=True, input_names=input_names, output_names=output_names)

input = f(input)
input = g(input, [])
output = f(output)
output = g(output, [])
        
SaveData(data_dir, 'input', input)
SaveData(data_dir, 'output', output)
