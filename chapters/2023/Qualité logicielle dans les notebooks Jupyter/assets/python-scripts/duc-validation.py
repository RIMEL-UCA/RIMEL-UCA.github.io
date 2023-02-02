#!/usr/bin/env python
# coding: utf-8

# # Validation notebook for DUC models
# 
# ## Overview
# Use this notebook to verify the accuracy of a trained DUC model in ONNX format on the validation set of cityscapes dataset.
# 
# ## Models supported
# * ResNet101_DUC_HDC
# 
# ## Prerequisites
# The following packages need to be installed before proceeding:
# * Protobuf compiler - `sudo apt-get install protobuf-compiler libprotoc-dev` (required for ONNX. This will work for any linux system. For detailed installation guidelines head over to [ONNX documentation](https://github.com/onnx/onnx#installation))
# * ONNX - `pip install onnx`
# * MXNet - `pip install mxnet-cu90mkl --pre -U` (tested on this version GPU, can use other versions. `--pre` indicates a pre build of MXNet which is required here for ONNX version compatibility. `-U` uninstalls any existing MXNet version allowing for a clean install)
# * numpy - `pip install numpy`
# * OpenCV - `pip install opencv-python`
# * PIL - `pip install pillow`
# 
# Also the following scripts (included in the repo) must be present in the same folder as this notebook:
# * `cityscapes_loader.py` (load and prepare validation images and labels)
# * `utils.py` (helper script used by `cityscapes_loader.py`)
# * `cityscapes_labels.py` (contains segmentation category labels)
# 
# The validation set of Cityscapes must be prepared before proceeding. Follow guidelines in the [dataset](README.md/#dset) section.
# 
# In order to do inference with a python script:
# * Generate the script : In Jupyter Notebook browser, go to File -> Download as -> Python (.py)
# * Run the script: `python duc-validation.py`

# ### Import dependencies
# Verify that all dependencies are installed using the cell below. Continue if no errors encountered, warnings can be ignored.

# In[1]:


from __future__ import print_function
import mxnet as mx
import numpy as np
import glob
import os
from mxnet.contrib.onnx import import_model
from cityscapes_loader import CityLoader

# ### Set paths and parameters
# Prepare the validation set of cityscapes according to the guidelines provided in [dataset](README.md/#dset) section. Set paths `data_dir` and `label_dir` accordingly.

# In[2]:


# Determine and set context
if len(mx.test_utils.list_gpus())==0:
    ctx = mx.cpu()
else:
    ctx = mx.gpu(0)

# Path to validation data
data_dir = '/home/ubuntu/TuSimple-DUC/dataset/leftImg8bit/val'
# Path to validation labels
label_dir = '/home/ubuntu/TuSimple-DUC/dataset/gtFine/val'
# Set batch size
batch_size = 16

# ### Download ONNX model

# In[ ]:


mx.test_utils.download('https://s3.amazonaws.com/onnx-model-zoo/duc/ResNet101_DUC_HDC.onnx')
# Path to ONNX model
model_path = 'ResNet101_DUC_HDC.onnx'

# ### Prepare dataset list
# Prepare validation images list (val.lst) containing image and label paths along with cropping metrics.

# In[ ]:


index = 0
val_lst = []
# images
all_images = glob.glob(os.path.join(data_dir, '*/*.png'))
all_images.sort()
for p in all_images:
    l = p.replace(data_dir, label_dir).replace('leftImg8bit', 'gtFine_labelIds')
    if os.path.isfile(l):
        index += 1
        for i in range(1, 8):
            val_lst.append([str(index), p, l, "512", str(256 * i)])

val_out = open('val.lst', "w")
for line in val_lst:
    print('\t'.join(line),file=val_out)

# ### Define evaluation metric
# `class IoUMetric` : Defines mean Intersection Over Union (mIOU) custom evaluation metric.
# 
# `check_label_shapes` : Checks the shape of target labels and network output.

# In[3]:


def check_label_shapes(labels, preds, shape=0):
    if shape == 0:
        label_shape, pred_shape = len(labels), len(preds)
    else:
        label_shape, pred_shape = labels.shape, preds.shape

    if label_shape != pred_shape:
        raise ValueError("Shape of labels {} does not match shape of "
                         "predictions {}".format(label_shape, pred_shape))

class IoUMetric(mx.metric.EvalMetric):
    def __init__(self, ignore_label, label_num, name='IoU'):
        self._ignore_label = ignore_label
        self._label_num = label_num
        super(IoUMetric, self).__init__(name=name)

    def reset(self):
        self._tp = [0.0] * self._label_num
        self._denom = [0.0] * self._label_num

    def update(self, labels, preds):
        check_label_shapes(labels, preds)
        for i in range(len(labels)):
            pred_label = mx.ndarray.argmax_channel(preds[i]).asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')

            check_label_shapes(label, pred_label)

            iou = 0
            eps = 1e-6
            for j in range(self._label_num):
                pred_cur = (pred_label.flat == j)
                gt_cur = (label.flat == j)
                tp = np.logical_and(pred_cur, gt_cur).sum()
                denom = np.logical_or(pred_cur, gt_cur).sum() - np.logical_and(pred_cur, label.flat == self._ignore_label).sum()
                assert tp <= denom
                self._tp[j] += tp
                self._denom[j] += denom
                iou += self._tp[j] / (self._denom[j] + eps)
            iou /= self._label_num
            self.sum_metric = iou
            self.num_inst = 1

            
# Create evaluation metric
met = IoUMetric(ignore_label=255, label_num=19, name="IoU")
metric = mx.metric.create(met)

# ### Configure data loader
# An object of CityLoader class (inherited from mx.io.DataIter) is instantiated for loading and precessing the validation data.

# In[4]:


loader = CityLoader
val_args = {
    'data_path'             : data_dir,
    'label_path'            : label_dir,
    'rgb_mean'              : (122.675, 116.669, 104.008),
    'batch_size'            : batch_size,
    'scale_factors'         : [1],
    'data_name'             : 'data',
    'label_name'            : 'seg_loss_label',
    'data_shape'            : [tuple(list([batch_size, 3, 800, 800]))],
    'label_shape'           : [tuple([batch_size, (160000)])],
    'use_random_crop'       : False,
    'use_mirror'            : False,
    'ds_rate'               : 8,
    'convert_label'         : True,
    'multi_thread'          : False,
    'cell_width'            : 2,
    'random_bound'          : [120,120],
}
val_dataloader = loader('val.lst', val_args)

# ### Load ONNX model

# In[5]:


# import ONNX model into MXNet symbols and params
sym,arg,aux = import_model(model_path)
# define network module
mod = mx.mod.Module(symbol=sym, data_names=['data'], context=ctx, label_names=None)
# bind parameters to the network
mod.bind(for_training=False, data_shapes=[('data', (batch_size, 3, 800, 800))], label_shapes=mod._label_shapes)
mod.set_params(arg_params=arg, aux_params=aux,allow_missing=True, allow_extra=True)

# ### Compute evaluations
# Perform forward pass over each batch and generate evaluations

# In[6]:


# reset data loader
val_dataloader.reset()
# reset evaluation metric
metric.reset()
# loop over batches
for nbatch, eval_batch in enumerate(val_dataloader):
    # perform forward pass
    mod.forward(eval_batch, is_train=False)
    # get outputs
    outputs=mod.get_outputs()
    # update evaluation metric
    metric.update(eval_batch.label,outputs)
    # print progress
    if nbatch%10==0:
        print('{} / {} batches done'.format(nbatch,int(3500/batch_size)))

# ### Print results

# In[7]:


print("mean Intersection Over Union (mIOU): {}".format(metric.get()[1]))

# In[ ]:



