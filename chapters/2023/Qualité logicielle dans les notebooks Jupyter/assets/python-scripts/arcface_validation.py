#!/usr/bin/env python
# coding: utf-8

# # Validation notebook for ArcFace models
# 
# ## Overview
# Use this notebook to verify the accuracy of a trained ArcFace model in ONNX format on the validation datasets.
# 
# ## Models supported
# * LResNet100E-IR (ResNet100 backend with ArcFace loss)
# 
# ## Prerequisites
# The following packages need to be installed before proceeding:
# * Protobuf compiler - `sudo apt-get install protobuf-compiler libprotoc-dev` (required for ONNX. This will work for any linux system. For detailed installation guidelines head over to [ONNX documentation](https://github.com/onnx/onnx#installation))
# * ONNX - `pip install onnx`
# * MXNet - `pip install mxnet-cu90mkl --pre -U` (tested on this version GPU, can use other versions. `--pre` indicates a pre build of MXNet which is required here for ONNX version compatibility. `-U` uninstalls any existing MXNet version allowing for a clean install)
# * numpy - `pip install numpy`
# * matplotlib - `pip install matplotlib`
# * OpenCV - `pip install opencv-python`
# * Scikit-learn - `pip install scikit-learn`
# * EasyDict - `pip install easydict`
# 
# In order to do inference with a python script:
# * Generate the script : In Jupyter Notebook browser, go to File -> Download as -> Python (.py)
# * Run the script: `python arcface_validation.py`

# ### Import dependencies
# Verify that all dependencies are installed using the cell below. Continue if no errors encountered, warnings can be ignored.

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate
import sklearn
import cv2
import math
import datetime
import pickle
import mxnet as mx
from mxnet import ndarray as nd
from easydict import EasyDict as edict
from mxnet.contrib.onnx.onnx2mx.import_model import import_model

# ### Data loading helper code
# * `load_bin()` loads validation datasets
# * `load_property()` loads information like image size, number of identities from the property file in the dataset folder

# In[2]:


def load_bin(path, image_size):
    try:
        # python 3
        bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    except:
        # python 2
        bins, issame_list = pickle.load(open(path, 'rb'))
    data_list = []
    for flip in [0,1]:
        data = nd.empty((len(issame_list)*2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for i in range(len(issame_list)*2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin)
        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0,1]:
            if flip==1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][i][:] = img
        if i%1000==0:
            print('loading bin', i)
    print(data_list[0].shape)
    return (data_list, issame_list)

def load_property(data_dir):
    prop = edict()
    for line in open(os.path.join(data_dir, 'property')):
        vec = line.strip().split(',')
        assert len(vec)==3
        prop.num_classes = int(vec[0])
        prop.image_size = [int(vec[1]), int(vec[2])]
    return prop

# ### Evaluation helper code
# * `class LFold` is used to split the data for K-fold crossvalidation
# * `calculate_roc()` computes accuracy on each fold of cross validation using `calculate_accuracy()`
# * `calculate_accuracy()` computes the actual accuracy for test samples by thresholding the distance between embedding vectors of a test image pair and comparing the output with the ground truth for the pair
# * `evaluate()` splits embeddings into test pairs and computes accuracies using `calculate_roc()`

# In[3]:


class LFold:
    def __init__(self, n_splits = 2, shuffle = False):
        self.n_splits = n_splits
        if self.n_splits>1:
            self.k_fold = KFold(n_splits = n_splits, shuffle = shuffle)

    def split(self, indices):
        if self.n_splits>1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)
    
    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff),1)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):        
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
          
    tpr = np.mean(tprs,0)
    fpr = np.mean(fprs,0)
    return tpr, fpr, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc

def evaluate(embeddings, actual_issame, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds)
    return accuracy

# ### Define test function
# `test()` takes a validation set `data_set` and the MXNet model `mx_model` as input and computes accuracies using `evaluate()` on the set using `nfolds` cross validation.

# In[4]:


def test(data_set, mx_model, batch_size, nfolds=10, data_extra = None, label_shape = None):
    print('testing verification..')
    data_list = data_set[0]
    issame_list = data_set[1]
    model = mx_model
    embeddings_list = []
    if data_extra is not None:
        _data_extra = nd.array(data_extra)
    time_consumed = 0.0
    if label_shape is None:
        _label = nd.ones( (batch_size,) )
    else:
        _label = nd.ones( label_shape )
    for i in range( len(data_list) ):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba<data.shape[0]:
            bb = min(ba+batch_size, data.shape[0])
            count = bb-ba
            _data = nd.slice_axis(data, axis=0, begin=bb-batch_size, end=bb)
            time0 = datetime.datetime.now()
            if data_extra is None:
                db = mx.io.DataBatch(data=(_data,), label=(_label,))
            else:
                db = mx.io.DataBatch(data=(_data,_data_extra), label=(_label,))
            model.forward(db, is_train=False)
            net_out = model.get_outputs()
            _embeddings = net_out[0].asnumpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed+=diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros( (data.shape[0], _embeddings.shape[1]) )
            embeddings[ba:bb,:] = _embeddings[(batch_size-count):,:]
            ba = bb
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm=np.linalg.norm(_em)
            _xnorm+=_norm
            _xnorm_cnt+=1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0].copy()
    embeddings = sklearn.preprocessing.normalize(embeddings)
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)
    print('infer time', time_consumed)
    accuracy = evaluate(embeddings, issame_list, nrof_folds=nfolds)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc2, std2, _xnorm, embeddings_list

# ### Specify paths and parameters

# In[5]:


# Path to dataset
data_dir = '/home/ubuntu/faces_ms1m_112x112/'
# Path to model file
model = '/home/ubuntu/resnet100.onnx'
# Verification targets
target = 'lfw,cfp_ff,cfp_fp,agedb_30'
# Batch size
batch_size = 64
# Number of folds for cross validation
nfolds = 10

# ### Compute validation accuracies
# This is the main function and it loads the ONNX model, loads the validation datasets and computes the validation accuracies on the data according to the verification targets specified above. The accuracy for each target is displayed in the output of the cell.

# In[6]:


if __name__ == '__main__':
    
    # Load image size
    prop = load_property(data_dir)
    image_size = prop.image_size
    print('image_size', image_size)
    
    # Determine and set context
    if len(mx.test_utils.list_gpus())==0:
        ctx = mx.cpu()
        batch_size=1
    else:
        ctx = mx.gpu(0)
        
    time0 = datetime.datetime.now()
    
    # Import ONNX model
    sym, arg_params, aux_params = import_model(model)
    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    # Define model
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    # Bind parameters to the model
    model.bind(data_shapes=[('data', (batch_size, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    time_now = datetime.datetime.now()
    diff = time_now - time0
    print('model loading time', diff.total_seconds())

    ver_list = []
    ver_name_list = []
    
    # Iterate over verification targets
    for name in target.split(','):
        path = os.path.join(data_dir,name+".bin")
        # Load data
        if os.path.exists(path):
            print('loading.. ', name)
            data_set = load_bin(path, image_size)
            ver_list.append(data_set)
            ver_name_list.append(name)
    
    # Iterate over verification targets
    for i in range(len(ver_list)):
        # Compute and print validation accuracies
        acc2, std2, xnorm, embeddings_list = test(ver_list[i], model, batch_size, nfolds)
        print('[%s]XNorm: %f' % (ver_name_list[i], xnorm))
        print('[%s]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], acc2, std2))

# In[ ]:



