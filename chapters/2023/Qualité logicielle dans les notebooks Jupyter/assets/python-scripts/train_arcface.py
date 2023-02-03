#!/usr/bin/env python
# coding: utf-8

# # Training Notebook for ArcFace model on Refined MS1M dataset
# 
# ## Overview
# Use this notebook to train a ArcFace model from scratch. Make sure to have the Refined MS1M dataset prepared before proceeding.
# 
# ## Prerequisites
# The training notebooks and scripts are tested on python 2.7. The following additional packages need to be installed before proceeding:
# * MXNet - `pip install mxnet-cu90mkl` (tested on this version, can use other versions)
# * OpenCV - `pip install opencv-python`
# * Scikit-learn - `pip install scikit-learn`
# * Scikit-image - `pip install scikit-image`
# * EasyDict - `pip install easydict`
# * numpy - `pip install numpy`
# 
# Also the following scripts (included in the repo) must be present in the same folder as this notebook:
# * `face_image.py` (prepares face images in the dataset for training)
# * `face_preprocess.py` (performs preprocessing on face images)
# * `fresnet.py` (contains model definition of ResNet100)
# * `image_iter.py` (helper script)
# * `symbol_utils.py` (helper script)
# * `verification.py` (performs verification on validation sets)
# 
# In order to train the model with a python script:
# * Generate the script : In Jupyter Notebook browser, go to File -> Download as -> Python (.py)
# * Run the script: `python train_arcface.py`

# ### Import dependencies
# Verify that all dependencies are installed using the cell below. Continue if no errors encountered, warnings can be ignored.

# In[18]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import numpy as np
import random
import logging
import pickle
import numpy as np
from image_iter import FaceImageIter
from image_iter import FaceImageIterList
import mxnet as mx
from mxnet import ndarray as nd
import mxnet.optimizer as optimizer
import fresnet
import verification
import sklearn
from easydict import EasyDict as edict
import multiprocessing

# ### Specify model, hyperparameters and paths
# The training was done on a p3.8xlarge ec2 instance on AWS. It has 4 Nvidia Tesla V100 GPUs (16GB each) and Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz with 32 threads.
# 
# The batch_size set below is per device. For multiple GPUs there are different batches in each GPU of size batch_size simultaneously.
# 
# The rest of the parameters can be tuned to fit the needs of a user. The values shown below were used to train the model in the model zoo.

# In[2]:


# Path to dataset
data_dir = '/home/ubuntu/faces_ms1m_112x112'
# Path to directory where models will be saved
prefix = '/home/ubuntu/resnet100'
# Load pretrained model
pretrained = ''
# Checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save
ckpt = 1
# do verification testing and model saving every verbose batches
verbose = 2000
# max training batches
max_steps = 0
# number of training epochs
end_epoch = 30
# initial learning rate
lr = 0.1
# learning rate decay iterations
lr_steps = [100000, 140000, 160000]
# weight decay
wd = 0.0005
# weight decay multiplier for fc7
fc7_wd_mult = 1.0
# momentum
mom = 0.9
# embedding length
emb_size = 512
# batch size in each context
per_batch_size = 64
# margin for loss
margin_m = 0.5
# scale for feature
margin_s = 64.0
# verification targets
target = 'lfw,cfp_fp,agedb_30'
beta = 1000.0
beta_min = 5.0
beta_freeze = 0
gamma = 0.12
power = 1.0
scale = 0.9993

# ### Helper code
# class `AccMetric` : used to define and update accuracy metrics
# 
# class `LossValueMetric` : used to define and update loss metrics
# 
# `load_property()` : Function for loading num_classes and image_size from datasets folder

# In[3]:


# Helper class for accuracy metrics
class AccMetric(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(AccMetric, self).__init__(
            'acc', axis=self.axis,
            output_names=None, label_names=None)
        self.losses = []
        self.count = 0

    def update(self, labels, preds):
        self.count+=1
        preds = [preds[1]] #use softmax output
        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.asnumpy().astype('int32').flatten()
            label = label.asnumpy()
            if label.ndim==2:
                label = label[:,0]
            label = label.astype('int32').flatten()
            assert label.shape==pred_label.shape
            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat)

# Helper class for loss metrics
class LossValueMetric(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(LossValueMetric, self).__init__(
            'lossvalue', axis=self.axis,
            output_names=None, label_names=None)
        self.losses = []

    def update(self, labels, preds):
        loss = preds[-1].asnumpy()[0]
        self.sum_metric += loss
        self.num_inst += 1.0
        gt_label = preds[-2].asnumpy()

# Helper function for loading num_classes and input image sizes
def load_property(data_dir):
    prop = edict()
    for line in open(os.path.join(data_dir, 'property')):
        vec = line.strip().split(',')
        assert len(vec)==3
        prop.num_classes = int(vec[0])
        prop.image_size = [int(vec[1]), int(vec[2])]
    return prop

# ### Prepare network and define loss
# `get_symbol()` : Loads the model from the model definition file, defines ArcFace loss

# In[4]:


def get_symbol(arg_params, aux_params, image_channel, image_h, image_w, num_layers, num_classes, data_dir,prefix,pretrained,ckpt,verbose,max_steps,end_epoch,lr,lr_steps,wd,fc7_wd_mult,
              mom,emb_size,per_batch_size,margin_m,margin_s,target,beta,beta_min,beta_freeze,gamma,power,scale):
    data_shape = (image_channel,image_h,image_w)
    image_shape = ",".join([str(x) for x in data_shape])
    margin_symbols = []
    print('init resnet', num_layers)
    
    # Load Resnet100 model - model definition is present in fresnet.py
    embedding = fresnet.get_symbol(emb_size, num_layers, 
        version_se=0, version_input=1, 
        version_output='E', version_unit=3,
        version_act='prelu')
    all_label = mx.symbol.Variable('softmax_label')
    gt_label = all_label
    extra_loss = None
    _weight = mx.symbol.Variable("fc7_weight", shape=(num_classes, emb_size), lr_mult=1.0, wd_mult=fc7_wd_mult)
    
    # Define ArcFace loss
    s = margin_s
    m = margin_m
    assert s>0.0
    assert m>=0.0
    assert m<(math.pi/2)
    _weight = mx.symbol.L2Normalization(_weight, mode='instance')
    nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')*s
    fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=num_classes, name='fc7')
    zy = mx.sym.pick(fc7, gt_label, axis=1)
    cos_t = zy/s
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = math.sin(math.pi-m)*m
    threshold = math.cos(math.pi-m)
    cond_v = cos_t - threshold
    cond = mx.symbol.Activation(data=cond_v, act_type='relu')
    body = cos_t*cos_t
    body = 1.0-body
    sin_t = mx.sym.sqrt(body)
    new_zy = cos_t*cos_m
    b = sin_t*sin_m
    new_zy = new_zy - b
    new_zy = new_zy*s
    zy_keep = zy - s*mm
    new_zy = mx.sym.where(cond, new_zy, zy_keep)
    diff = new_zy - zy
    diff = mx.sym.expand_dims(diff, 1)
    gt_one_hot = mx.sym.one_hot(gt_label, depth = num_classes, on_value = 1.0, off_value = 0.0)
    body = mx.sym.broadcast_mul(gt_one_hot, diff)
    fc7 = fc7+body
    out_list = [mx.symbol.BlockGrad(embedding)]
    softmax = mx.symbol.SoftmaxOutput(data=fc7, label = gt_label, name='softmax', normalization='valid')
    out_list.append(softmax)
    out = mx.symbol.Group(out_list)
    return (out, arg_params, aux_params)

# ### Define train function
# `train_net()` : Train model, log training progress, save periodic checkpoints, compute and display validation accuracies periodically

# In[5]:


def train_net(data_dir,prefix,pretrained,ckpt,verbose,max_steps,end_epoch,lr,lr_steps,wd,fc7_wd_mult,
              mom,emb_size,per_batch_size,margin_m,margin_s,target,beta,beta_min,beta_freeze,gamma,power,scale):
    # define context
    ctx = []
    num_gpus = max(mx.test_utils.list_gpus()) + 1
    if num_gpus>0:
        for i in range(num_gpus):
            ctx.append(mx.gpu(i))
    if len(ctx)==0:
        ctx = [mx.cpu()]
        print('use cpu')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    prefix_dir = os.path.dirname(prefix)
    if not os.path.exists(prefix_dir):
        os.makedirs(prefix_dir)
    ctx_num = len(ctx)
    num_layers = 100
    print('num_layers',num_layers)
    batch_size = per_batch_size*ctx_num
    rescale_threshold = 0
    image_channel = 3

    os.environ['BETA'] = str(beta)
    data_dir_list = data_dir.split(',')
    assert len(data_dir_list)==1
    data_dir = data_dir_list[0]
    path_imgrec = None
    path_imglist = None
    prop = load_property(data_dir)
    num_classes = prop.num_classes
    image_size = prop.image_size
    image_h = image_size[0]
    image_w = image_size[1]
    print('image_size', image_size)
    assert(num_classes>0)
    print('num_classes', num_classes)
    path_imgrec = os.path.join(data_dir, "train.rec")

    data_shape = (image_channel,image_size[0],image_size[1])
    mean = None

    begin_epoch = 0
    base_lr = lr
    base_wd = wd
    base_mom = mom
    if len(pretrained)==0:
        arg_params = None
        aux_params = None
        sym, arg_params, aux_params = get_symbol(arg_params, aux_params, image_channel, image_h, image_w, 
                                                 num_layers, num_classes, data_dir,prefix,pretrained,ckpt,
                                                 verbose,max_steps,end_epoch,lr,lr_steps,wd,fc7_wd_mult,
                                                 mom,emb_size,per_batch_size,margin_m,margin_s,target,beta,
                                                 beta_min,beta_freeze,gamma,power,scale)
    else:
        vec = pretrained.split(',')
        print('loading', vec)
        _, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
        sym, arg_params, aux_params = get_symbol(arg_params, aux_params)


    model = mx.mod.Module(
        context       = ctx,
        symbol        = sym,
    )
    val_dataiter = None

    train_dataiter = FaceImageIter(
        batch_size           = batch_size,
        data_shape           = data_shape,
        path_imgrec          = path_imgrec,
        shuffle              = True,
        rand_mirror          = 1,
        mean                 = mean,
        cutoff               = 0,
    )

    _metric = AccMetric()
    eval_metrics = [mx.metric.create(_metric)]

    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    _rescale = 1.0/ctx_num
    opt = optimizer.SGD(learning_rate=base_lr, momentum=base_mom, wd=base_wd, rescale_grad=_rescale)
    som = 20
    _cb = mx.callback.Speedometer(batch_size, som)

    ver_list = []
    ver_name_list = []
    for name in target.split(','):
        path = os.path.join(data_dir,name+".bin")
        if os.path.exists(path):
            data_set = verification.load_bin(path, image_size)
            ver_list.append(data_set)
            ver_name_list.append(name)
            print('ver', name)



    def ver_test(nbatch):
        results = []
        for i in xrange(len(ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(ver_list[i], model, batch_size, 10, None, None)
            print('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
            print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
            results.append(acc2)
        return results



    highest_acc = [0.0, 0.0]  #lfw and target
    global_step = [0]
    save_step = [0]
    
    p = 512.0/batch_size
    for l in xrange(len(lr_steps)):
        lr_steps[l] = int(lr_steps[l]*p)
    print('lr_steps', lr_steps)
    def _batch_callback(param):
        global_step[0]+=1
        mbatch = global_step[0]
        for _lr in lr_steps:
            if mbatch==beta_freeze+_lr:
                opt.lr *= 0.1
                print('lr change to', opt.lr)
                break

        _cb(param)
        if mbatch%1000==0:
            print('lr-batch-epoch:',opt.lr,param.nbatch,param.epoch)

        if mbatch>=0 and mbatch%verbose==0:
            acc_list = ver_test(mbatch)
            save_step[0]+=1
            msave = save_step[0]
            do_save = False
            if len(acc_list)>0:
                lfw_score = acc_list[0]
                if lfw_score>highest_acc[0]:
                    highest_acc[0] = lfw_score
                    if lfw_score>=0.998:
                        do_save = True
                if acc_list[-1]>=highest_acc[-1]:
                    highest_acc[-1] = acc_list[-1]
                    if lfw_score>=0.99:
                        do_save = True
            if ckpt==0:
                do_save = False
            elif ckpt>1:
                do_save = True
            if do_save:
                print('saving', msave)
                arg, aux = model.get_params()
                mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)
            print('[%d]Accuracy-Highest: %1.5f'%(mbatch, highest_acc[-1]))
        if mbatch<=beta_freeze:
            _beta = beta
        else:
            move = max(0, mbatch-beta_freeze)
            _beta = max(beta_min, beta*math.pow(1+gamma*move, -1.0*power))
        os.environ['BETA'] = str(_beta)
        if max_steps>0 and mbatch>max_steps:
            sys.exit(0)

    epoch_cb = None

    model.fit(train_dataiter,
        begin_epoch        = begin_epoch,
        num_epoch          = end_epoch,
        eval_data          = val_dataiter,
        eval_metric        = eval_metrics,
        kvstore            = 'device',
        optimizer          = opt,
        initializer        = initializer,
        arg_params         = arg_params,
        aux_params         = aux_params,
        allow_missing      = True,
        batch_end_callback = _batch_callback,
        epoch_end_callback = epoch_cb )

# ### Train model
# * Run the cell below to start training
# * Logs are displayed in the cell output
# * An example run of 2000 batches is shown here
# * Symbols and params files are saved periodically in the `prefix` folder

# In[6]:


def main():
    train_net(data_dir,prefix,pretrained,ckpt,verbose,max_steps,end_epoch,lr,lr_steps,wd,fc7_wd_mult,
              mom,emb_size,per_batch_size,margin_m,margin_s,target,beta,beta_min,beta_freeze,gamma,power,scale)

if __name__ == '__main__':
    main()

# ### Slim model
# The last layer in the trained model is not required for inference and can be discarded (using cell below) which reduces the model size.

# In[27]:


# Choose model to slim (give path to syms and params)
prefix = '/home/ubuntu/resnet100'
epoch = 1

# Load model
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
# Populate list containing nodes to be removed
all_layers = sym.get_internals()
sym = all_layers['fc1_output']
dellist = []
for k,v in arg_params.iteritems():
    if k.startswith('fc7'):
        dellist.append(k)
        
# Remove nodes
for d in dellist:
    del arg_params[d]

# Save slimed model
mx.model.save_checkpoint(prefix, 0, sym, arg_params, aux_params)

# In[ ]:



