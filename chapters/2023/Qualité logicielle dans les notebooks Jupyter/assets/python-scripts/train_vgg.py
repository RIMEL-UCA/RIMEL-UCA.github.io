#!/usr/bin/env python
# coding: utf-8

# # Training Notebook for VGG models on ImageNet Dataset
# 
# ## Overview
# Use this notebook to train a VGG model from scratch. **Make sure to have the ImageNet dataset prepared** according to the guidelines in the dataset section in [VGG readme](README.md) before proceeding.

# ## Prerequisites
# The following dependencies need to be installed before proceeding.
# * mxnet - `pip install mxnet-cu90mkl` (tested on this version GPU, can use other versions)
# * gluoncv - `pip install gluoncv`
# * numpy - `pip install numpy`
# * matplotlib - `pip install matplotlib`
# 
# In order to train the model with a python script: 
# * Generate the script : In Jupyter Notebook browser, go to File -> Download as -> Python (.py)
# * Run the script: `python train_vgg.py`

# ### Import dependencies
# Verify that all dependencies are installed using the cell below. Continue if no errors encountered, warnings can be ignored.

# In[ ]:


import matplotlib
matplotlib.use('Agg')

import time, logging

import mxnet as mx
import numpy as np
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.data import imagenet
from gluoncv.utils import makedirs, TrainingHistory

import os
from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.initializer import Xavier
from mxnet.gluon.contrib.nn import HybridConcurrent
from __future__ import division
import multiprocessing

# ### Specify model, hyperparameters and save locations
# 
# The training was done on a p3.16xlarge ec2 instance on AWS. It has 8 Nvidia Tesla V100 GPUs (16GB each) and Intel Xeon E5-2686 v4 @ 2.70GHz with 64 threads.
# 
# The `batch_size` set below is per device. For multiple GPUs there are different batches in each GPU of size `batch_size` simultaneously.
# 
# The rest of the parameters can be tuned to fit the needs of a user. The values shown below were used to train the model in the model zoo.

# In[2]:


# specify model - choose from vgg16, vgg16_bn, vgg19, vgg19_bn (_bn versions are with batch normalization)
model_name = 'vgg16'

# training and validation pictures to use
data_dir = '/home/ubuntu/imagenet/img_dataset'

# training batch size per device (CPU/GPU)
batch_size = 32

# number of GPUs to use (automatically detect the number of GPUs)
num_gpus = len(mx.test_utils.list_gpus())

# number of pre-processing workers (automatically detect the number of workers)
num_workers = multiprocessing.cpu_count()

# number of training epochs
num_epochs = 100

# learning rate
lr = 0.01

# momentum value for optimizer
momentum = 0.9

# weight decay rate
wd = 0.0001

# decay rate of learning rate
lr_decay = 0.1

# interval for periodic learning rate decays
lr_decay_period = 0

# epoches at which learning rate decays
lr_decay_epoch = '50,80'

# mode in which to train the model. options are symbolic, imperative, hybrid
mode = 'hybrid'

# Number of batches to wait before logging
log_interval = 1000

# frequency of model saving
save_frequency = 10

# directory of saved models
save_dir = 'params'

#directory of training logs
logging_dir = 'logs'

# the path to save the history plot
save_plot_dir = '.'

# ### Model definition in Gluon
# The class `VGG` contains model definitions of all VGG models and the required model is retrieved using the relevant constructor function: `vgg16` / `vgg16_bn` / `vgg19` / `vgg19_bn`. 

# In[3]:


"""VGG, implemented in Gluon."""
__all__ = ['VGG', 'vgg16', 'vgg19', 'vgg16_bn', 'vgg19_bn', 'get_vgg']


class VGG(HybridBlock):
    r"""VGG model from the `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.
    Parameters
    ----------
    layers : list of int
        Numbers of layers in each feature block.
    filters : list of int
        Numbers of filters in each feature block. List length should match the layers.
    classes : int, default 1000
        Number of classification classes.
    batch_norm : bool, default False
        Use batch normalization.
    """
    def __init__(self, layers, filters, classes=1000, batch_norm=False, **kwargs):
        super(VGG, self).__init__(**kwargs)
        assert len(layers) == len(filters)
        with self.name_scope():
            self.features = self._make_features(layers, filters, batch_norm)
            self.features.add(nn.Dense(4096, activation='relu',
                                       weight_initializer='normal',
                                       bias_initializer='zeros'))
            self.features.add(nn.Dropout(rate=0.5))
            self.features.add(nn.Dense(4096, activation='relu',
                                       weight_initializer='normal',
                                       bias_initializer='zeros'))
            self.features.add(nn.Dropout(rate=0.5))
            self.output = nn.Dense(classes,
                                   weight_initializer='normal',
                                   bias_initializer='zeros')

    def _make_features(self, layers, filters, batch_norm):
        featurizer = nn.HybridSequential(prefix='')
        for i, num in enumerate(layers):
            for _ in range(num):
                featurizer.add(nn.Conv2D(filters[i], kernel_size=3, padding=1,
                                         weight_initializer=Xavier(rnd_type='gaussian',
                                                                   factor_type='out',
                                                                   magnitude=2),
                                         bias_initializer='zeros'))
                if batch_norm:
                    featurizer.add(nn.BatchNorm())
                featurizer.add(nn.Activation('relu'))
            featurizer.add(nn.MaxPool2D(strides=2))
        return featurizer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


# Specification
vgg_spec = {16: ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]),
            19: ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512])}


# Constructors
def get_vgg(num_layers, root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""VGG model from the `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.
    Parameters
    ----------
    num_layers : int
        Number of layers for the variant of densenet. Options are 16, 19.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    layers, filters = vgg_spec[num_layers]
    net = VGG(layers, filters, **kwargs)
    return net

def vgg16(**kwargs):
    r"""VGG-16 model from the `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.
    Parameters
    ----------
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_vgg(16, **kwargs)

def vgg19(**kwargs):
    r"""VGG-19 model from the `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.
    Parameters
    ----------
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_vgg(19, **kwargs)

def vgg16_bn(**kwargs):
    r"""VGG-16 model with batch normalization from the
    `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.
    Parameters
    ----------
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    kwargs['batch_norm'] = True
    return get_vgg(16, **kwargs)

def vgg19_bn(**kwargs):
    r"""VGG-19 model with batch normalization from the
    `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.
    Parameters
    ----------
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    kwargs['batch_norm'] = True
    return get_vgg(19, **kwargs)

models = {    'vgg16': vgg16,
              'vgg19': vgg19,
              'vgg16_bn': vgg16_bn,
              'vgg19_bn': vgg19_bn
         }

# ### Helper code
# Define context, optimizer, accuracy metrics, retireve gluon model

# In[4]:


# Specify logging function
logging.basicConfig(level=logging.INFO)

# Specify classes (1000 for ImageNet)
classes = 1000
# Extrapolate batches to all devices
batch_size *= max(1, num_gpus)
# Define context
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]

lr_decay_epoch = [int(i) for i in lr_decay_epoch.split(',')] + [np.inf]

kwargs = {'classes': classes, 'batch_norm': False}

# Define optimizer (nag = Nestrov Accelerated Gradient)
optimizer = 'nag'
optimizer_params = {'learning_rate': lr, 'wd': wd, 'momentum': momentum}

# Retrieve gluon model
net = models[model_name](**kwargs)

# Define accuracy measures - top1 error and top5 error
acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)
train_history = TrainingHistory(['training-top1-err', 'training-top5-err',
                                 'validation-top1-err', 'validation-top5-err'])

makedirs(save_dir)

# ### Define preprocessing functions
# `preprocess_train_data(normalize, jitter_param, lighting_param)` : Do pre-processing and data augmentation of train images -> take random crops of size 224x224, do random left right flips, jitter image color and lighting, mormalize image
# 
# `preprocess_test_data(normalize)` : Pre-process validation images -> resize to size 256x256, take center crop of size 224x224, normalize image

# In[5]:


normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
jitter_param = 0.4
lighting_param = 0.1

# Input pre-processing for train data
def preprocess_train_data(normalize, jitter_param, lighting_param):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                     saturation=jitter_param),
        transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),
        normalize
    ])
    return transform_train

# Input pre-processing for validation data
def preprocess_test_data(normalize):
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    return transform_test

# ### Define test function
# `test(ctx, val_data)` : Computes and returns validation errors on `val_data` using `ctx` context

# In[ ]:


# Test function
def test(ctx, val_data):
    # Reset accuracy metrics
    acc_top1.reset()
    acc_top5.reset()
    for i, batch in enumerate(val_data):
        # Load validation batch
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        # Perform forward pass
        outputs = [net(X) for X in data]
        # Update accuracy metrics
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)
    # Retrieve and return top1 and top5 errors
    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    return (1-top1, 1-top5)

# ### Define train function
# `train(epochs, ctx)` : Train model for `epochs` epochs using `ctx` context, log training progress, compute and display validation errors after each epoch, take periodic snapshots of the model, generates training plot 

# In[6]:


# Train function
def train(epochs, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    # Initialize network - Use method in MSRA paper <https://arxiv.org/abs/1502.01852>
    net.initialize(mx.init.MSRAPrelu(), ctx=ctx)
    # Prepare train and validation batches
    transform_train = preprocess_train_data(normalize, jitter_param, lighting_param)
    transform_test = preprocess_test_data(normalize)
    train_data = gluon.data.DataLoader(
        imagenet.classification.ImageNet(data_dir, train=True).transform_first(transform_train),
        batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)
    val_data = gluon.data.DataLoader(
        imagenet.classification.ImageNet(data_dir, train=False).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # Define trainer
    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
    # Define loss
    L = gluon.loss.SoftmaxCrossEntropyLoss()

    lr_decay_count = 0

    best_val_score = 1
    # Main training loop - loop over epochs
    for epoch in range(epochs):
        tic = time.time()
        # Reset accuracy metrics
        acc_top1.reset()
        acc_top5.reset()
        btic = time.time()
        train_loss = 0
        num_batch = len(train_data)
        
        # Check and perform learning rate decay
        if lr_decay_period and epoch and epoch % lr_decay_period == 0:
            trainer.set_learning_rate(trainer.learning_rate*lr_decay)
        elif lr_decay_period == 0 and epoch == lr_decay_epoch[lr_decay_count]:
            trainer.set_learning_rate(trainer.learning_rate*lr_decay)
            lr_decay_count += 1
        # Loop over batches in an epoch
        for i, batch in enumerate(train_data):
            # Load train batch
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            label_smooth = label
            # Perform forward pass
            with ag.record():
                outputs = [net(X) for X in data]
                loss = [L(yhat, y) for yhat, y in zip(outputs, label_smooth)]
            # Perform backward pass
            ag.backward(loss)
            # PErform updates
            trainer.step(batch_size)
            # Update accuracy metrics
            acc_top1.update(label, outputs)
            acc_top5.update(label, outputs)
            # Update loss
            train_loss += sum([l.sum().asscalar() for l in loss])
            # Log training progress (after each `log_interval` batches)
            if log_interval and not (i+1)%log_interval:
                _, top1 = acc_top1.get()
                _, top5 = acc_top5.get()
                err_top1, err_top5 = (1-top1, 1-top5)
                logging.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\ttop1-err=%f\ttop5-err=%f'%(
                             epoch, i, batch_size*log_interval/(time.time()-btic), err_top1, err_top5))
                btic = time.time()

        # Retrieve training errors and loss
        _, top1 = acc_top1.get()
        _, top5 = acc_top5.get()
        err_top1, err_top5 = (1-top1, 1-top5)
        train_loss /= num_batch * batch_size

        # Compute validation errors
        err_top1_val, err_top5_val = test(ctx, val_data)
        # Update training history
        train_history.update([err_top1, err_top5, err_top1_val, err_top5_val])
        # Update plot
        train_history.plot(['training-top1-err', 'validation-top1-err','training-top5-err', 'validation-top5-err'],
                           save_path='%s/%s_top_error.png'%(save_plot_dir, model_name))

        # Log training progress (after each epoch)
        logging.info('[Epoch %d] training: err-top1=%f err-top5=%f loss=%f'%(epoch, err_top1, err_top5, train_loss))
        logging.info('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
        logging.info('[Epoch %d] validation: err-top1=%f err-top5=%f'%(epoch, err_top1_val, err_top5_val))

        # Save a snapshot of the best model - use net.export to get MXNet symbols and params
        if err_top1_val < best_val_score and epoch > 50:
            best_val_score = err_top1_val
            net.export('%s/%.4f-imagenet-%s-best'%(save_dir, best_val_score, model_name), epoch)
        # Save a snapshot of the model after each 'save_frequency' epochs
        if save_frequency and save_dir and (epoch + 1) % save_frequency == 0:
            net.export('%s/%.4f-imagenet-%s'%(save_dir, best_val_score, model_name), epoch)
    # Save a snapshot of the model at the end of training
    if save_frequency and save_dir:
        net.export('%s/%.4f-imagenet-%s'%(save_dir, best_val_score, model_name), epochs-1)

# ### Train model
# * Run the cell below to start training
# * Logs are displayed in the cell output
# * An example run of 1 epoch is shown here
# * Once training completes, the symbols and params files are saved in the root folder

# In[7]:


def main():
    net.hybridize()
    train(num_epochs, context)
if __name__ == '__main__':
    main()

# ### Export model to ONNX format
# The conversion of the model to ONNX format is done using an internal converter which will be released soon. The notebook will be updated with the code for the export once the converter is released.
