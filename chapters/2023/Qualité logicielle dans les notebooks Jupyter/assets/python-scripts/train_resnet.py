#!/usr/bin/env python
# coding: utf-8

# # Training Notebook for ResNet v1 and v2 models on ImageNet Dataset
# 
# ## Overview
# Use this notebook to train a ResNet model from scratch. **Make sure to have the ImageNet dataset prepared** according to the guidelines in the dataset section in [ResNet readme](README.md) before proceeding.

# ## Prerequisites
# The following dependencies need to be installed before proceeding.
# * mxnet - `pip install mxnet-cu90mkl` (tested on this version GPU, can use other versions)
# * gluoncv - `pip install gluoncv`
# * numpy - `pip install numpy`
# * matplotlib - `pip install matplotlib`
# 
# In order to train the model with a python script: 
# * Generate the script : In Jupyter Notebook browser, go to File -> Download as -> Python (.py)
# * Run the script: `python train_resnet.py`

# ### Import dependencies
# Verify that all dependencies are installed using the cell below. Continue if no errors encountered

# In[ ]:


import matplotlib
matplotlib.use('Agg')

import argparse, time, logging

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
from mxnet.gluon.contrib.nn import HybridConcurrent
import multiprocessing

# ### Specify model, hyperparameters and save locations
# 
# The training was done on a p3.16xlarge ec2 instance on AWS. It has 8 Nvidia Tesla V100 GPUs (16GB each) and Intel Xeon E5-2686 v4 @ 2.70GHz with 64 threads.
# 
# The `batch_size` set below is per device. For multiple GPUs there are different batches in each GPU of size `batch_size` simultaneously.
# 
# The rest of the parameters can be tuned to fit the needs of a user. The values shown below were used to train the model in the model zoo.

# In[2]:


# specify model - choose from (resnet18_v1,resnet18_v2,resnet34_v1,resnet34_v2,resnet50_v1,resnet50_v2,
#resnet101_v1,resnet101_v2,resnet152_v1,resnet152_v2)
model_name = 'resnet18_v1' 

# path to training and validation images to use
data_dir = '/home/ubuntu/imagenet/img_dataset'

# training batch size per device (CPU/GPU)
# Used batch size = 64 for resnet18_v1,resnet18_v2,resnet34_v1,resnet34_v2,resnet50_v1,resnet50_v2,resnet101_v1,
#resnet101_v2
#Used batch size=32 for resnet152_v1,resnet152_v2
batch_size = 64

# number of GPUs to use (automatically detect the number of GPUs)
num_gpus = len(mx.test_utils.list_gpus())

# number of pre-processing workers (automatically detect the number of workers)
num_workers = multiprocessing.cpu_count()

# number of training epochs 
#used as 150 for all of the models , used 1 over here to show demo for 1 epoch
num_epochs = 1

# learning rate
lr = 0.01

# momentum value for optimizer
momentum = 0.9

# weight decay rate
wd = 0.0002

# decay rate of learning rate
lr_decay = 0.1

# interval for periodic learning rate decays
lr_decay_period = 0

# epoches at which learning rate decays
lr_decay_epoch = '30,60,90'

# mode in which to train the model. options are symbolic, imperative, hybrid
mode = 'hybrid'

# Number of batches to wait before logging
log_interval = 50

# frequency of model saving
save_frequency = 10

# directory of saved models
save_dir = 'params'

#directory of training logs
logging_dir = 'logs'

# the path to save the history plot
save_plot_dir = '.'


# ### Model definition in Gluon

# In[3]:


##This block contains definition for Resnet v1 and v2

#Helpers
def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)


# Blocks
class BasicBlockV1(HybridBlock):
    r"""BasicBlock V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 18, 34 layers.
    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BasicBlockV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(_conv3x3(channels, stride, in_channels))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels, 1, channels))
        self.body.add(nn.BatchNorm())
        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(residual+x, act_type='relu')

        return x


class BottleneckV1(HybridBlock):
    r"""Bottleneck V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 50, 101, 152 layers.
    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BottleneckV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(channels//4, kernel_size=1, strides=stride))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels//4, 1, channels//4))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels, kernel_size=1, strides=1))
        self.body.add(nn.BatchNorm())
        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(x + residual, act_type='relu')
        return x


class BasicBlockV2(HybridBlock):
    r"""BasicBlock V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 18, 34 layers.
    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BasicBlockV2, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm()
        self.conv1 = _conv3x3(channels, stride, in_channels)
        self.bn2 = nn.BatchNorm()
        self.conv2 = _conv3x3(channels, 1, channels)
        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                        in_channels=in_channels)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)

        return x + residual


class BottleneckV2(HybridBlock):
    r"""Bottleneck V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 50, 101, 152 layers.
    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BottleneckV2, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm()
        self.conv1 = nn.Conv2D(channels//4, kernel_size=1, strides=1, use_bias=False)
        self.bn2 = nn.BatchNorm()
        self.conv2 = _conv3x3(channels//4, stride, channels//4)
        self.bn3 = nn.BatchNorm()
        self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False)
        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                        in_channels=in_channels)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)

        x = self.bn3(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv3(x)

        return x + residual


# Nets
class ResNetV1(HybridBlock):
    r"""ResNet V1 model from
    `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default False
        Enable thumbnail.
    """
    def __init__(self, block, layers, channels, classes=1000, thumbnail=False, **kwargs):
        super(ResNetV1, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            if thumbnail:
                self.features.add(_conv3x3(channels[0], 1, 0))
            else:
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False))
                self.features.add(nn.BatchNorm())
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(3, 2, 1))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=channels[i]))
            self.features.add(nn.GlobalAvgPool2D())

            self.output = nn.Dense(classes, in_units=channels[-1])

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            prefix=''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)

        return x


class ResNetV2(HybridBlock):
    r"""ResNet V2 model from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default False
        Enable thumbnail.
    """
    def __init__(self, block, layers, channels, classes=1000, thumbnail=False, **kwargs):
        super(ResNetV2, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.BatchNorm(scale=False, center=False))
            if thumbnail:
                self.features.add(_conv3x3(channels[0], 1, 0))
            else:
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False))
                self.features.add(nn.BatchNorm())
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(3, 2, 1))

            in_channels = channels[0]
            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=in_channels))
                in_channels = channels[i+1]
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.GlobalAvgPool2D())
            self.features.add(nn.Flatten())

            self.output = nn.Dense(classes, in_units=in_channels)

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            prefix=''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


# Specification
resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50: ('bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: ('bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}

resnet_net_versions = [ResNetV1, ResNetV2]
resnet_block_versions = [{'basic_block': BasicBlockV1, 'bottle_neck': BottleneckV1},
                         {'basic_block': BasicBlockV2, 'bottle_neck': BottleneckV2}]


# Constructor
def get_resnet(version, num_layers, **kwargs):
    r"""ResNet V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    ResNet V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    Parameters
    ----------
    version : int
        Version of ResNet. Options are 1, 2.
    num_layers : int
        Numbers of layers. Options are 18, 34, 50, 101, 152.
    """
    assert num_layers in resnet_spec, \
        "Invalid number of layers: %d. Options are %s"%(
            num_layers, str(resnet_spec.keys()))
    block_type, layers, channels = resnet_spec[num_layers]
    assert version >= 1 and version <= 2, \
        "Invalid resnet version: %d. Options are 1 and 2."%version
    resnet_class = resnet_net_versions[version-1]
    block_class = resnet_block_versions[version-1][block_type]
    net = resnet_class(block_class, layers, channels, **kwargs)
    return net

def resnet18_v1(**kwargs):
    r"""ResNet-18 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    """
    return get_resnet(1, 18, **kwargs)

def resnet34_v1(**kwargs):
    r"""ResNet-34 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    """
    return get_resnet(1, 34, **kwargs)

def resnet50_v1(**kwargs):
    r"""ResNet-50 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    """
    return get_resnet(1, 50, **kwargs)

def resnet101_v1(**kwargs):
    r"""ResNet-101 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    """
    return get_resnet(1, 101, **kwargs)

def resnet152_v1(**kwargs):
    r"""ResNet-152 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    """
    return get_resnet(1, 152, **kwargs)

def resnet18_v2(**kwargs):
    r"""ResNet-18 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    """
    return get_resnet(2, 18, **kwargs)

def resnet34_v2(**kwargs):
    r"""ResNet-34 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    """
    return get_resnet(2, 34, **kwargs)

def resnet50_v2(**kwargs):
    r"""ResNet-50 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    """
    return get_resnet(2, 50, **kwargs)

def resnet101_v2(**kwargs):
    r"""ResNet-101 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    """
    return get_resnet(2, 101, **kwargs)

def resnet152_v2(**kwargs):
    r"""ResNet-152 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    """
    return get_resnet(2, 152, **kwargs)

models = {    'resnet18_v1': resnet18_v1,
              'resnet34_v1': resnet34_v1,
              'resnet50_v1': resnet50_v1,
              'resnet101_v1': resnet101_v1,
              'resnet152_v1': resnet152_v1,
              'resnet18_v2': resnet18_v2,
              'resnet34_v2': resnet34_v2,
              'resnet50_v2': resnet50_v2,
              'resnet101_v2': resnet101_v2,
              'resnet152_v2': resnet152_v2
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

kwargs = { 'classes': classes}

# Define optimizer (nag = Nestrov Accelerated Gradient)
optimizer = 'nag'
optimizer_params = {'learning_rate': lr, 'wd': wd, 'momentum': momentum}
kwargs['thumbnail'] = False

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
        transforms.Resize(480),
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


def train(epochs, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    # Initialize network
    net.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)
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
            for l in loss:
                l.backward()
            # Perform updates
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
    #net.export(model_name)
if __name__ == '__main__':
    main()

# ### Export model to ONNX format
# The conversion of the model to ONNX format is done using an internal converter which will be released soon. The notebook will be updated with the code for the export once the converter is released.
