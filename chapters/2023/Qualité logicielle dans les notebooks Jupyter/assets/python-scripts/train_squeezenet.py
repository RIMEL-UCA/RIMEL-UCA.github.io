#!/usr/bin/env python
# coding: utf-8

# # Training Notebook for SqueezeNet 1.0 and 1.1 on ImageNet Dataset
# 
# ## Overview
# Use this notebook to train a SqueezeNet model from scratch. **Make sure to have the ImageNet dataset prepared** according to the guidelines in the dataset section in [SqueezeNet readme](README.md) before proceeding.

# ## Prerequisites
# The following dependencies need to be installed before proceeding.
# * mxnet - `pip install mxnet-cu90mkl` (tested on this version GPU, can use other versions)
# * gluoncv - `pip install gluoncv`
# * numpy - `pip install numpy`
# * matplotlib - `pip install matplotlib`
# 
# In order to train the model with a python script: 
# * Generate the script : In Jupyter Notebook browser, go to File -> Download as -> Python (.py)
# * Run the script: `python train_squeezenet.py`

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
from mxnet.gluon.contrib.nn import HybridConcurrent
import multiprocessing

# ### Specify model, hyperparameters and save locations
# 
# The training was done on a p3.8xlarge ec2 instance on AWS. It has 4 Nvidia Tesla V100 GPUs (16GB each) and Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz with 32 threads.
# 
# The `batch_size` set below is per device. For multiple GPUs there are different batches in each GPU of size `batch_size` simultaneously.
# 
# The rest of the parameters can be tuned to fit the needs of a user. The values shown below were used to train the model in the model zoo.

# In[10]:


# specify model - squeezenet1.0 or squeezenet1.1
model_name = 'squeezenet1.0'

# training and validation pictures to use
data_dir = '/home/ubuntu/imagenet/img_dataset'

# training batch size per device (CPU/GPU)
batch_size = 128

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
wd = 0.0002

# decay rate of learning rate
lr_decay = 0.1

# interval for periodic learning rate decays
lr_decay_period = 0

# epoches at which learning rate decays
lr_decay_epoch = '60,90'

# mode in which to train the model. options are symbolic, imperative, hybrid
mode = 'hybrid'

# Number of batches to wait before logging
log_interval = 100

# frequency of model saving
save_frequency = 10

# directory of saved models
save_dir = 'params'

#directory of training logs
logging_dir = 'logs'

# the path to save the history plot
save_plot_dir = '.'

# ### Model definition in Gluon
# 
# The class `SqueezeNet` contains model definitions of both SqueezeNet models and the required model is retrieved using the relevant constructor function: `Squeezenet1_0` or `Squeezenet1_1`. 
# 
# `_make_fire` and `_make_fire_conv` are helper functions which define the fire modules unique to SqueezeNet .

# In[11]:


"""SqueezeNet, implemented in Gluon."""
__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']

# Helpers
def _make_fire(squeeze_channels, expand1x1_channels, expand3x3_channels):
    out = nn.HybridSequential(prefix='')
    out.add(_make_fire_conv(squeeze_channels, 1))

    paths = HybridConcurrent(axis=1, prefix='')
    paths.add(_make_fire_conv(expand1x1_channels, 1))
    paths.add(_make_fire_conv(expand3x3_channels, 3, 1))
    out.add(paths)

    return out

def _make_fire_conv(channels, kernel_size, padding=0):
    out = nn.HybridSequential(prefix='')
    out.add(nn.Conv2D(channels, kernel_size, padding=padding))
    out.add(nn.Activation('relu'))
    return out

# Net
class SqueezeNet(HybridBlock):
    r"""SqueezeNet model from the `"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
    and <0.5MB model size" <https://arxiv.org/abs/1602.07360>`_ paper.
    SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    Parameters
    ----------
    version : str
        Version of squeezenet. Options are '1.0', '1.1'.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self, version, classes=1000, **kwargs):
        super(SqueezeNet, self).__init__(**kwargs)
        assert version in ['1.0', '1.1'], ("Unsupported SqueezeNet version {version}:"
                                           "1.0 or 1.1 expected".format(version=version))
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            if version == '1.0':
                self.features.add(nn.Conv2D(96, kernel_size=7, strides=2))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2, ceil_mode=True))
                self.features.add(_make_fire(16, 64, 64))
                self.features.add(_make_fire(16, 64, 64))
                self.features.add(_make_fire(32, 128, 128))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2, ceil_mode=True))
                self.features.add(_make_fire(32, 128, 128))
                self.features.add(_make_fire(48, 192, 192))
                self.features.add(_make_fire(48, 192, 192))
                self.features.add(_make_fire(64, 256, 256))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2, ceil_mode=True))
                self.features.add(_make_fire(64, 256, 256))
            else:
                self.features.add(nn.Conv2D(64, kernel_size=3, strides=2))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2, ceil_mode=True))
                self.features.add(_make_fire(16, 64, 64))
                self.features.add(_make_fire(16, 64, 64))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2, ceil_mode=True))
                self.features.add(_make_fire(32, 128, 128))
                self.features.add(_make_fire(32, 128, 128))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2, ceil_mode=True))
                self.features.add(_make_fire(48, 192, 192))
                self.features.add(_make_fire(48, 192, 192))
                self.features.add(_make_fire(64, 256, 256))
                self.features.add(_make_fire(64, 256, 256))
            self.features.add(nn.Dropout(0.5))

            self.output = nn.HybridSequential(prefix='')
            self.output.add(nn.Conv2D(classes, kernel_size=1))
            self.output.add(nn.Activation('relu'))
            self.output.add(nn.AvgPool2D(13))
            self.output.add(nn.Flatten())

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x

# Constructor
def get_squeezenet(version, root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""SqueezeNet model from the `"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
    and <0.5MB model size" <https://arxiv.org/abs/1602.07360>`_ paper.
    SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    Parameters
    ----------
    version : str
        Version of squeezenet. Options are '1.0', '1.1'.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    net = SqueezeNet(version, **kwargs)
    return net

def squeezenet1_0(**kwargs):
    r"""SqueezeNet 1.0 model from the `"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
    and <0.5MB model size" <https://arxiv.org/abs/1602.07360>`_ paper.
    Parameters
    ----------
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_squeezenet('1.0', **kwargs)

def squeezenet1_1(**kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    Parameters
    ----------
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_squeezenet('1.1', **kwargs)

# ### Helper code
# Define context, optimizer, accuracy metrics, retireve gluon model

# In[12]:


# Specify logging fucntion
logging.basicConfig(level=logging.INFO)

# Specify classes (1000 for ImageNet)
classes = 1000
# Extrapolate batches to all devices
batch_size *= max(1, num_gpus)
# Define context
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]

lr_decay_epoch = [int(i) for i in lr_decay_epoch.split(',')] + [np.inf]

kwargs = {'classes': classes}

# Define optimizer (nag = Nestrov Accelerated Gradient)
optimizer = 'nag'
optimizer_params = {'learning_rate': lr, 'wd': wd, 'momentum': momentum}

# Retrieve gluon model
if model_name == 'squeezenet1.0':
    net = squeezenet1_0(**kwargs)
else:
    net = squeezenet1_1(**kwargs)

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

# In[13]:


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

# In[14]:


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

# In[15]:


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
            # Perform updates
            trainer.step(batch_size)
            # Initialize last conv layer weights with normal distribution after first forward pass
            if i==0 and epoch==0:
                new_classifier_w = mx.nd.random_normal(shape=(1000, 512, 1, 1), scale=0.01)
                final_conv_layer_params = net.output[0].params
                final_conv_layer_params.get('weight').set_data(new_classifier_w)
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

# In[16]:


def main():
    net.hybridize()
    train(num_epochs, context)
if __name__ == '__main__':
    main()

# ### Export model to ONNX format
# The conversion of the model to ONNX format is done using an internal converter which will be released soon. The notebook will be updated with the code for the export once the converter is released.
