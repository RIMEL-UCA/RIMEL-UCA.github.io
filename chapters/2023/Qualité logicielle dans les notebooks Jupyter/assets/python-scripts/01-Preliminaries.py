#!/usr/bin/env python
# coding: utf-8

# <small><i>This notebook was put together by [Jake Vanderplas](http://www.vanderplas.com). Source and license info is on [GitHub](https://github.com/jakevdp/sklearn_tutorial/).</i></small>

# # An Introduction to scikit-learn: Machine Learning in Python

# ## Goals of this Tutorial

# - **Introduce the basics of Machine Learning**, and some skills useful in practice.
# - **Introduce the syntax of scikit-learn**, so that you can make use of the rich toolset available.

# ## Schedule:

# **Preliminaries: Setup & introduction** (15 min)
# * Making sure your computer is set-up
# 
# **Basic Principles of Machine Learning and the Scikit-learn Interface** (45 min)
# * What is Machine Learning?
# * Machine learning data layout
# * Supervised Learning
#     - Classification
#     - Regression
#     - Measuring performance
# * Unsupervised Learning
#     - Clustering
#     - Dimensionality Reduction
#     - Density Estimation
# * Evaluation of Learning Models
# * Choosing the right algorithm for your dataset
# 
# **Supervised learning in-depth** (1 hr)
# * Support Vector Machines
# * Decision Trees and Random Forests
# 
# **Unsupervised learning in-depth** (1 hr)
# * Principal Component Analysis
# * K-means Clustering
# * Gaussian Mixture Models
# 
# **Model Validation** (1 hr)
# * Validation and Cross-validation

# ## Preliminaries

# This tutorial requires the following packages:
# 
# - Python version 2.7 or 3.4+
# - `numpy` version 1.8 or later: http://www.numpy.org/
# - `scipy` version 0.15 or later: http://www.scipy.org/
# - `matplotlib` version 1.3 or later: http://matplotlib.org/
# - `scikit-learn` version 0.15 or later: http://scikit-learn.org
# - `ipython`/`jupyter` version 3.0 or later, with notebook support: http://ipython.org
# - `seaborn`: version 0.5 or later, used mainly for plot styling
# 
# The easiest way to get these is to use the [conda](http://store.continuum.io/) environment manager.
# I suggest downloading and installing [miniconda](http://conda.pydata.org/miniconda.html).
# 
# The following command will install all required packages:
# ```
# $ conda install numpy scipy matplotlib scikit-learn ipython-notebook
# ```
# 
# Alternatively, you can download and install the (very large) Anaconda software distribution, found at https://store.continuum.io/.

# ### Checking your installation
# 
# You can run the following code to check the versions of the packages on your system:
# 
# (in IPython notebook, press `shift` and `return` together to execute the contents of a cell)

# In[ ]:


from __future__ import print_function

import IPython
print('IPython:', IPython.__version__)

import numpy
print('numpy:', numpy.__version__)

import scipy
print('scipy:', scipy.__version__)

import matplotlib
print('matplotlib:', matplotlib.__version__)

import sklearn
print('scikit-learn:', sklearn.__version__)

# ## Useful Resources

# - **scikit-learn:** http://scikit-learn.org (see especially the narrative documentation)
# - **matplotlib:** http://matplotlib.org (see especially the gallery section)
# - **Jupyter:** http://jupyter.org (also check out http://nbviewer.jupyter.org)
