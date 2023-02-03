#!/usr/bin/env python
# coding: utf-8

# This notebook presents a working example of adjusting texts for multiple subplots, related to https://github.com/Phlya/adjustText/issues/58

# In[1]:


import matplotlib.pyplot as plt # Matplotlib 2.0 shown here
from adjustText import adjust_text
import numpy as np
import pandas as pd

# # With multiple subplots, run `adjust_text` for one subplot at a time

# In[2]:


fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
axes = axes.ravel()

for k, ax in enumerate(axes):
    np.random.seed(0)
    x, y = np.random.random((2,30))
    ax.plot(x, y, 'bo')

    texts = []
    for i in range(len(x)):
        t = ax.text(x[i], y[i], 'Text%s' %i, ha='center', va='center')
        texts.append(t)
    %time adjust_text(texts, ax=ax)

# **Note** that there is time discrepancy between the two subplots, which is under investigation.

# In[ ]:



