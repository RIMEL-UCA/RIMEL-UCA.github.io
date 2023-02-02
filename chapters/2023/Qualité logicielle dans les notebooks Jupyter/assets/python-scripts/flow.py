#!/usr/bin/env python
# coding: utf-8

# ## Example of Flow Usage<a class="anchor" id="home"></a>:
# * [Flow](#flow)
#     * [Sequential](#seq_pipeline)
#     * [Sometimes](#sometimes_pipeline)

# In[1]:


import os
os.environ["MODEL_DIR"] = '../model'

# # Config

# In[8]:


import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naf

from nlpaug.util import Action

# In[9]:


text = 'The quick brown fox jumps over the lazy dog .'
print(text)

# # Flow <a class="anchor" id="flow">
# 
# To make use of multiple augmentation, `sequential` and `sometimes` pipelines are introduced to connect augmenters.

# ### Sequential Pipeline<a class="anchor" id="seq_pipeline">

# ##### Apply different augmenters sequentially

# In[19]:


aug = naf.Sequential([
    nac.RandomCharAug(action="insert"),
    naw.RandomWordAug()
])

aug.augment(text)

# ### Generate mulitple synthetic data

# In[10]:


aug = naf.Sequential([
    nac.RandomCharAug(action="insert"),
    naw.RandomWordAug()
])

aug.augment(text, n=3)

# ### Sometimes Pipeline<a class="anchor" id="sometimes_pipeline">

# ##### Apply some augmenters randomly

# In[20]:


aug = naf.Sometimes([
    nac.RandomCharAug(action="delete"),
    nac.RandomCharAug(action="insert"),
    naw.RandomWordAug()
])

aug.augment(text)
