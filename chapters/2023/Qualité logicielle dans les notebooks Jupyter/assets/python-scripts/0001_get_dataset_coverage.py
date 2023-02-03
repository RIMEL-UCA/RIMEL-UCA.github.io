#!/usr/bin/env python
# coding: utf-8

# Examples require an initialized GsSession and relevant entitlements. External clients need to substitute thier own client id and client secret below. Please refer to [Authentication](https://developer.gs.com/p/docs/institutional/platform/authentication/) for details.

# In[1]:


from gs_quant.session import GsSession
GsSession.use(client_id=None, client_secret=None, scopes=('read_product_data',))

# In[2]:


from gs_quant.data import Dataset

# In[3]:


# get the dataset
dataset = Dataset('EDRVOL_PERCENT_STANDARD')

# ### Dataset Coverage
# Dataset coverage is a list of the symbol dimensions in a dataset. This allows users to quickly see what is available in a dataset.
# Coverage can by fetched via API as below or in the [dataset product page](https://marquee.gs.com/s/developer/datasets/EDRVOL_PERCENT_STANDARD/coverage)

# In[4]:


# get the coverage
coverage = dataset.get_coverage(include_history=True)

# show the number of records and columns
coverage.shape

# In[5]:


# show the columns names
coverage.keys()

# In[6]:


# show the coverage
coverage

# In[7]:


# find specific asset
coverage[coverage['bbid'] == 'SPX']
