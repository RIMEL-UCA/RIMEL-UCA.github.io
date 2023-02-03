#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.session import Environment, GsSession
from gs_quant.markets.index import Index

# ## Pre Requisites
# To use below functionality on **STS Indices**, your application needs to have access to the following datasets:
# 4. [STS_UNDERLIER_WEIGHTS](https://marquee.gs.com/s/developer/datasets/STS_UNDERLIER_WEIGHTS) - Weights of index underliers of STS Indices
# 5. [STS_UNDERLIER_ATTRIBUTION](https://marquee.gs.com/s/developer/datasets/STS_UNDERLIER_ATTRIBUTION) - Attribution of index underliers
# 
# You can request access by going to the Dataset Catalog Page linked above.
# 
# Note - Please skip this if you are an internal user

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('read_product_data',))

# In[ ]:


strategy_index = Index.get('GSXXXXXX') # substitute input with any identifier for an index

# ### These functions currently supports STS indices only

# In[ ]:


# Returns a pandas dataframe containing latest weights of the immediate underliers of the index
strategy_index.get_underlier_weights()

# In[ ]:


# Returns a pandas dataframe containing latest attribution of the immediate underliers of the index
strategy_index.get_underlier_attribution()

# In[ ]:


# Get the latest weights and attribution of the underliers at all levels
strategy_index.get_underlier_tree().to_frame()

# *Have any other questions? Reach out to the [Marquee STS team](mailto:gs-marquee-sts-support@gs.com)!*
