#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.session import Environment, GsSession
from gs_quant.markets.index import Index

# ## Pre Requisites
# To use below functionality on **STS Indices**, your application needs to have access to the following datasets:
# 1. [STS_UNDERLIER_WEGHTS](https://marquee.gs.com/s/developer/datasets/STS_UNDERLIER_WEGHTS) - Weights of index underliers of STS Indices
# 5. [STS_UNDERLIER_WEGHTS](https://marquee.gs.com/s/developer/datasets/STS_UNDERLIER_WEGHTS) - Attribution of index underliers
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


# Returns the root node of the underlier tree formed by the index 
strategy_index.get_underlier_tree()

# To know more about how to use the AssetTreeNode object, please refer to the example notebook for the Tree Entity class, found in gs_quant/documentation/08_tree_entity

# In[ ]:


# Returns the full tree formed by the the index as a pandas dataframe
strategy_index.get_underlier_tree().to_frame()

# In[ ]:


# This functionality requires treelib to be installed
# Prints the tree structure formed by the Index for easy visualisation
strategy_index.visualise_tree(visualise_by='name')

# If data is missing for any field, then assetId will be used instead
strategy_index.visualise_tree(visualise_by='bbid')

strategy_index.visualise_tree(visualise_by='id')


# In[ ]:


# To refresh and rebuild the tree
strategy_index.get_underlier_tree(refresh_tree=True)

# *Have any other questions? Reach out to the [Marquee STS team](mailto:gs-marquee-sts-support@gs.com)!*
