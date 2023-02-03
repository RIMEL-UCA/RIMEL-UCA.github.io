#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.markets.securities import SecurityMaster, AssetIdentifier
from gs_quant.session import GsSession, Environment

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterh

GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('read_product_data',))

# In[ ]:


asset = SecurityMaster.get_asset('2407966', AssetIdentifier.SEDOL, sort_by_rank=True)


# In[ ]:


asset

# In[ ]:


asset.get_identifiers() # get all identifiers of the asset as of now
