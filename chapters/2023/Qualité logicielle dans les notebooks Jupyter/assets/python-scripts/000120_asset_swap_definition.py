#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.session import Environment, GsSession
from gs_quant.instrument import IRAssetSwapFxdFlt

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


# The default instrument is a 0 PV asset swap on a US10Y Bond, which price is defaulted to 100
default = IRAssetSwapFxdFlt()
default.resolve()
default.as_dict()

# In[ ]:


default.price()

# In[ ]:


# We may change the underlier bond and price which will resolve the other attributes accordingly
asset_swap = IRAssetSwapFxdFlt(identifier='GB5Y', traded_clean_price=99)
asset_swap.resolve()
asset_swap.as_dict()

# In[ ]:


asset_swap.price()
