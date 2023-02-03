#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime as dt
from gs_quant.markets.index import Index
from gs_quant.session import Environment, GsSession

# ## Pre Requisites
# To use below functionality on **STS Indices**, your application needs to have access to the following datasets:
# 1. [STSCONSTITUENTS](https://marquee.gs.com/s/developer/datasets/STSCONSTITUENTS) - Bottom level index constituents and associated weights
# 
# You can request access by going to the Dataset Catalog Page linked above.
# 
# Note - Please skip this if you are an internal user

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('read_product_data',))

# In[ ]:


index = Index.get('GSXXXXXX')  # substitute input with any identifier for an index

# ## Get Constituents
# 
# Returns a pandas dataframe of the constituents

# In[ ]:


index.get_latest_constituents()      # returns the latest constituents of the index.

# In[ ]:


index.get_constituents_for_date(dt.date(2021, 7, 1))      # returns the constituents of the index for the specified date

# In[ ]:


index.get_constituents(dt.date(2021, 6, 1), dt.date(2021, 6, 10))     # returns the constituents sets of the index for the specified date range

# ## Get Constituents as Instruments
# 
# Returns the constituents as Instrument objects

# In[ ]:


index.get_latest_constituent_instruments()      # returns the latest constituents of the index as instrument objects.

# In[ ]:


index.get_constituent_instruments_for_date(dt.date(2021, 7, 1))      # returns the constituents of the index for the specified date as Instrument objects

# In[ ]:


index.get_constituent_instruments(dt.date(2021, 6, 1), dt.date(2021, 6, 10))     # returns the constituents sets of the index for the specified date range as Instrument objects

# *Have any other questions? Reach out to the [Marquee STS team](mailto:gs-marquee-sts-support@gs.com)!*
