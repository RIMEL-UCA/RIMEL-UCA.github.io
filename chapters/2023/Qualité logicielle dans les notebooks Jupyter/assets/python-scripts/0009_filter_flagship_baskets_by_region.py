#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.markets.indices_utils import *
from gs_quant.session import Environment, GsSession

# In[ ]:


client = 'CLIENT ID'
secret = 'CLIENT SECRET'

GsSession.use(Environment.PROD, client_id=client, client_secret=secret, scopes=('read_product_data',))

# You may choose any combination of the following regions:
# 
# * **Americas:** Region.*AMERICAS*
# * **Asia:** Region.*ASIA*
# * **EM:** Region.*EM*
# * **Europe:** Region.*EUROPE*
# * **Global:** Region.*GLOBAL*
# 
# These options will work with any of the following functions:

# In[ ]:


get_flagship_baskets(region=[Region.ASIA])

get_flagships_with_assets(identifiers=['AAPL UW'], region=[Region.AMERICAS])

get_flagships_performance(region=[Region.EUROPE, Region.GLOBAL])

get_flagships_constituents(region=[Region.EM])
