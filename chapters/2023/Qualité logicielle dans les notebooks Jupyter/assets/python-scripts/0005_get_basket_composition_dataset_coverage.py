#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd

from gs_quant.api.gs.assets import GsAssetApi
from gs_quant.markets.indices_utils import BasketType
from gs_quant.session import Environment, GsSession

client = 'CLIENT_ID'
secret = 'CLIENT_SECRET'

GsSession.use(Environment.PROD, client_id=client, client_secret=secret, scopes=('read_product_data',))

# In[13]:


fields = ['id', 'name', 'region', 'ticker', 'type']

custom_baskets = GsAssetApi.get_many_assets_data_scroll(fields=fields, listed=[True], type=[BasketType.CUSTOM_BASKET])
research_baskets = GsAssetApi.get_many_assets_data_scroll(fields=fields, listed=[True], type=[BasketType.RESEARCH_BASKET])

GSBASKETCONSTITUENTS_COVERAGE = pd.DataFrame(custom_baskets)
GIRBASKETCONSTITUENTS_COVERAGE = pd.DataFrame(research_baskets)
