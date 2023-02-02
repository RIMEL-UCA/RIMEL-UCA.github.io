#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.markets.indices_utils import *
from gs_quant.session import Environment, GsSession

# In[ ]:


client = 'CLIENT ID'
secret = 'CLIENT SECRET'

GsSession.use(Environment.PROD, client_id=client, client_secret=secret, scopes=('read_product_data',))

# You may choose any combination of the following basket types:
# 
# * **Custom Basket:** BasketType.*CUSTOM_BASKET*
# * **Research Basket:** BasketType.*RESEARCH_BASKET*
# 
# These options will work with any of the following functions:

# In[ ]:


get_flagship_baskets(basket_type=[BasketType.CUSTOM_BASKET])

get_flagships_with_assets(identifiers=['AAPL UW'], basket_type=[BasketType.CUSTOM_BASKET])

get_flagships_performance(basket_type=[BasketType.RESEARCH_BASKET])

get_flagships_constituents(basket_type=[BasketType.RESEARCH_BASKET])
