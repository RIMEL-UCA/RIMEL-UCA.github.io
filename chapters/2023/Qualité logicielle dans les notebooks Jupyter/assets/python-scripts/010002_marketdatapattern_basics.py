#!/usr/bin/env python
# coding: utf-8

# ### Market Data Representation
# Market data is a key part of gs_quant. The `MarketDataPattern` class allows the user to select a collection of this 
# observable market data.

# In[ ]:


from gs_quant.risk import MarketDataPattern
MarketDataPattern?

# A `MarketDataPattern` can be instantiated with any combination of the following attributes:
# 
# - `mkt_type`  - natural category of risk representation
# - `mkt_asset` - marked observable
# - `mkt_class` - market priceable
# - `mkt_point`  - unique identifier for the marked observable within a mkt_class
# 
# The highest level of market data classification is the `mkt_type`. Examples of mkt_type include: 'IR', 'IR Vol' and 'FX'

# In[ ]:


# you can instantiate a MarketDataPattern with keyword arguments
ir_mdp = MarketDataPattern(mkt_type='IR')

# In[ ]:


# or positional arguments
fx_mdp2 = MarketDataPattern('FX')
