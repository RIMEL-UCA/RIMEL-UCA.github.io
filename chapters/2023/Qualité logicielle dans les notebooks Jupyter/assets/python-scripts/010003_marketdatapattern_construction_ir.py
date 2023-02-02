#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# The MarketDataPattern class allows the user to select a collection of observable market data
# MarketDataPattern arguments include: mkt_type, mkt_asset, mkt_class, mkt_point and mkt_quoting_style
from gs_quant.risk import MarketDataPattern

# In[ ]:


# A MarketDataPattern defined with 'IR' mkt_type contains market data related to IR Delta for rates instruments 
# (such as swaps, eurodollar futures, fras, etc) 

ir = MarketDataPattern(mkt_type='IR')

# In[ ]:


# The collection of 'IR' market data can be further constrained by setting the'mkt_asset'
# For data with mkt_type 'IR', the mkt_asset represents a rate curve's denominated currency
ir_gbp = MarketDataPattern(mkt_type ='IR', mkt_asset='GBP')
ir_jpy = MarketDataPattern('IR', 'JPY')
ir_usd = MarketDataPattern('IR', 'USD')
