#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# The MarketDataPattern class allows the user to select a collection of observable market data
# MarketDataPattern arguments include: mkt_type, mkt_asset, mkt_class, mkt_point and mkt_quoting_style
from gs_quant.risk import MarketDataPattern

# In[ ]:


# A MarketDataPattern defined with 'IR XC' mkt_type contains market data related to cross currency risk
ir_xc = MarketDataPattern(mkt_type='IR XC')

# In[ ]:


# The collection of 'IR XC' market data can be further constrained by setting the 'mkt_asset'
# For data with mkt_type 'IR XC', mkt_asset is a single currency, which denotes the cross-currency curve 
# (the large majority of which are vs. USD)
ir_xc_eur = MarketDataPattern(mkt_type='IR XC', mkt_asset='EUR')
ir_xc_gbp = MarketDataPattern('IR XC', 'GBP')
ir_xc_chf = MarketDataPattern('IR XC', 'CHF')
