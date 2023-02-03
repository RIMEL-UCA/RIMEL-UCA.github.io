#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# The MarketDataPattern class allows the user to select a collection of observable market data
# MarketDataPattern arguments include: mkt_type, mkt_asset, mkt_class, mkt_point and mkt_quoting_style
from gs_quant.risk import MarketDataPattern

# In[ ]:


# A MarketDataPattern defined with 'FX' mkt_type contains FX Spot rate data
fx = MarketDataPattern(mkt_type='FX')

# In[ ]:


# The collection of 'FX' market data can be further constrained by setting the 'mkt_asset'
# For data with mkt_type 'FX', the mkt_asset is a cross against USD
fx_eur = MarketDataPattern(mkt_type='FX', mkt_asset='USD/EUR')
fx_gbp = MarketDataPattern('FX', 'USD/GBP')
fx_chf = MarketDataPattern('FX', 'CHF/USD')
