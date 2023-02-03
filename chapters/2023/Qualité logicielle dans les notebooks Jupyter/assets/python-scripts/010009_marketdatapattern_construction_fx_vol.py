#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# The MarketDataPattern class allows the user to select a collection of observable market data
# MarketDataPattern arguments include: mkt_type, mkt_asset, mkt_class, mkt_point and mkt_quoting_style
from gs_quant.risk import MarketDataPattern

# In[ ]:


# A MarketDataPattern defined with 'FX Vol' mkt_type contains market data related to the implied volatility for
# FX instruments
fx_vol = MarketDataPattern(mkt_type='FX Vol')

# In[ ]:


# The collection of 'FX Vol' market data can be further constrained by setting the'mkt_asset'
# For data with mkt_type 'FX Vol', the mkt_asset is a cross
fx_vol_gbp_eur = MarketDataPattern(mkt_type='FX Vol', mkt_asset='GBP/EUR', mkt_class='ATM Vol')
fx_vol_dkk_usd = MarketDataPattern('FX Vol', 'DKK/USD')
fx_vol_jpy_aud = MarketDataPattern('FX Vol', 'JPY/AUD')
