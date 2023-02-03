#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# The MarketDataPattern class allows the user to select a collection of observable market data
# MarketDataPattern arguments include: mkt_type, mkt_asset, mkt_class, mkt_point and mkt_quoting_style
from gs_quant.risk import MarketDataPattern

# In[ ]:


# A MarketDataPattern defined with 'FX Fwd' mkt_type contains FX Forward curve data
fx_fwd = MarketDataPattern(mkt_type='FX Fwd')

# In[ ]:


# The collection of 'FX Fwd' market data can be further constrained by setting the 'mkt_asset'
# For data with mkt_type 'FX Fwd', the mkt_asset is a cross 
# Note - FX Fwd market data are represented as USD crosses

fx_fwd_eur = MarketDataPattern(mkt_type='FX Fwd', mkt_asset='USD/EUR')
fx_fwd_dkk = MarketDataPattern('FX Fwd', 'DKK/USD')
fx_fwd_jpy = MarketDataPattern('FX Fwd', 'JPY/USD')
