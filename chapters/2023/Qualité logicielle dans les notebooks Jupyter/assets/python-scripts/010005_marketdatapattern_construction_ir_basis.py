#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# The MarketDataPattern class allows the user to select a collection of observable market data
# MarketDataPattern arguments include: mkt_type, mkt_asset, mkt_class, mkt_point and mkt_quoting_style
from gs_quant.risk import MarketDataPattern

# In[ ]:


# A MarketDataPattern defined with 'IR Basis' mkt_type contains market data for rates spreads such as tenor basis,
# funding/libor basis, etc.
ir_basis = MarketDataPattern(mkt_type='IR Basis')

# In[ ]:


# The collection of 'IR Basis' market data can be further constrained by setting the 'mkt_asset'
# For data with mkt_type 'IR Basis', the mkt_asset is the basis curve, represented by two currency/tenor pairs
ir_basis_usd_ois1m = MarketDataPattern(mkt_type='IR Basis', mkt_asset='USD OIS/USD-1M')
ir_basis_eur_3m6m = MarketDataPattern('IR Basis', 'EUR-3M/EUR-6M')
ir_basis_gbp_3m6m = MarketDataPattern('IR Basis', 'GBP-3M/GBP-6M')
