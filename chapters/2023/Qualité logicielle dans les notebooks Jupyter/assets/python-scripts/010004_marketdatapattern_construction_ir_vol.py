#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# The MarketDataPattern class allows the user to select a collection of observable market data
# MarketDataPattern arguments include: mkt_type, mkt_asset, mkt_class, mkt_point and mkt_quoting_style
from gs_quant.risk import MarketDataPattern

# In[ ]:


# A MarketDataPattern defined with 'IR Vol' mkt_type contains market data related to implied volatility for 
# IR Swaptions/Caplets
ir_vol = MarketDataPattern(mkt_type='IR_Vol')

# In[ ]:


# The collection of 'IR Vol' market data can be further constrained by setting the'mkt_asset'
# For data with mkt_type 'IR Vol', the mkt_asset is a benchmark rate
ir_vol_eur_euribor = MarketDataPattern(mkt_type='IR Vol', mkt_asset='EUR-EURIBOR-TELERATE')
ir_vol_gbp_libor = MarketDataPattern('IR Vol', 'GBP-LIBOR-BBA')
ir_vol_chf_libor = MarketDataPattern('IR Vol', 'CHF-LIBOR-BBA')

# In[ ]:


# Additionally we can select subset of curve instruments using the 'mkt_class'
# valid mkt_classes for 'IR Vol' data are 'swaption' and 'caplet' 
ir_vol_gbp_libor_swaption = MarketDataPattern(mkt_type = 'IR Vol', mkt_asset = 'GBP-LIBOR-BBA', mkt_class ='swaption')
ir_vol_gbp_libor_caplet = MarketDataPattern('IR Vol', 'GBP-LIBOR-BBA', 'caplet')
