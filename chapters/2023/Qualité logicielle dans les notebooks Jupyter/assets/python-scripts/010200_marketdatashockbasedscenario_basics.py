#!/usr/bin/env python
# coding: utf-8

# In[1]:


# The gs_quant risk package contains Scenarios, which allow the user to perform scenario analysis on a set of trades 
# 'MarketDataShockBasedScenario' is the most flexible scenario, which allows users to construct bespoke market data 
# scenarios
from gs_quant.risk import MarketDataShockBasedScenario
MarketDataShockBasedScenario?

# In[3]:


# MarketDataShockBasedScenarios has one property: 'shocks', a mapping of MarketDataPattern to MarketDataShock
MarketDataShockBasedScenario.shocks

# In[ ]:


# The MarketDataPattern defines a collection of observable Market Data. Examples for the MarketDataPattern class are
# in gs_quant/examples/01_pricing_and_risk/01_market_objects.
