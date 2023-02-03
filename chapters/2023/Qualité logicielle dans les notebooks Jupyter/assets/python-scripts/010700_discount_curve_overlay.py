#!/usr/bin/env python
# coding: utf-8

# # Curve Overlay Context
# Scenario contexts enable the users to price and calculate risk using their own curves.

# In[1]:


from gs_quant.instrument import IRSwaption
from gs_quant.risk import MarketDataPattern, MarketDataShock, MarketDataShockType, MarketDataShockBasedScenario, \
    RollFwd, CurveScenario, IndexCurveShift, CurveOverlay
from gs_quant.session import Environment, GsSession
from datetime import datetime

# In[2]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[3]:


# Create and price a swaption
swaption = IRSwaption('Pay', '5y', 'USD', expiration_date='3m')
base_price = swaption.price()
print('Original price: {:,.2f}'.format(base_price))

# ### Curve Overlay Scenario 
# A predefined scenario used to overlay discount curve or index curve. This allows the user to use customized discount factors to overwrite existing curves on graph.
# * discount_factors - Discount factors
# * dates - Dates of the discount factors.
# * curve_type - Discount curve or index curve.
# * rate_option - Rate option of the index curve.
# * tenor - Tenor of the index curve.
# * csa_term - The funding scenario of the curve.
# * denominated - The denominated currency of the curve.
# 

# In[4]:


import pandas as pd

# Read discount factors from a csv file
df = pd.read_csv (r'CurveExample.csv', sep="," )
dates = df.MaturityDate.tolist()
dates_reformat = [ datetime.strptime(date, "%d-%b-%Y").strftime("%Y-%m-%d") for date in dates ]
discount_factors = df.DiscountFactor.tolist()

curve_overlay_scenario = CurveOverlay(
    dates = dates_reformat,
    discount_factors = discount_factors,
    denominated = "USD",
    csa_term = "USD-SOFR",
    curve_type = "Discount Curve"
)
with curve_overlay_scenario:
    price_with_overlay = swaption.price()

print('Scenario price: {:,.2f}'.format(price_with_overlay))
