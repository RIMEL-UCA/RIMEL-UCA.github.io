#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.common import OptionType, OptionStyle
from gs_quant.instrument import EqOption
from gs_quant.markets import PricingContext
from gs_quant.markets.securities import AssetIdentifier
from gs_quant.risk.scenario_utils import *
from gs_quant.session import Environment, GsSession
from gs_quant.target.common import UnderlierType
from datetime import date

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics','read_product_data'))

# In[ ]:


# Construct a vol override scenario.  

# This example uses a sample vol dataset EDRVOL_PERCENT_EXPIRY_PREMIUM_SAMPLE
# Full dataset and additional info available in the data catalog 
# https://marquee.gs.com/s/developer/datasets/EDRVOL_PERCENT_EXPIRY_PREMIUM

eq_vol_scenario = build_eq_vol_scenario_eod('TMK UN', 'EDRVOL_PERCENT_EXPIRY_PREMIUM_SAMPLE', 
                                            vol_date=date(2019, 6, 3), asset_name_type=AssetIdentifier.BLOOMBERG_ID)

# In[ ]:


# Define an option and price with and without the vol override scenario

option = EqOption('TMK UN', underlierType=UnderlierType.BBID, expirationDate='3m', strikePrice=95, 
                  optionType=OptionType.Call, optionStyle=OptionStyle.European)

with PricingContext(date(2019,6,10)):
    historic_option_price = option.price()      

with PricingContext(date(2019,6,10)), eq_vol_scenario:
    historic_option_vol_scenario_price = option.price()       

# Look at the difference between scenario and base prices

print('Base price:     {:,.4f}'.format(historic_option_price.result()))
print('Scenario price: {:,.4f}'.format(historic_option_vol_scenario_price.result()))
