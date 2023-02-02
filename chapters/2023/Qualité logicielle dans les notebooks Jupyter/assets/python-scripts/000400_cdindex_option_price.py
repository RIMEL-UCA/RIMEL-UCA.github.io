#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import date
from gs_quant.common import PayReceive, Currency
from gs_quant.instrument import CDIndexOption
from gs_quant.session import Environment, GsSession
from gs_quant.markets import PricingContext
from gs_quant.risk import CDATMSpread, CDFwdSpread, CDImpliedVolatility

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


# View properties of a CD Index Option
CDIndexOption.properties()

# Available properties (defaults) for determining the underlying index are:
# - index_family (iTraxx Europe): 'iTraxx Europe', 'iTraxx Europe FNSE', 'iTraxx Europe XOVER', 'CDX.NA.IG', 'CDX.NA.HY'
# - index_series (OTR): OTR
# - index_version (latest): latest
# 
# Available properties for determining the option are:
# - option_type (Call): 'Call', 'Put'
# - strike (0.01): Any
# - strike_type (Spread): 'Spread', 'Price'
# - expriation_date (next market expiry - third Wednesday of month): Between 1m and 6m
# - termination_date (5y next IMM date): 5y
# - notional_amount (10000000): Any

# In[ ]:


# Create a 3 month call on an OTR iTraxx Main 5y underlier with a strike of 65
cd_option = CDIndexOption(strike=0.0065, option_type='Call', expiration_date='19Jan22', termination_date='5y',  
                          index_family='iTraxx Europe FNSE', notional_amount=10000000)

# In[ ]:


with PricingContext(market_data_location='NYC'):
    cd_option.resolve()
cd_option.as_dict()

# In[ ]:


print(cd_option.price())

# For index options we can also calculate the ATM (reference) spread, forward spread and implied annual volatility used for pricing the option
# 
# - CDATMSpread: At the money value of the index in spread terms
# - CDFwdSpread: Expected forward value of the underlying index at option expiry, in spread terms
# - CDImpliedVolatility: Annual volatility for the underlying index hazard rate 

# In[ ]:


cd_option.calc(CDATMSpread)

# In[ ]:


cd_option.calc(CDFwdSpread)

# In[ ]:


cd_option.calc(CDImpliedVolatility)
