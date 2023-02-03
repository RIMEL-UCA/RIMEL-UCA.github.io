#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import date

import gs_quant.timeseries as ts
from gs_quant.data import DataContext
from gs_quant.markets.securities import SecurityMaster, AssetIdentifier, ExchangeCode
from gs_quant.session import GsSession, Environment

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('read_product_data',))

# In[ ]:


# Create a data context covering 2018
data_ctx = DataContext(start=date(2018, 1, 1), end=date(2018, 12, 31))

# Lookup S&P 500 Index via Security Master
spx = SecurityMaster.get_asset('SPX', AssetIdentifier.TICKER, exchange_code=ExchangeCode.NYSE)

# Use the data context
with data_ctx:
    # Get 25 delta call implied volatility
    vol = ts.implied_volatility(spx, '1m', ts.VolReference.DELTA_CALL, 25)

# In[ ]:


vol.tail()

