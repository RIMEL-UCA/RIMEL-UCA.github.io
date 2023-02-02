#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.data import DataMeasure
from gs_quant.markets.securities import SecurityMaster, AssetIdentifier, ExchangeCode
from gs_quant.session import Environment, GsSession

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterh
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('read_product_data',))

# In[ ]:


# Lookup S&P 500 Index via Security Master
spx = SecurityMaster.get_asset('SPX', AssetIdentifier.TICKER, exchange_code=ExchangeCode.NYSE)

# Resolve the data coordinate for SPX without explicitly specifying the dataset
spot_data_coordinate = spx.get_data_coordinate(DataMeasure.SPOT_PRICE)

# Fetch the series
series = spot_data_coordinate.get_series()

# In[ ]:


series.tail()
