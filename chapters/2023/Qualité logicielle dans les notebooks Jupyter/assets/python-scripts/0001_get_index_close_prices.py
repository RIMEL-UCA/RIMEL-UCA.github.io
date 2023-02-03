#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.markets.index import Index, PriceType
from gs_quant.session import Environment, GsSession
import datetime as dt

# ## Pre Requisites
# To use below functionality on **STS Indices**, your application needs to have access to the following datasets:
# 1. [STSLEVELS](https://marquee.gs.com/s/developer/datasets/STSLEVELS) - Official Values of STS Indices
# 2. [STS_INDICATIVE_LEVELS](https://marquee.gs.com/s/developer/datasets/STS_INDICATIVE_LEVELS) - Indicative Values of STS Indices
# 
# You can request access by going to the Dataset Catalog Page linked above.
# 
# Note - Please skip this if you are an internal user

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('read_product_data',))

# In[ ]:


index = Index.get('GSXXXXXX')  # substitute input with any identifier for an index

# #### Close price functions supports the following price types

# You may choose one of the following price types:
# 
# - **Official Price:** PriceType.OFFICIAL_PRICE
# - **Indicative Price** PriceType.INDICATIVE_CLOSE_PRICE - Currently supports STS indices only.
# 
# Default returns the official close price

# In[ ]:


index.get_latest_close_price(price_type=[PriceType.OFFICIAL_CLOSE_PRICE])      # returns latest official levels for the index.

# In[ ]:


index.get_close_price_for_date(dt.date(2021, 1, 7), price_type=[PriceType.OFFICIAL_CLOSE_PRICE])  # returns official levels for a given date.

# In[ ]:


index.get_close_prices(start=dt.date(2021, 1, 7), end=dt.date(2021, 3, 27), price_type=[PriceType.OFFICIAL_CLOSE_PRICE]) # returns official levels for a given date range.

# In[ ]:


index.get_close_prices(price_type=[PriceType.OFFICIAL_CLOSE_PRICE]) # returns all the official levels of the index.

# #### STS indices can use PriceType.INDICATIVE_CLOSE_PRICE as well to get the indicative levels

# In[ ]:


index.get_latest_close_price(price_type=[PriceType.OFFICIAL_CLOSE_PRICE, PriceType.INDICATIVE_CLOSE_PRICE])      # returns latest indicative and official levels of the index.

# In[ ]:


index.get_close_price_for_date(dt.date(2021, 1, 7), price_type=[PriceType.OFFICIAL_CLOSE_PRICE, PriceType.INDICATIVE_CLOSE_PRICE])  # returns both indicative and official levels of the index for a given date.

# In[ ]:


index.get_close_prices(start=dt.date(2021, 1, 7), end=dt.date(2021, 3, 27), price_type=[PriceType.OFFICIAL_CLOSE_PRICE, PriceType.INDICATIVE_CLOSE_PRICE]) # returns both indicative and official levels of the index for a given date range.

# In[ ]:


index.get_close_prices(price_type=[PriceType.OFFICIAL_CLOSE_PRICE, PriceType.INDICATIVE_CLOSE_PRICE]) # returns all the indicative and official levels of the index.

# *Have any other questions? Reach out to the [Marquee STS team](mailto:gs-marquee-sts-support@gs.com)!*
