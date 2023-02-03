#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime as dt

from gs_quant.data.fields import DataMeasure
from gs_quant.markets.index import Index
from gs_quant.session import Environment, GsSession

# ## Pre Requisites
# To use below functionality on **STS Indices**, your application needs to have access to the following datasets:
# 1. [STS_FUNDAMENTALS](https://marquee.gs.com/s/developer/datasets/STS_FUNDAMENTALS) - Fundamental metrics of STS Indices
# 
# You can request access by going to the Dataset Catalog Page linked above.
# 
# Note - Please skip this if you are an internal user

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('read_product_data',))

# In[ ]:


index = Index.get('GSXXXXXX')  # substitute input with any identifier for a single stock STS index.

# In[ ]:


index.get_fundamentals() # default period is one year, default period direction is forward, default metrics are all available metrics

# In[ ]:


index.get_fundamentals(start=dt.date(2021, 1, 7), end=dt.date(2021, 3, 27))  # get fundamentals for the specified date range

# You may choose one of the following periods:
# 
# - **1 year:** DataMeasure.ONE_YEAR
# - **2 years:** DataMeasure.TWO_YEARS
# - **3 years:** DataMeasure.THREE_YEARS
# 
# You may choose one of the following period directions:
# 
# - **Forward:** DataMeasure.FORWARD
# - **Trailing:** DataMeasure.TRAILING

# In[ ]:


index.get_fundamentals(period=DataMeasure.TWO_YEARS, direction=DataMeasure.TRAILING)

# You may choose any combinations of the following metrics:
# 
# - **Dividend Yield:** DataMeasure.DIVIDEND_YIELD
# - **Earnings per Share:** DataMeasure.EARNINGS_PER_SHARE
# - **Earnings per Share Positive:** DataMeasure.EARNINGS_PER_SHARE_POSITIVE
# - **Net Debt to EBITDA:** DataMeasure.NET_DEBT_TO_EBITDA
# - **Price to Book:** DataMeasure.PRICE_TO_BOOK
# - **Price to Cash:** DataMeasure.PRICE_TO_CASH
# - **Price to Earnings:** DataMeasure.PRICE_TO_EARNINGS
# - **Price to Earnings Positive:** DataMeasure.PRICE_TO_EARNINGS_POSITIVE
# - **Price to Sales:** DataMeasure.PRICE_TO_SALES
# - **Return on Equity:** DataMeasure.RETURN_ON_EQUITY
# - **Sales per Share:** DataMeasure.SALES_PER_SHARE

# In[ ]:


index.get_fundamentals(metrics=[DataMeasure.PRICE_TO_CASH, DataMeasure.SALES_PER_SHARE])    

# *Have any other questions? Reach out to the [Marquee STS team](mailto:gs-marquee-sts-support@gs.com)!*
