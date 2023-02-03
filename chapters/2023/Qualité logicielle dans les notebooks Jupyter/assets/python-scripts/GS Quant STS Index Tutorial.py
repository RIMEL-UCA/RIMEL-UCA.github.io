#!/usr/bin/env python
# coding: utf-8

# # GS Quant STS Index Tutorial

# Welcome to the GS Quant Index tutorial!  This tutorial will walk you through using the Index class functionality in GS Quant with a focus on STS Indices.
# 
# Marquee [Systematic Trading Strategy (STS)](https://marquee.gs.com/welcome/products/index-solutions/systematic-trading-strategies) offering provides a variety of information for STS Indices including prices, compositions and PnL attribution.
# GS Quant makes accessing this data via API intuitive and fast.
# 
# In this tutorial, you will fetch an STS Index and learn how to:
# 1. Get strategy values
# 3. Get bottom level composition
# 4. Get full underlier tree
# 5. Get factor risk data
# 2. Get fundamentals metrics
# 
# 
# Please ensure that you have followed the setup instructions for GS Quant mentioned [here](https://developer.gs.com/docs/gsquant/getting-started/). 

# ## Pre Requisites
# To use below functionality on **STS Indices**, your application needs to have access to the following datasets:
# 1. [STSLEVELS](https://marquee.gs.com/s/developer/datasets/STSLEVELS) - Official Values of STS Indices
# 2. [STS_INDICATIVE_LEVELS](https://marquee.gs.com/s/developer/datasets/STS_INDICATIVE_LEVELS) - Indicative Values of STS Indices
# 3. [STS_FUNDAMENTALS](https://marquee.gs.com/s/developer/datasets/STS_FUNDAMENTALS) - Fundamental metrics of STS Indices
# 4. [STS_UNDERLIER_WEIGHTS](https://marquee.gs.com/s/developer/datasets/STS_UNDERLIER_WEIGHTS) - Weights of index underliers of STS Indices
# 5. [STS_UNDERLIER_ATTRIBUTION](https://marquee.gs.com/s/developer/datasets/STS_UNDERLIER_ATTRIBUTION) - Attribution of index underliers
# 6. [STSCONSTITUENTS](https://marquee.gs.com/s/developer/datasets/STSCONSTITUENTS) - Bottom level index constituents and associated weights
# 
# You can request access by going to the Dataset Catalog Page linked above.
# 
# **Note** - If you're using GS Quant as an internal user, you need to raise the above for yourself. 

# # First Steps
# ### 1. Authenticate & Initialize your session
# 
# First you will import the necessary modules and add your client id and client secret. To checke your app's authentication, follow our [Getting Started guide.](https://developer.gs.com/docs/gsquant/authentication/gs-session/)

# In[ ]:


from gs_quant.markets.index import Index
from gs_quant.markets.indices_utils import *
from gs_quant.data.fields import DataMeasure
from gs_quant.session import Environment, GsSession

import datetime as dt

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('read_product_data',))

# ### 2. Fetch the index information
# 
# #### Load the index
# Next you will fetch the index you'd like to work with, by passing in any of its identifiers such as Bloomberg Id, Ticker, etc.
# In this tutorial, we're using some sample STS indices:
# 1. GSMBMA5S - A multi-asset strategy
# 2. GSISM37E - A strategy based on single stocks
# 
# To access any STS Index, your app should have access to view that index. Please note that **apps do not automatically inherit the access of their owners**. This means that your apps need to be separately granted access to the required STS Indices.
# 
# To request access to the above samples, please [send an email](mailto:gs-sts-marquee-permissioning-global@gs.com?cc=gs-marquee-sts-support@gs.com&subject=Request%20to%20access%20<index%20identifier>%20via%20API&body=I%20would%20like%20to%20access%20the%20index%20<index%20identifier>%20via%20my%20app%20<app%20client%20id>) to our team mentioning your app's id.
# 
# If you app already has access to some other STS indices, you can use them as well. Just replace the ticker below with the ticker of your index.

# In[ ]:


ma_index = Index.get('GSMBMA5S') # Substitute input with the identifier for an Index
sstk_index  = Index.get('GSISM37E')

# #### Get the Index URL on Marquee

# In[ ]:


# Retrieves the url of the product page of the index on Marquee
ma_index.get_url()

# In[ ]:


start_date = dt.date(2022, 3, 1)
end_date = dt.date(2022, 4, 1)

# # Strategy Values of an STS Index
# 
# In general, all Indices support official close prices but for STS Indices, official and indicatvive strategy values are supported. 

# The strategy values of an STS Index can be obtained by using the `get_close_prices` function. You may choose one of the following price types:
# 
# **Official Strategy Values**: PriceType.OFFICIAL_PRICE <br>
# **Indicative Strategy Values**: PriceType.INDICATIVE_CLOSE_PRICE - _Currently available for STS indices only._

# ### Get Official Close Prices

# In[ ]:


# Returns strategy values between start and end date. When no price_type is passed, it defaults to official strategy values
ma_index.get_close_prices(start=start_date,
                       end=end_date)

# Returns official strategy values between start and end date 
sstk_index.get_close_prices(start=start_date,
                       end=end_date,
                       price_type=[PriceType.OFFICIAL_CLOSE_PRICE])

# ### Get Indicative Close Prices

# In[ ]:


# Returns indicative strategy values between start and end date 
ma_index.get_close_prices(start=start_date,
                       end=end_date,
                       price_type=[PriceType.INDICATIVE_CLOSE_PRICE])

# ### Get all close prices

# In[ ]:


# Returns official and indicative strategy values between start and end date for the STS index
ma_index.get_close_prices(start=start_date,
                       end=end_date,
                       price_type=[PriceType.OFFICIAL_CLOSE_PRICE, PriceType.INDICATIVE_CLOSE_PRICE])

# # Composition Data of STS Index
# 
# The composition of an STS Index can be understood as a tree structure - the index is the topmost node, and it has some child nodes.
# These child nodes can either be bottom level assets or have child nodes of their own.
# The intermediate child nodes (i.e. those which have their own children) are called **underliers**. While the bottom level assets are called **constituents**.
# 
# ![Tree](./images/index_composition_diagram.png)

# <a id="constituents"></a>
# 
# ### Get Constituents
# 
# The constituents of an Index are the bottom level assets. Fetch the constituents of the index using the `get_constituents` method.
# 
# You can also get convert the constituents of an index into Instrument objects.

# In[ ]:


# Returns constituents of the index as a pandas DataFrame object
sstk_index.get_constituents_for_date(date=start_date)

# In[ ]:


# Returns constituents of the index as a list of instrument class objects
sstk_instruments = sstk_index.get_constituent_instruments_for_date(date=start_date)

# ### Get Underlier Tree
# 
# You can fetch the underlier tree of an index using the `get_underlier_tree` method. This method returns the composition tree alongwith the weights and attributions of the nodes.

# In[ ]:


# Returns the top node of the tree structure formed by the index
ma_index.get_underlier_tree()

# In[ ]:


# Returns the tree structure formed by the index as a pandas dataframe
ma_index.get_underlier_tree().to_frame()

# We proVisualisation functions require installing the `treelib` package, so let's go ahead and install it.
# 
# *Note:* This is part of GS Quant's `notebook` dependencies. You can run `pip install gs-quant[notebook]` to install all of them.

# In[ ]:


!pip install treelib

# In[ ]:


# Prints the tree structure formed by the Index for easy visualisation
ma_index.visualise_tree(visualise_by='name')

# In[ ]:


# If data is missing for any field, then assetId will be used instead
ma_index.visualise_tree(visualise_by='bbid')

# In[ ]:


ma_index.visualise_tree(visualise_by='id')

# ### Get  Underlier Weights and Attribution
# The underliers of an Index are the intermediate nodes in the composition tree.
# 
# You can fetch the weights and attribution of the underliers one level down using `get_underlier_weights` and `get_underlier_attribution` methods.

# In[ ]:


# Returns immediate underlier weights (one level down) as a pandas dataframe
ma_index.get_underlier_weights()

# In[ ]:


# Returns immediate underlier attribution (one level down) as a pandas dataframe
ma_index.get_underlier_attribution()

# # Fundamental Metrics of STS Index

# Single Stock STS Indices offer Fundamental Metrics data via API, which can be obtained using the `get_fundamentals` function.
# 
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
# 
# You may choose any combination of the following metrics:
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

# ### Get Fundamentals

# In[ ]:


# Returns fundamentals data between start and end date for the STS index
sstk_index.get_fundamentals(start=start_date, end=end_date)

# In[ ]:


# Returns fundamentals data between start and end date for the STS index for one year period with trailing direction and Price to Cash metric
sstk_index.get_fundamentals(start=start_date,
                       end=end_date,
                       period=DataMeasure.ONE_YEAR,
                       direction=DataMeasure.TRAILING,
                       metrics=[DataMeasure.PRICE_TO_CASH, DataMeasure.SALES_PER_SHARE])

# ### You're all set, Congrats!
# 
# *Have any other questions? Reach out to the [Marquee STS team](mailto:gs-marquee-sts-support@gs.com)!*
