#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.markets import PricingContext
from gs_quant.session import Environment, GsSession

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


# Each PricingContext has a Market Object
mkt = PricingContext.current.market

# In[ ]:


# By default, the PricingContext's market is a CloseMarket object
print(mkt.market_type)

# In[ ]:


# The CloseMarket Object is paramaterized by a location and date

# The default location is the PricingContext.current.market_data_location
print(mkt.location)

# The default date is the prior business date to the current pricing date
print(mkt.date)

# In[ ]:


# There are five types of Market Objects currently supported in gs_quant
from gs_quant.markets import CloseMarket, LiveMarket, TimestampedMarket, OverlayMarket, RelativeMarket

# In[ ]:


# The Live Market, which is paramaterized by a location
mkt = LiveMarket()
print(mkt.location)

#You can set the location by using the PricingLocation Enum or a string
from gs_quant.common import PricingLocation
mkt.location = PricingLocation.HKG
print(mkt.location)

# In[ ]:


# The TimestampedMarket has a required timestamp argument and optional location
import time as t
mkt = TimestampedMarket(t.time(), location=PricingLocation.TKO)
