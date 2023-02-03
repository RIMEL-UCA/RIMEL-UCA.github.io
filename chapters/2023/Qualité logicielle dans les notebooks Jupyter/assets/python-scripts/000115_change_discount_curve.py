#!/usr/bin/env python
# coding: utf-8

# In[7]:


from gs_quant.common import PayReceive, Currency
from gs_quant.instrument import IRSwaption
from gs_quant.session import Environment, GsSession
from gs_quant.markets import PricingContext
import datetime as dt

# In[2]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[3]:


swaption = IRSwaption(PayReceive.Receive, '5y', Currency.EUR, settlement='Cash.CollatCash')
swaption.resolve()

# In[30]:


# price using default OIS rate
with PricingContext(pricing_date=dt.date.today(), csa_term='EUR-OIS'):
    price = swaption.price()

print(price.result())

# In[26]:


# price using ESTR
with PricingContext(pricing_date=dt.date.today(), csa_term='EUR-EuroSTR'):
    price = swaption.price()

print(price.result())

# In[ ]:



