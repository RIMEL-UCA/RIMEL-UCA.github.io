#!/usr/bin/env python
# coding: utf-8

# In[2]:


from gs_quant.common import  Currency
from gs_quant.instrument import IRXccySwap
from gs_quant.session import Environment, GsSession

# In[3]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[12]:


xswap = IRXccySwap(payer_currency=Currency.EUR, receiver_currency=Currency.USD,
                   effective_date='3m', termination_date='10y')

# In[14]:


print(xswap.price())
