#!/usr/bin/env python
# coding: utf-8

# In[5]:


from gs_quant.session import Environment, GsSession
from gs_quant.common import PayReceive, Currency
from gs_quant.instrument import IRSwap
from gs_quant.risk import Cashflows

# In[3]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[6]:


swap = IRSwap(PayReceive.Receive, '5y', Currency.EUR, fixed_rate='atm+10')

# In[12]:


# returns a dataframe of future cashflows 
# note this feature will be expanded to cover portfolios in future releases
swap.calc(Cashflows)
