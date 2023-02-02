#!/usr/bin/env python
# coding: utf-8

# In[3]:


from gs_quant.common import Currency
from gs_quant.instrument import IRXccySwap
from gs_quant.risk import IRDelta, IRXccyDelta, IRXccyDeltaParallel, IRXccyDeltaParallelLocalCurrency
from gs_quant.session import Environment, GsSession

# In[4]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[5]:


xswap = IRXccySwap(payer_currency=Currency.EUR, receiver_currency=Currency.USD,
                   effective_date='3m', termination_date='10y')

# In[8]:


delta = xswap.calc((IRDelta, IRXccyDelta)) 
parallel_delta = xswap.calc((IRXccyDeltaParallel, IRXccyDeltaParallelLocalCurrency))

# In[14]:


print(delta)  # all results for delta

# In[13]:


print(parallel_delta)  # all results for parallel delta

# In[12]:


print(parallel_delta[IRXccyDeltaParallel])  # single measure
