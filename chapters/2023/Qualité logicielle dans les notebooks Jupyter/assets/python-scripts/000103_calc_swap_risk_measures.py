#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.common import PayReceive, Currency
from gs_quant.instrument import IRSwap
from gs_quant.session import Environment, GsSession
from gs_quant.risk import DollarPrice, IRDelta, IRDeltaParallel

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


swap = IRSwap(PayReceive.Pay, '10y', Currency.GBP)

# In[ ]:


result = swap.calc((DollarPrice, IRDeltaParallel)) 

# In[ ]:


print(result)  # all results

# In[ ]:


print(result[IRDeltaParallel])  # single measure
