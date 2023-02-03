#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.common import PayReceive, Currency
from gs_quant.instrument import IRSwap
from gs_quant.session import Environment, GsSession

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


my_swap = IRSwap(PayReceive.Pay, '7y', Currency.EUR)

# In[ ]:


my_swap.termination_date = '10y'
print(my_swap.termination_date)
