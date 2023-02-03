#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.common import PayReceive, Currency
from gs_quant.instrument import IRSwap
from gs_quant.session import Environment, GsSession

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# Some available properties are enums available in the common modules. 'Currency' and 'PayReceive' in this examples are enums. To see possible values in an IDE please use autocomplete (tab).

# In[ ]:


my_swap = IRSwap(PayReceive.Receive, '5y', Currency.GBP, effective_date='1m') 
print(list(PayReceive))
