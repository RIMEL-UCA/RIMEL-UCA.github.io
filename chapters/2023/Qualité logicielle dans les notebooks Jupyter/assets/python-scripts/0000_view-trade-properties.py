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


#View IRSwap instrument properties
IRSwap.properties()

# In[ ]:


#When you create an instance of IRSwap, properties can be set
my_swap = IRSwap(PayReceive.Pay, '5y', Currency.USD)

# In[ ]:


#View these property values by calling 'as_dict' on the swap instance
print(my_swap.as_dict())
