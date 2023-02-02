#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.common import PayReceive, Currency
from gs_quant.instrument import IRSwap
from gs_quant.session import Environment, GsSession

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# * Calling the `resolve()` method will resolve any relative parameters to absolute values and fill in unspecified parameters.
# * `resolve()` will change the state of the instrument object in place by default unless argument `in_place` is `False`.
# * `resolve` can be used to "fix" the instrument parameters when evaluating pricing & risk.

# In[ ]:


my_swap = IRSwap(PayReceive.Pay, '13m', fixed_rate='atm+30')  

# In[ ]:


# note fixed_rate and termination_date are resolved to a float and date, respectively
print(my_swap.resolve(in_place=False).as_dict())
