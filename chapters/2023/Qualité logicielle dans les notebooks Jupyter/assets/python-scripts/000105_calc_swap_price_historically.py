#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import date
from gs_quant.common import PayReceive, Currency
from gs_quant.instrument import IRSwap
from gs_quant.markets import HistoricalPricingContext
from gs_quant.session import Environment, GsSession

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


# create 2 swaps and fix swap_b parameters by resolving. Swap_a parameters will resolve in HistoricalPricingContext
swap_a = IRSwap(PayReceive.Pay, '30y', Currency.USD, fixed_rate='atm+5')
swap_b = IRSwap(PayReceive.Pay, '30y', Currency.USD, fixed_rate='atm+5')
swap_b.resolve()

# In[ ]:


with HistoricalPricingContext(date(2019,1,15), date.today()):
    res_a_f = swap_a.price()
    res_b_f = swap_b.price()  

# In[ ]:


res_a = res_a_f.result()
res_b = res_b_f.result()

# In[ ]:


import matplotlib.pyplot as plt

plt.plot(res_a, 'g-', label='Rolling 30y GBP Payer struck atm+5bps')
plt.plot(res_b, 'b-', label='30y GBP Payer struck atm+5bps over time')
plt.xlabel('time')
plt.ylabel('PV ($)')
plt.title('PV over Time')
plt.legend()
