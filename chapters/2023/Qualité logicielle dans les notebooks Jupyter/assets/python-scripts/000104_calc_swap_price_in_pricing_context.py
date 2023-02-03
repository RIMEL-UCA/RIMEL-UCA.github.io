#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.common import PayReceive, Currency
from gs_quant.instrument import IRSwap
from gs_quant.session import Environment, GsSession
from gs_quant.markets import PricingContext
from datetime import date

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


swap = IRSwap(PayReceive.Pay, '10y', notional_currency=Currency.GBP)

# In[ ]:


# price as of a specific date using nyc close
with PricingContext(pricing_date = date(2019, 1, 15), market_data_location='NYC'):
    pv_f = swap.price()

# In[ ]:


print(pv_f.result())
