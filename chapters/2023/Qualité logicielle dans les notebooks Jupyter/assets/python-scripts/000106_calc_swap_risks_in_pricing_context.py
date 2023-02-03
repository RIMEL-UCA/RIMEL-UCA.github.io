#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.common import PayReceive, Currency
from gs_quant.instrument import IRSwap
from gs_quant.session import Environment, GsSession
from gs_quant.risk import DollarPrice, IRDelta, IRDeltaParallel
from gs_quant.markets import PricingContext
from datetime import date

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


swap = IRSwap(PayReceive.Pay, '12y', notional_currency=Currency.GBP)

# In[ ]:


with PricingContext(pricing_date=date(2019, 1, 15)):
    res_f = swap.calc((DollarPrice, IRDelta, IRDeltaParallel))

# In[ ]:


print(res_f.result())  # retrieve all measures

# In[ ]:


print(res_f[DollarPrice])  # retrieve specific measure
