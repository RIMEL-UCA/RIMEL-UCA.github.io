#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.common import PayReceive, Currency
from gs_quant.instrument import IRSwap
from gs_quant.session import Environment, GsSession
from gs_quant.risk import DollarPrice, IRDelta
from gs_quant.common import AggregationLevel
from gs_quant.markets import HistoricalPricingContext
from datetime import date

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id = None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


swap_10bps = IRSwap(PayReceive.Receive, '5y', Currency.EUR, fixed_rate='atm+10')

# In[ ]:


with HistoricalPricingContext(date(2020, 3, 2), date(2020, 4, 1), show_progress=True):
    res_f = swap_10bps.calc((DollarPrice, IRDelta, IRDelta(aggregation_level=AggregationLevel.Type, currency='local')))

# In[ ]:


print(res_f.result())  # retrieve all results

# In[ ]:


print(res_f[DollarPrice])  # retrieve historical prices
