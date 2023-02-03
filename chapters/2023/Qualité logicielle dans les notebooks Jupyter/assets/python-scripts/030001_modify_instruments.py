#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime as dt
from gs_quant.common import PayReceive, Currency 
from gs_quant.instrument import IRSwaption 
from gs_quant.markets.portfolio import Portfolio
from gs_quant.session import Environment, GsSession

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


swaption1 = IRSwaption(PayReceive.Pay, '5y', Currency.EUR, expiration_date='3m', name='EUR-3m5y Payer')
swaption2 = IRSwaption(PayReceive.Pay, '7y', Currency.EUR, expiration_date='9m', name='EUR-9m7y Payer')
swaption_midcurve = IRSwaption(PayReceive.Receive, '10y', Currency.EUR, effective_date='10y', expiration_date='6m', name='EUR-6m10y10y Receiver')

# In[ ]:


portfolio = Portfolio((swaption1, swaption2))
print(portfolio.instruments)

# In[ ]:


portfolio.pricables = (swaption1, swaption2, swaption_midcurve)
print(portfolio.instruments)
