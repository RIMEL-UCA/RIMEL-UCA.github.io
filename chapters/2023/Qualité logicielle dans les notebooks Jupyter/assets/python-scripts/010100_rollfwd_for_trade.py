#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime as dt
from gs_quant.common import PayReceive, Currency
from gs_quant.instrument import IRSwaption
from gs_quant.risk import RollFwd
from gs_quant.session import GsSession, Environment

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


swaption = IRSwaption(PayReceive.Receive, '5y', Currency.EUR, expirationDate=dt.date(2029, 10, 8), strike='atm')
base_price = swaption.price()

# In[ ]:


swaption.resolve()  # fix expiry and maturity

# RollFwd Scenario - Roll forward 1 month
with RollFwd(date='1m', holiday_calendar='NYC'):
    fwd_price = swaption.price()

print('Base price:     {:,.2f}'.format(base_price))
print('Scenario price: {:,.2f}'.format(fwd_price))
print('Diff:           {:,.2f}'.format(fwd_price - base_price))
