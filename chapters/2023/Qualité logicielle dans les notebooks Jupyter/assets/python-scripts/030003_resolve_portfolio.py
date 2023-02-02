#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.common import PayReceive, Currency  # import constants
from gs_quant.instrument import IRSwaption  # import instruments
from gs_quant.markets.portfolio import Portfolio
from gs_quant.session import Environment, GsSession  # import sessions

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


swaption1 = IRSwaption(PayReceive.Pay, '12m', Currency.EUR, expiration_date='3m', strike='atm+50', name='EUR-3m5y')
swaption2 = IRSwaption(PayReceive.Pay, '5y', Currency.EUR, expiration_date='6m', notional_amount=5e7, name='EUR-6m5y')

# In[ ]:


portfolio = Portfolio((swaption1, swaption2))
portfolio.resolve()
print(portfolio['EUR-3m5y'].as_dict())  # resolved instrument in the portfolio
