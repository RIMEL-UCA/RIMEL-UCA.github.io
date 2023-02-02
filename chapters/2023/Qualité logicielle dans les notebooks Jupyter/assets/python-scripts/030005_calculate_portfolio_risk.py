#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.common import PayReceive, Currency  # import constants
from gs_quant.instrument import IRSwaption  # import instruments
from gs_quant.markets.portfolio import Portfolio
import gs_quant.risk as risk
from gs_quant.session import Environment, GsSession  # import sessions

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


swaption1 = IRSwaption(PayReceive.Pay, '5y', Currency.EUR, expiration_date='3m', name='EUR-3m5y')
swaption2 = IRSwaption(PayReceive.Pay, '5y', Currency.EUR, expiration_date='6m', name='EUR-6m5y')
portfolio = Portfolio((swaption1, swaption2))
result = portfolio.calc((risk.DollarPrice, risk.IRDelta))

# In[ ]:


# portfolio risk
print(result[risk.IRDelta].aggregate())

# In[ ]:


# instrument risk
print(result[risk.DollarPrice]['EUR-3m5y'])
print(result['EUR-3m5y'][risk.DollarPrice])
