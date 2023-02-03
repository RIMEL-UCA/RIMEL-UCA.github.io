#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.common import PayReceive, Currency
from gs_quant.instrument import IRSwaption
from gs_quant.markets.portfolio import Portfolio
from gs_quant.session import Environment, GsSession

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


swaption1 = IRSwaption(PayReceive.Pay, '5y', Currency.EUR, expiration_date='3m', name='EUR-3m5y')
swaption2 = IRSwaption(PayReceive.Pay, '8y', Currency.EUR, expiration_date='5w', name='EUR-5w8y')
portfolio = Portfolio((swaption1, swaption2))

# In[ ]:


price_result = portfolio.price()
print(price_result.aggregate())  # aggregate portfolio price

# In[ ]:


print(price_result['EUR-3m5y'])  # price of specific instrument
