#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.session import Environment, GsSession
from gs_quant.instrument import FXForward
from gs_quant.markets.portfolio import Portfolio
from datetime import date
import pandas as pd

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


# get list of properties of an fx forward
FXForward.properties()

# In[ ]:


# in this example we will construct and price a portfolio of FXForwards
fx_fwds = Portfolio()

# In[ ]:


# pair is the forward's currency pair.  It a string of two ccy iso codes, optionally separated with a space (' ')
# The first currency is the base currency and the second is the quote currency

# In this case, base currency is 'EUR' and quote currency is 'GBP'. 
fx_fwds.append(FXForward(pair='EUR GBP', settlement_date='3y'))
fx_fwds.append(FXForward(pair='EURGBP', settlement_date='3y'))

# In[ ]:


# notional_amount is the quantity of the base currency to be exchanged in the future; it can be a string (eg: '100mm')
# or a double (10e8)

# For these two forwards - some amount of GBP will be exchanged for 100mm EUR at a future date
fx_fwds.append(FXForward(pair='EURGBP', notional_amount='100m', settlement_date='3y'))
fx_fwds.append(FXForward(pair='EURGBP', notional_amount=10e8, settlement_date='3y'))

# In[ ]:


# settlement_date is the contract settlement date. It can either be a date or string
fx_fwds.append(FXForward(settlement_date='3y'))
fx_fwds.append(FXForward(settlement_date=date(2023,9,25)))

# In[ ]:


# forward rate is the exchange rate which will be used on the settlement date.  It is a double.  
# If not set then it will resolve to the fair fx fwd rate

# In this example, the individual long the FXForward will pay 95mm GBP in exchange for 100mm EUR on the settlement date   
fx_fwds.append(FXForward(pair='EURGBP', notional_amount=10e8, forward_rate=.95, settlement_date='3y'))

# If not set then it will resolve to the fair fx fwd rate
fx_fwds.append(FXForward(pair='EURGBP', notional_amount=10e8, settlement_date='3y'))

# In[ ]:


pd.DataFrame(fx_fwds.price())
