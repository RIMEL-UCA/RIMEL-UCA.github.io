#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.instrument import IRCMSSpreadOption
from gs_quant.markets import PricingContext
from gs_quant.markets.portfolio import Portfolio
from gs_quant.risk import IRAnnualImpliedVol, Price
import pandas as pd

pd.options.display.float_format = '{:,.4f}'.format

# In[ ]:


# instantiate session if not running in jupyter hub - external clients will have a client_id and secret.  Internal clients will use SSO
from gs_quant.session import GsSession
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics',)) 

# In[ ]:


# define a set of pairs and option expiries
pairs = [('5y','2y'), ('10y','2y'), ('30y','2y'), ('30y','5y'), ('30y','10y')]
expiries = ['1m', '2m', '3m', '6m', '9m', '1y', '18m', '2y', '3y', '4y', '5y', '7y', '10y', '12y', '15y', '20y']
portfolios = Portfolio([Portfolio([
    IRCMSSpreadOption(termination_date=e, notional_currency='EUR', notional_amount=10000, index1_tenor=p[0], index2_tenor=p[1], 
                      name='{}_{}{}'.format(e, p[0], p[1]))for e in expiries]) for p in pairs])

# price our list of portfolios
with PricingContext():
    result_p = portfolios.calc(Price)
    result_v = portfolios.calc(IRAnnualImpliedVol)

# In[ ]:


prices = result_p.to_frame() * 100
prices

# In[ ]:


vols = result_v.to_frame() * 10000
vols
