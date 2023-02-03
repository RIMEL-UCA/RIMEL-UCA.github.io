#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import date
from gs_quant.common import PayReceive, Currency
from gs_quant.instrument import CDIndex
from gs_quant.risk import CDATMSpread
from gs_quant.session import Environment, GsSession
from gs_quant.markets import PricingContext
from gs_quant.markets.portfolio import Portfolio

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# Rolling a position on XOVER from S34 to S35

# In[ ]:


xover_34 = CDIndex(index_family='iTraxx Europe XOVER', index_series=34, buy_sell='Sell', termination_date='5y', name='sell34')
xover_35 = CDIndex(index_family='iTraxx Europe XOVER', index_series=35, buy_sell='Buy', termination_date='5y', name='buy35')

# In[ ]:


xover_roll = Portfolio((xover_34, xover_35))

# In[ ]:


with PricingContext(market_data_location='LDN'):
    roll_spreads = xover_roll.calc(CDATMSpread)
    roll_price = xover_roll.price()

# In[ ]:


df = roll_spreads.to_frame()

# In[ ]:


df.loc['buy35'] - df.loc['sell34']

# In[ ]:


roll_price.aggregate()
