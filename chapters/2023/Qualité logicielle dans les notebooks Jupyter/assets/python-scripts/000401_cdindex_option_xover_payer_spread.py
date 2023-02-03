#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import date
from gs_quant.common import PayReceive, Currency, BuySell
from gs_quant.instrument import CDIndexOption
from gs_quant.markets.portfolio import Portfolio
from gs_quant.session import Environment, GsSession
from gs_quant.markets import HistoricalPricingContext

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# Here we price a payer spread on the iTraxx Crossover OTR index
# 
# We buy a XOVER call and sell an offsetting XOVER call with a higher strike

# In[ ]:


# Buy 3m XO call k=300
long_leg = CDIndexOption(index_family='iTraxx Europe XOVER', strike=0.0300, option_type='Call', 
                         expiration_date='3m', termination_date='5y', buy_sell='Buy', name='Long_XOVER_Call')
# Sell 3m XO call k=350
short_leg = CDIndexOption(index_family='iTraxx Europe XOVER', strike=0.0350, option_type='Call', 
                          expiration_date='3m', termination_date='5y', buy_sell='Sell', name='Short_XOVER_Call')

# In[ ]:


payer_spread = Portfolio((long_leg, short_leg))

# In[ ]:


with HistoricalPricingContext(date(2021, 4, 1), date(2021, 4, 30), show_progress=True):
    prices = payer_spread.price()

# In[ ]:


prices.to_frame()

# In[ ]:


# Plot both legs of the payer spraed as well as the overall PV
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(prices.to_frame()['Long_XOVER_Call'], label='Buy leg, k=300')
ax.plot(prices.to_frame()['Short_XOVER_Call'], label='Sell leg, k=350')
ax.plot(prices.aggregate(), label='Overall cost')
ax.set_xlabel('time')
ax.set_ylabel('PV ($)')
ax.set_title('PV over Time')
ax.legend(bbox_to_anchor=(1, 0.5))
fig.autofmt_xdate()

# In[ ]:


prices.aggregate()
