#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import date
from gs_quant.common import PayReceive, Currency
from gs_quant.instrument import CDIndex
from gs_quant.session import Environment, GsSession
from gs_quant.markets import HistoricalPricingContext, PricingContext
from gs_quant.risk import CDATMSpread

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics'))

# #### Available index families for CDIndex
# - iTraxx Europe OTR: 3y, 5y, 7y, 10y
# - iTraxx Europe XOVER OTR: 5y
# - CDX.NA.IG OTR: 5y
# - CDX.NA.HY OTR: 3y, 5y, 7y, 10y
# 

# In[ ]:


# Create an index product on S34 main
itraxx_main = CDIndex(index_family='iTraxx Europe', termination_date='5y', index_series=32)

with PricingContext(date(2021, 8, 10)):
    itraxx_main.resolve()
itraxx_main.as_dict()

# Also create an index product on iTraxx Xover and price historically alongside iTraxx Main.
# 
# Instead of pricing these, we will calculate the ATM spread

# In[ ]:


itraxx_xover = CDIndex(index_family='iTraxx Europe XOVER', index_series=32)

with HistoricalPricingContext(date(2020, 3, 9), date(2020, 3, 27), show_progress=True, is_batch=True):
    itraxx_main_res_f = itraxx_main.calc(CDATMSpread)
    itraxx_xover_res_f = itraxx_xover.calc(CDATMSpread)

# In[ ]:


itraxx_main_res_f.result()

# In[ ]:


itraxx_main_res = itraxx_main_res_f.result()
itraxx_xover_res = itraxx_xover_res_f.result()

# In[ ]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(itraxx_main_res, label="iTraxx Main")
ax.plot(itraxx_xover_res, label="iTraxx XOVER" )
ax.set_xlabel('time')
ax.set_ylabel('ATM Spread ($)')
ax.set_title('ATM Spread over Time')
ax.legend()
fig.autofmt_xdate()
