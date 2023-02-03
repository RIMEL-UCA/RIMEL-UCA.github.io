#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import date
from gs_quant.common import PayReceive, Currency
from gs_quant.instrument import CDIndexOption
from gs_quant.risk import DollarPrice, CDDelta, CDVega, CDTheta, CDGamma
from gs_quant.session import Environment, GsSession
from gs_quant.markets import PricingContext, HistoricalPricingContext

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics'))

# In[ ]:


cd_option = CDIndexOption(index_family='iTraxx Europe XOVER', strike=0.0300, option_type='Call', 
                          buy_sell='Sell', expiration_date='1m', termination_date='5y', notional_amount=1)
cd_option.as_dict()

# Index Option specific risk measures are:
# - CDDelta - amount of underlying index in local notional to buy in order to hedge the change in Dollar Price due to a 1bp shift in the underlying index spread
# - CDGamma - amount of underlying index in local notional to buy in order to hedge the change in option delta due to a 1bp shift in the underlying index spread
# - CDVega - change in option Dollar Price due to a 1bp shift in the implied volatility of the underlying index 
# - CDTheta - change in option Dollar Price over one day, assuming constant vol

# In[ ]:


with PricingContext(market_data_location='LDN'):
    greeks = cd_option.calc((CDDelta, CDGamma, CDVega, CDTheta))

# In[ ]:


greeks.result()

# View historic delta:

# In[ ]:


with HistoricalPricingContext(date(2021, 5, 5), date(2021, 5, 11), show_progress=True, is_batch=True):
    previous_option_deltas_holder = cd_option.calc(CDDelta)

# In[ ]:


previous_option_deltas = previous_option_deltas_holder.result()
print(previous_option_deltas.value)

# In[ ]:


# plot the instrument delta over time
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

fig, ax = plt.subplots()
ax.plot(previous_option_deltas.value)
ax.set_xlabel('time')
ax.set_ylabel('delta')
ax.set_title('delta over time')
fig.autofmt_xdate()
