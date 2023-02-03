#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import date
from gs_quant.instrument import EqOption, OptionType, OptionStyle
from gs_quant.markets import HistoricalPricingContext
from gs_quant.session import Environment, GsSession

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


# create Option
eq_option = EqOption('.STOXX50E', expiration_date='3m', strike_price='ATMS', option_type=OptionType.Call, 
                     option_style=OptionStyle.European)
eq_option.as_dict()

# In[ ]:


# price daily for dates between 14Jan20 and 21Jan20
with HistoricalPricingContext(date(2020, 1, 14), date(2020, 1, 21)):
    previous_prices_holder = eq_option.price()

# In[ ]:


# get the results
previous_prices = previous_prices_holder.result()
print(previous_prices)

# In[ ]:


# plot the instrument price over time
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.plot(previous_prices)
plt.xlabel('time')
plt.ylabel('price')
plt.title('Price over time')
