#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import date
from gs_quant.instrument import EqOption, OptionType, OptionStyle
from gs_quant.markets import HistoricalPricingContext
from gs_quant.session import Environment, GsSession
import gs_quant.risk as risk

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


# create Option
eq_option = EqOption('.SPX', expiration_date='3m', strike_price='ATMS', option_type=OptionType.Call, 
                     option_style=OptionStyle.European)
eq_option.as_dict()

# In[ ]:


# price daily for dates between 20Jan20 and 27Jan20
with HistoricalPricingContext(date(2020, 1, 20), date(2020, 1, 27)):
    previous_dollar_deltas_holder = eq_option.calc(risk.EqDelta)

# In[ ]:


# get the results
previous_dollar_deltas = previous_dollar_deltas_holder.result()
print(previous_dollar_deltas)

# In[ ]:


# plot the instrument delta over time
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.plot(previous_dollar_deltas)
plt.xlabel('time')
plt.ylabel('$delta')
plt.title('$delta over time')
