#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import date
from gs_quant.instrument import EqOption, OptionType, OptionStyle
from gs_quant.markets import PricingContext
from gs_quant.session import Environment, GsSession
import gs_quant.risk as risk

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


# create Option
eq_option = EqOption('.STOXX50E', expiration_date='3m', strike_price='ATMS', option_type=OptionType.Call, 
                     option_style=OptionStyle.European)
eq_option.as_dict()

# In[ ]:


# price as of a specific date using LDN close
with PricingContext(pricing_date = date(2020, 2, 20), market_data_location='LDN'):
    previous_dollar_delta_holder = eq_option.calc(risk.EqDelta)

# In[ ]:


# get the result
previous_dollar_delta = previous_dollar_delta_holder.result()
print(previous_dollar_delta)
