#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.instrument import EqOption, OptionType, OptionStyle
from gs_quant.session import Environment, GsSession
import gs_quant.risk as risk

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


# create eq option
eq_option = EqOption('.STOXX50E', expiration_date='3m', strike_price='ATMS', option_type=OptionType.Call, 
                     option_style=OptionStyle.European)
eq_option.as_dict()

# In[ ]:


# single measure
dollar_delta = eq_option.calc(risk.EqDelta)
dollar_delta

# In[ ]:


# multiple measures
risks = eq_option.calc((risk.EqDelta, risk.EqGamma, risk.EqVega))
risks

# In[ ]:


# result for a single measure
risks[risk.EqDelta] 

# In[ ]:


# single measure with parameterized currency
euro_delta = eq_option.calc(risk.EqDelta(currency='EUR'))
euro_delta

# In[ ]:



