#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.session import Environment, GsSession
from gs_quant.instrument import FXOption
from gs_quant.risk.measures import FXAnnualImpliedVol, FXAnnualATMImpliedVol, Price
from datetime import date
import pandas as pd

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


fx_option = FXOption(pair='USDJPY', option_type='Put', expiration_date='3m', premium=0)
result = fx_option.calc((Price, FXAnnualImpliedVol, FXAnnualATMImpliedVol)).to_frame()
result
