#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime as dt

from gs_quant.markets.indices_utils import get_flagships_performance
from gs_quant.session import Environment, GsSession

# In[ ]:


client = 'CLIENT ID'
secret = 'CLIENT SECRET'

GsSession.use(Environment.PROD, client_id=client, client_secret=secret, scopes=('read_product_data',))

# In[ ]:


get_flagships_performance(start=dt.date(2021, 1, 7), end=dt.date(2021, 3, 27))
