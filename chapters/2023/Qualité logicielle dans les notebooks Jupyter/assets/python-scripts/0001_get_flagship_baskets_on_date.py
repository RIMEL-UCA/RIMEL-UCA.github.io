#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime as dt

from gs_quant.markets.indices_utils import get_flagship_baskets
from gs_quant.session import Environment, GsSession

# In[ ]:


client = 'CLIENT ID'
secret = 'CLIENT SECRET'

GsSession.use(Environment.PROD, client_id=client, client_secret=secret, scopes=('read_product_data',))

# In[ ]:


get_flagship_baskets(as_of=dt.date(2021, 1, 7))
