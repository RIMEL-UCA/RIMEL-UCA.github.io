#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime as dt
from gs_quant.markets.baskets import Basket
from gs_quant.session import Environment, GsSession

# In[ ]:


client = 'CLIENT ID'
secret = 'CLIENT SECRET'

GsSession.use(Environment.PROD, client_id=client, client_secret=secret, scopes=('read_product_data read_user_profile',))

# In[ ]:


basket = Basket.get('GSMBXXXX') # substitute input with any identifier for a basket

# In[ ]:


basket.get_corporate_actions(start=dt.date(2021, 1, 7), end=dt.date(2021, 3, 27))
