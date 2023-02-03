#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

from datetime import date
from gs_quant.markets.position_set import PositionSet
from gs_quant.session import Environment, GsSession

# In[ ]:


client = 'CLIENT ID'
secret = 'CLIENT SECRET'

GsSession.use(Environment.PROD, client_id=client, client_secret=secret, scopes=('read_product_data',))

# In[ ]:


positions_df = pd.read_csv('positions_data.csv') 

# In[ ]:


position_set = PositionSet.from_frame(positions_df)
