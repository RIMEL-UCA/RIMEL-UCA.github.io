#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime as dt

from gs_quant.data import Dataset
from gs_quant.session import Environment, GsSession

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('read_product_data',))

# In[ ]:


# fetch the data
weather_ds = Dataset(Dataset.GS.WEATHER)
panda_df = weather_ds.get_data(dt.date(2016, 1, 15), dt.date(2016, 1, 16), city=['Boston', 'Austin'])

# In[ ]:


panda_df.to_csv('my_data.csv')
