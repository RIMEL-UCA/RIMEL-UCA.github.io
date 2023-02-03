#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import date
from datetime import timedelta
from gs_quant.datetime.relative_date import RelativeDate
from gs_quant.markets import PricingContext
from gs_quant.session import GsSession, Environment

# In[3]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterh
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('read_product_data',))

# In[5]:


yesterday = date.today() - timedelta(days=1)
with PricingContext(pricing_date=yesterday):
    my_date = RelativeDate('-1d').apply_rule()  # Returns 2 days ago
my_date
