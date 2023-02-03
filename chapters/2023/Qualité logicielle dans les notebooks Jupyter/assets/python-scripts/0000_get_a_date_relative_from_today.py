#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.datetime.relative_date import RelativeDate
from gs_quant.session import GsSession, Environment

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterh
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('read_product_data',))

# In[ ]:


my_date = RelativeDate('-1d').apply_rule()  # Returns previous day
my_date
