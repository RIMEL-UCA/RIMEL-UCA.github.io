#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import date, timedelta
from gs_quant.datetime.relative_date import RelativeDate
from gs_quant.session import GsSession, Environment

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterh
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('read_product_data',))

# In[ ]:


yesterday = date.today() - timedelta(days=1)
my_date = RelativeDate('-1d', base_date=yesterday).apply_rule()  # Returns 2 days ago
my_date
