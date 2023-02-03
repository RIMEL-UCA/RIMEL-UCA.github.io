#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import date
from gs_quant.datetime.relative_date import RelativeDate
from gs_quant.session import GsSession, Environment

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterh
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('read_product_data',))

# In[ ]:


holiday_calendar = [date(2020, 12, 15)]
my_date = RelativeDate('-1b', base_date=date(2020, 12, 16)).apply_rule(holiday_calendar=holiday_calendar)
my_date
