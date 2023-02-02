#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import date

from gs_quant.datetime.relative_date import RelativeDate
from gs_quant.markets.securities import ExchangeCode
from gs_quant.session import GsSession, Environment

# In[1]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterh

GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('read_product_data',))

# ## Date Rule conventions
# A: first day of th year
# b: business days of calendar passed, USD by default
# d: Gregorian calendar days
# e: end of month (ignores number)
# m: month
# r: end of year
# u: business days ignoring USD holidays
# v: gets last business day of month (does not ignore number)
# x: gets last business day of month (ignores the number)
# y: add years, Result will be moved to the next week if falling on a weekend

# In[ ]:


# Returns four business days after the first business day on or after the 15th calendar day of the month
date: date = RelativeDate('14d+0u+4u').apply_rule(exchanges=[ExchangeCode.NYSE])
