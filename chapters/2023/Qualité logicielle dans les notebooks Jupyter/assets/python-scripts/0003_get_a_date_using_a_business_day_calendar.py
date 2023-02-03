#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.target.common import Currency
from gs_quant.datetime.relative_date import RelativeDate
from gs_quant.markets.securities import ExchangeCode
from gs_quant.session import GsSession, Environment

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterh
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('read_product_data',))

# In[ ]:


date_a = RelativeDate('-1b').apply_rule(exchanges=[ExchangeCode.NYSE])  # Returns previous business day base on NYSE calendar
date_b = RelativeDate('-1b').apply_rule(currencies=[Currency.BRL])  # Returns previous business day base on BRL Currency
print(date_a, date_b)
