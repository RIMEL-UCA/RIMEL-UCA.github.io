#!/usr/bin/env python
# coding: utf-8

# Risk calculation results in gs_quant extend the ResultInfo Class, which contain helpful information about 
# the risk calculations themselves. 
# 
# While you don't need to use these datatypes directly, they contain information about the returned result that 
# might be useful to you. One of these datatypes is 'FloatWithInfo'.
# 

# In[ ]:


from gs_quant.session import Environment, GsSession
from gs_quant.common import PayReceive, Currency
from gs_quant.instrument import IRSwap

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


swap_5y = IRSwap(PayReceive.Receive, '5y', Currency.EUR, fixed_rate='atm+10')
swap_10y = IRSwap(PayReceive.Receive, '10y', Currency.EUR, fixed_rate='atm+10')

# In[ ]:


#When 'price' is called on an instrument, a 'FloatWithInfo' is returned
res_5y = swap_5y.price()
type(res_5y)

# In[ ]:


#This datatype has a few members - most simply, the raw value:
res_5y.raw_value

# In[ ]:


#The risk unit
res_5y.unit

# In[ ]:


# The risk_key, which helpfully provides information of what market, date, scenario and risk_measure were used
# to produce the results
res_5y.risk_key

# In[ ]:


# You can add two instances of FloatWithInfo together when units are consistent
res_10y = swap_10y.price()
res_total=res_5y+res_10y
type(res_total)

# In[ ]:


# You can also add FloatWithInfo and other types of objects, but units won't be checked\n",
# For example, adding a FloatWithInfo and float returns a float\n",
res_total=res_5y+res_10y.raw_value
type(res_total)

