#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.session import Environment, GsSession

# In[ ]:


# There are currently two ways to access data in gs_quant:
# 1 - gs_quant.data module - which provides programmatic access to The Marquee Data Catalogue, 
#       which includes a range of market data, gs product data, GIR Data and Execution & Liquidity Data. 
#       More info here: https://marquee.gs.com/s/discover/data-services
# 2 - Market Objects - which can be used to get both close and live market data
#
# The notebooks in this 01_market_objects directory focusing on accessing data via the Market Object
# 

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
# querying data from Market Objects requires the 'run_analytics' scope 
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))
