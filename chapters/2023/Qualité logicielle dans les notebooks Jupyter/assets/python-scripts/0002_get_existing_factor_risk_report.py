#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.markets.baskets import Basket
from gs_quant.session import Environment, GsSession

# In[ ]:


client = 'CLIENT ID'
secret = 'CLIENT SECRET'

GsSession.use(Environment.PROD, client_id=client, client_secret=secret)

# In[ ]:


basket = Basket.get('GSMBXXXX') # substitute input with any identifier for a basket

# In[ ]:


basket.get_factor_risk_report(risk_model_id='AXUS4M') # substitute risk model id for the one corresponding to the report you'd like to retrieve
