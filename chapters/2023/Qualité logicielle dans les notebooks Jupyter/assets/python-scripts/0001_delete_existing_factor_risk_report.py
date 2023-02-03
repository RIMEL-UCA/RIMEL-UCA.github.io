#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.markets.baskets import Basket
from gs_quant.session import Environment, GsSession

# In[ ]:


client = 'CLIENT ID'
secret = 'CLIENT SECRET'

GsSession.use(Environment.PROD, client_id=client, client_secret=secret, scopes=('read_user_profile',))

# In[ ]:


basket = Basket.get('GSMBXXXX') # substitute input with any identifier for a basket

# In[ ]:


basket.delete_factor_risk_report(risk_model_id='AXUS4M') # substitute risk model id for the one corresponding to the report you'd like to delete
