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


basket.poll_report(report_id='R1234567890', timeout=300, step=20) # timeout/step are optional - default behavior will be to check status every 30 sec for <= 10 min
