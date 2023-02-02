#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import date
from gs_quant.markets.position_set import Position, PositionSet
from gs_quant.session import Environment, GsSession

# In[ ]:


client = 'CLIENT ID'
secret = 'CLIENT SECRET'

GsSession.use(Environment.PROD, client_id=client, client_secret=secret, scopes=('read_product_data',))

# In[ ]:


position_set = PositionSet([Position(identifier='MSFT UW', quantity=200)])

# In[ ]:


position_set.positions += Position(identifier='AAPL UW', weight=800) # can alternatively specify 'quantity' instead of 'weight'
