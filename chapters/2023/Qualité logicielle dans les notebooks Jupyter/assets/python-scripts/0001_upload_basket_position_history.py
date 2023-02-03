#!/usr/bin/env python
# coding: utf-8

# **Note:** The \"default_backcast\" parameter must be set to false *during basket creation* in order to upload custom position history after the basket has been published to Marquee.

# In[ ]:


from datetime import date
from gs_quant.markets.baskets import Basket
from gs_quant.markets.position_set import Position, PositionSet
from gs_quant.session import Environment, GsSession

# In[ ]:


client = 'CLIENT ID'
secret = 'CLIENT SECRET'

GsSession.use(Environment.PROD, client_id=client, client_secret=secret, scopes=('read_product_data read_user_profile',))

# In[ ]:


basket = Basket.get('GSMBXXXX')

# "Below is a simplied example on how to create position sets. You may refer to the position set examples in this directory for more options on how to upload larger/more complicated position sets."

# In[ ]:


positions = [Position(identifier='AAPL UW', weight=0.5), Position(identifier='MSFT UW', weight=0.5)]

pos_set_1 = PositionSet(positions, date(2019, 6, 3))
pos_set_2 = PositionSet(positions, date(2020, 1, 2))
pos_set_3 = PositionSet(positions, date(2020, 6, 1))
pos_set_4 = PositionSet(positions, date(2021, 1, 4))

# In[ ]:


basket.upload_position_history([pos_set_1, pos_set_2, pos_set_3, pos_set_4])
