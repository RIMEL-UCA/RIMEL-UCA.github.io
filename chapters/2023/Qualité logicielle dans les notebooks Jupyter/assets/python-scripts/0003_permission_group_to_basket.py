#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.entities.entitlements import Group
from gs_quant.markets.baskets import Basket
from gs_quant.session import Environment, GsSession

# In[ ]:


client = 'CLIENT ID'
secret = 'CLIENT SECRET'

GsSession.use(Environment.PROD, client_id=client, client_secret=secret, scopes=('read_user_profile modify_product_data',))

# In[ ]:


basket = Basket.get('GSMBXXXX') # substitute input with any identifier for a basket

# First, retrieve the group object by inputting the group id that you'd like to permission. Then, you may update the entitlements block corresponding to the level of permissioning you'd like to modify (e.g., *'entitlements.view'*, *'entitlements.rebalance'*, etc.)

# In[ ]:


group = Group.get(group_id='example.group')

basket.entitlements.admin.groups += [group] # update the entitlements block 'groups' property

basket.update()
