#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.entities.entitlements import User
from gs_quant.markets.baskets import Basket
from gs_quant.session import Environment, GsSession

# In[ ]:


client = 'CLIENT ID'
secret = 'CLIENT SECRET'

GsSession.use(Environment.PROD, client_id=client, client_secret=secret, scopes=('read_user_profile modify_product_data',))

# In[ ]:


basket = Basket.get('GSMBXXXX') # substitute input with any identifier for a basket

# First, retrieve the user object by inputting the user id that you'd like to unpermission (you may alternatively retrieve a user by email, or can retrieve a group instead. Refer to the other examples in this folder for examples of this). Then, you may update the entitlements block corresponding to the level of permissioning you'd like to modify (e.g., *'entitlements.view'*, *'entitlements.rebalance'*, etc.)

# In[ ]:


user = User.get(user_id='user_id')

basket.entitlements.admin.users.remove(user) # update the entitlements block 'groups' property

basket.update()
