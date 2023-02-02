#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.entities.entitlements import User
from gs_quant.markets.baskets import Basket
from gs_quant.session import Environment, GsSession

# In[ ]:


client = 'CLIENT ID'
secret = 'CLIENT SECRET'

GsSession.use(Environment.PROD, client_id=client, client_secret=secret, scopes=('read_user_profile ',))

# In[ ]:


basket = Basket.get('GSMBXXXX') # substitute input with any identifier for a basket

# First, retrieve the user object by inputting the user email that you'd like to permission. Then, you may update the entitlements block corresponding to the level of permissioning you'd like to modify (e.g., *'entitlements.view'*, *'entitlements.rebalance'*, etc.)

# In[ ]:


user = User.get(email='user_email@example.com')

basket.entitlements.admin.users += [user] # update the entitlements block 'users' property

basket.update()
