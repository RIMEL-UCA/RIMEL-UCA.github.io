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

# First, retrieve the user object by inserting the app id that you'd like to permission. Then, you may update the entitlements block corresponding to the level of permissioning you'd like to modify (e.g., *'entitlements.admin'*, *'entitlements.edit'*, etc.)

# In[ ]:


application = User.get(user_id='application_id')

basket.entitlements.view.users += [application] # update the entitlements block 'users' property

basket.update()
