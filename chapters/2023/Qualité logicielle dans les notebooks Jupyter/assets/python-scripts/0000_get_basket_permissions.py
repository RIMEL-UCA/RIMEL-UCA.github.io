#!/usr/bin/env python
# coding: utf-8

# ### What are the four levels of basket permissions?
# 
# 1. **Admin:** Basket admins can view all basket information, edit metadata, submit requests to update basket composition, and permission other users to the basket
# 2. **Edit:** Editors may view basket data and edit details such as name, description, etc.
# 3. **Rebalance:** Rebalance permissions enable a user to view basket data and submit and approve rebalance submissions
# 4. **View:** Viewers are able to see most basket information, but are not able to modify the basket in any way

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


basket.entitlements.to_frame()
