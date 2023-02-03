#!/usr/bin/env python
# coding: utf-8

# # Editing a Custom Basket
# 
# Editing a custom basket is very similar to the creation process. You simply fetch your basket, set its properties to match your preferences, and call update! Please note that not all settings are modifiable after creation (see the list of options below). Additionally, if you'd like to adjust your basket's composition or price settings, please refer to the [rebalance tutorial](./Basket%20Rebalance.ipynb) instead.

# ## Step 1: Authenticate & Initialize your session
# 
# First you will import the necessary modules and add your client id and client secret.

# In[ ]:


from gs_quant.markets.baskets import Basket
from gs_quant.session import Environment, GsSession

client = 'CLIENT ID'
secret = 'CLIENT SECRET'

GsSession.use(Environment.PROD, client_id=client, client_secret=secret, scopes=('read_product_data read_user_profile modify_product_data',))

# ## Step 2: Fetch your basket and set your changes
# 
# Next you will fetch the basket you'd like to edit by passing in any of its identifiers such as BloombergId, Cusip, Ticker, etc. If this is a basket you or someone from your organization created, please make sure your application has edit/admin entitlements or you will not be able to modify it. You will then be ready to change any of the properties listed below.
# 
# | Parameter Name      |Description |
# |:--------------------|:-----------|
# |name                 |Display name of the basket|
# |description          |Free text description of basket. Description provided will be indexed in the search service for free text relevance match.|
# |publish_to_bloomberg |If you'd like us to publish your basket to Bloomberg|
# |publish_to_reuters   |If you'd like us to publish your basket to Reuters  |
# |publish_to_factset   |If you'd like us to publish your basket to Factset  |
# |include_price_history|Republish price history based on current composition when publishing to Bloomberg|

# In[2]:


basket = Basket.get('GSMBXXXX')

basket.description = 'My New and Improved basket description'
basket.publish_to_reuters = True

# ## Step 3: Update your basket's entitlements
# 
# The application you use to create your basket will initially be the only one permissioned to view, edit, and submit rebalance requests. If you'd like to entitle other users or groups with view or admin access, you may update your basket's permissions at any time.
# 
# In order to add or remove permissions for a specific user, you will need either their Marquee user id or email. You may also permission groups using their group id. See the snippet below, or refer to the [baskets permissions examples](../examples/07_basket_permissions/0001_permission_application_to_basket.ipynb) for more options.

# In[ ]:


from gs_quant.entities.entitlements import Group, User

user = User.get(user_id='application_id')
basket.entitlements.view.users.remove(user)

group = Group.get(group_id='group_id')
basket.entitlements.admin.groups += [group]

basket.entitlements.to_frame() # call to verify the entitlement changes are now reflected properly

# ## Step 4: Submit your changes
# 
# Once you've ensured that your basket has been updated to your satisfaction, you're ready to officially submit these changes to Marquee! Once you call update on the basket, you may poll its status to make sure that it has processed successfully. This will check the report status every 30 seconds for 10 minutes by default, but you can override this option if you prefer as shown below.

# In[ ]:


my_basket.get_details() # call to make sure that your changes are all reflected properly

my_basket.update()

my_basket.poll_status(timeout=120, step=20) # optional: constantly checks edit status until report succeeds, fails, or the poll times out (this example checks every 20 seconds for 2 minutes)
