#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.markets.baskets import Basket
from gs_quant.markets.indices_utils import ReturnType
from gs_quant.session import Environment, GsSession

# In[ ]:


client = 'CLIENT ID'
secret = 'CLIENT SECRET'

GsSession.use(Environment.PROD, client_id=client, client_secret=secret, scopes=('read_product_data read_user_profile',))

# In[ ]:


parent_basket = Basket.get('GSMBXXXX')
new_basket = parent_basket.clone()

# If you'd like, you may now create a new basket in Marquee using this composition. See the Basket Create Tutorial for a more nuanced example/explanation on how to do this (you may skip step 3 in this case).

# In[ ]:


new_basket.ticker = 'GSMBCLNE'
new_basket.name = 'Clone of GSMBXXXX'
new_basket.currency = 'USD'
new_basket.return_type = ReturnType.PRICE_RETURN

new_basket.create()
