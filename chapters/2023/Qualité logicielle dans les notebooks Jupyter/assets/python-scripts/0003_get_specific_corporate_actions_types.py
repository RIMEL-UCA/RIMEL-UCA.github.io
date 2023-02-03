#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.markets.baskets import Basket
from gs_quant.markets.indices_utils import CorporateActionType
from gs_quant.session import Environment, GsSession

# In[ ]:


client = 'CLIENT ID'
secret = 'CLIENT SECRET'

GsSession.use(Environment.PROD, client_id=client, client_secret=secret, scopes=('read_product_data read_user_profile',))

# In[ ]:


basket = Basket.get('GSMBXXXX') # substitute input with any identifier for a basket

# You may choose any combination of the following corporate action types:
# 
# * **Aquisition:** CorporateActionType.*ACQUISITION*
# * **Cash Dividend:** CorporateActionType.*CASH_DIVIDEND*
# * **Identifier Change:** CorporateActionType.*IDENTIFIER_CHANGE*
# * **Rights Issue:** CorporateActionType.*RIGHTS_ISSUE*
# * **Share Change:** CorporateActionType.*SHARE_CHANGE*
# * **Special Dividend:** CorporateActionType.*SPECIAL_DIVIDEND*
# * **Spin Off:** CorporateActionType.*SPIN_OFF*
# * **Stock Dividend:** CorporateActionType.*STOCK_DIVIDEND*
# * **Stock Split:** CorporateActionType.*STOCK_SPLIT*

# In[ ]:


basket.get_corporate_actions(ca_type=[CorporateActionType.IDENTIFIER_CHANGE, CorporateActionType.RIGHTS_ISSUE])
