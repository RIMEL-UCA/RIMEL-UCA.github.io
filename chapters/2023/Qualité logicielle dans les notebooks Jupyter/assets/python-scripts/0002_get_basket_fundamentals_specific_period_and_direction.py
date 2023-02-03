#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.data.fields import DataMeasure
from gs_quant.markets.baskets import Basket
from gs_quant.session import Environment, GsSession

# In[ ]:


client = 'CLIENT ID'
secret = 'CLIENT SECRET'

GsSession.use(Environment.PROD, client_id=client, client_secret=secret, scopes=('read_product_data read_user_profile',))

# In[ ]:


basket = Basket.get('GSMBXXXX') # substitute input with any identifier for a basket

# You may choose one of the following periods:
# 
# * **1 year:** DataMeasure.*ONE_YEAR*
# * **2 years:** DataMeasure.*TWO_YEARS*
# * **3 years:** DataMeasure.*THREE_YEARS*

# You may choose one of the following period directions:
# 
# * **Forward:** DataMeasure.*FORWARD*
# * **Trailing:** DataMeasure.*TRAILING*

# In[ ]:


basket.get_fundamentals(period=DataMeasure.TWO_YEARS, direction=DataMeasure.TRAILING)
