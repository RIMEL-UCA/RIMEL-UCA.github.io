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

# You may choose any combinations of the following metrics:
# 
# * **Dividend Yield:** DataMeasure.*DIVIDEND_YIELD*
# * **Earnings per Share:** DataMeasure.*EARNINGS_PER_SHARE*
# * **Earnings per Share Positive:** DataMeasure.*EARNINGS_PER_SHARE_POSITIVE*
# * **Net Debt to EBITDA:** DataMeasure.*NET_DEBT_TO_EBITDA*
# * **Price to Book:** DataMeasure.*PRICE_TO_BOOK*
# * **Price to Cash:** DataMeasure.*PRICE_TO_CASH*
# * **Price to Earnings:** DataMeasure.*PRICE_TO_EARNINGS*
# * **Price to Earnings Positive:** DataMeasure.*PRICE_TO_EARNINGS_POSITIVE*
# * **Price to Sales:** DataMeasure.*PRICE_TO_SALES*
# * **Return on Equity:** DataMeasure.*RETURN_ON_EQUITY*
# * **Sales per Share:** DataMeasure.*SALES_PER_SHARE*

# In[ ]:


basket.get_fundamentals(metrics=[DataMeasure.PRICE_TO_CASH, DataMeasure.SALES_PER_SHARE])
