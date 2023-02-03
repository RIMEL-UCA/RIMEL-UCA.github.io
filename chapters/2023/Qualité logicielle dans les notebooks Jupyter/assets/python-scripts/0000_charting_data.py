#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt

from gs_quant.data import Dataset
from gs_quant.markets import PricingContext
from gs_quant.session import GsSession, Environment

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('read_product_data',))

# In[ ]:


market_date = PricingContext.current.market.date  # Determine current market date

vol_dataset = Dataset('EDRVOL_PERCENT_STANDARD')  # Initialize the equity implied volatility dataset
vol_data = vol_dataset.get_data(market_date, market_date, ticker='SPX', tenor='1m', strikeReference='forward')

strikes = vol_data['relativeStrike']
vols = vol_data['impliedVolatility'] * 100

# In[ ]:


plt.plot(strikes, vols, label='Implied Volatility by Strike')
plt.xlabel('Relative Strike')
plt.ylabel('Implied Volatility')
plt.title('Implied Volatility by Strike')

plt.show()
