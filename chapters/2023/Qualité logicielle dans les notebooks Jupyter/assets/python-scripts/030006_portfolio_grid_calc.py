#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.markets.portfolio import Portfolio
from gs_quant.instrument import IRSwaption
from gs_quant.risk import IRAnnualImpliedVol
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# In[ ]:


from gs_quant.session import GsSession
# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


tails = ['1y', '3y', '5y', '10y', '15y', '20y', '25y', '30y']
expiries = ['3m', '6m', '9m', '1y', '18m', '2y', '3y', '4y', '5y']
pay_rec = 'Pay'
ccy = 'EUR'
moneyness = 25

# In[ ]:


portfolios = Portfolio([Portfolio([IRSwaption(pay_or_receive=pay_rec, notional_currency=ccy, termination_date=t, 
                                              expiration_date=e, strike='ATM+{}'.format(moneyness), name=e) for e in expiries], name=t) for t in tails])
results = portfolios.calc(IRAnnualImpliedVol)

# In[ ]:


frame = results.to_frame('value','portfolio_name_0','instrument_name') * 10000
frame

# In[ ]:


plt.subplots(figsize=(12, 8))
ax = sns.heatmap(frame, annot=True, fmt='.2f', cmap='coolwarm')
ax.set(ylabel='Expiries', xlabel='Tails', title='Implied Vol Grid')
ax.xaxis.tick_top()
