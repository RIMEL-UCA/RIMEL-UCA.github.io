#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.session import Environment, GsSession
from gs_quant.instrument import IRSwaption
from gs_quant.markets.portfolio import Portfolio
from gs_quant.markets import PricingContext
from gs_quant.risk import RollFwd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


# basic usage of RollFwd scenario
eur1y10y = IRSwaption('Pay', '10y', 'EUR', expiration_date='1y')

# care needs to be taken when creating relative trades like the one above.  
# If you don't resolve the trade, the resolution of the trade parameters will be done with 
# reference to the active pricing context.  Under the RollFwd scenario this means that
# if you don't resolve the trade will be a different trade when priced under the rollfwd scenario.
eur1y10y.resolve()

# Roll forward 1 month
rollfwd_scenario = RollFwd(date='1m',holiday_calendar='LDN')
with rollfwd_scenario:
    fwd_price = eur1y10y.price()

print('Base price:     {:,.2f}'.format(eur1y10y.price()))
print('Scenario price: {:,.2f}'.format(fwd_price))

# In[ ]:


# show how the option value will roll down moving forward 66 business days assuming either fwds 
# or spot rates are realised.

short_swaption = IRSwaption('Pay', '5y', 'USD', expirationDate='6m', notionalAmount=1e8)
short_swaption.resolve()

prices = []
roll_spot_prices = []
with PricingContext():
    for bus_days in range(66):
        with RollFwd(date=f'{bus_days}b', holiday_calendar= 'NYC', realise_fwd=True):
            prices.append(short_swaption.price())
        with RollFwd(date=f'{bus_days}b', holiday_calendar= 'NYC', realise_fwd=False):
            roll_spot_prices.append(short_swaption.price())

pd.Series([p.result() for p in prices], dtype=np.dtype(float)).plot(figsize=(10, 6),
                                                                    title="Swaption Price Forward in Time", 
                                                                    label='roll to fwd')
pd.Series([rp.result() for rp in roll_spot_prices], dtype=np.dtype(float)).plot(figsize=(10, 6), 
                                                       label='roll to spot')
plt.xlabel('Business Days from Pricing Date')
plt.ylabel('PV')

# In[ ]:


# create a grid of expiry by tenor swaptions showing the pv under the rollfwd scenario minus the base pv.
def calc_risk_matrix(ccy, strike, pay_rec, date, roll_to_fwds, expiries, tenors):
    portfolio = Portfolio([IRSwaption(pay_rec, tenor, ccy,
                                      expiration_date=expiry, strike=strike, name='{}_{}'.format(expiry, tenor)) 
                          for expiry in expiries for tenor in tenors])
    portfolio.resolve()
    with RollFwd(date=date, holiday_calendar='LDN', realise_fwd=roll_to_fwds):
        rollfwd_results = portfolio.price()
    
    base_results = portfolio.price()
    
    rollfwd_records = [(rollfwd_results[t.name]-base_results[t.name], t.name.split('_')[0], t.name.split('_')[1]) for t in portfolio]
    rollfwd_df = pd.DataFrame(rollfwd_records, columns=['Value', 'Expiry', 'Tenor'])
    
    pivot_df = rollfwd_df.pivot_table(values='Value', index='Expiry', columns='Tenor')
    return pivot_df[tenors].reindex(expiries)

# In[ ]:


ccy = 'EUR'
strike = 'ATM'
pay_rec = 'Pay'  # or 'Receive' or 'Straddle'
date = '1m'  # enter relative or actual date
roll_to_fwds = True
expiries = ['1m', '2m', '3m', '6m', '9m', '1y', '18m', '2y', '3y', '5y', '7y', '10y']
tenors = ['1y', '2y', '3y', '5y', '7y', '10y', '15y', '20y', '25y', '30y']

calc_risk_matrix(ccy, strike, pay_rec, date, roll_to_fwds, expiries, tenors)

# In[ ]:



