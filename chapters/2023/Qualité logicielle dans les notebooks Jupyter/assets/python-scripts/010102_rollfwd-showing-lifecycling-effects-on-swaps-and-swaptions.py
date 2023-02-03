#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.session import GsSession, Environment
from gs_quant.risk import RollFwd
from gs_quant.markets.portfolio import Portfolio
from gs_quant.instrument import IRSwap, IRSwaption
from gs_quant.markets import PricingContext
import matplotlib.pyplot as plt
import gs_quant.risk as risk
import pandas as pd
import numpy as np

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


# create a swap which has a 1m floating frequency
swap = IRSwap('Pay', '10y', 'EUR', fixed_rate='ATM-5', floating_rate_frequency='1m', name='EUR10y')

# resolve the trade as of today to fix the dates and rate
swap.resolve()

# roll daily for 66 business days assuming both forward curve is realised and spot curve is realised
fwd_price = []
fwd_cash = []
spot_price = []
spot_cash = []
r = range(0, 66, 6)
# by wrapping all the scenarios into one PricingContext we package all the requests into one call to GS
with PricingContext():
    for bus_days in r:
        with PricingContext(is_async=True), RollFwd(date=f'{bus_days}b', holiday_calendar='LDN', realise_fwd=True):
            fwd_price.append(swap.price())
            fwd_cash.append(swap.calc(risk.Cashflows))
        with PricingContext(is_async=True), RollFwd(date=f'{bus_days}b', holiday_calendar='LDN', realise_fwd=False):
            spot_price.append(swap.price())
            spot_cash.append(swap.calc(risk.Cashflows))

# In[ ]:


fwd_pv  = pd.Series([p.result() for p in fwd_price], index=r)
spot_pv   = pd.Series([p.result() for p in spot_price], index=r)

# The output of the cashflows measure is a dataframe of the past and implied future cashflows. We could filter by payment date
# but conviently the discount factor is 0 for paid cashflows
cash_fwd = pd.Series([c.result()[c.result().discount_factor == 0].payment_amount.sum() for c in fwd_cash], index=r)
cash_spot = pd.Series([c.result()[c.result().discount_factor == 0].payment_amount.sum() for c in spot_cash], index=r)

fwd_pv.plot(figsize=(10, 6), title='Swap Carry', label='{} Realise Fwd'.format(swap.name))
spot_pv.plot(label='{} Realise Spot'.format(swap.name))
(fwd_pv+cash_fwd).plot(label='{} Realise Fwd (inc. cash)'.format(swap.name))
(spot_pv+cash_spot).plot(label='{} Realise Spot (inc. cash)'.format(swap.name))

plt.xlabel('Business Days from Pricing Date')
plt.ylabel('PV')
plt.legend()
plt.show()
# note that the steps represent the move in MTM as the cashflows are paid.  The libor fixing is implied from the fwd

# In[ ]:


itm_swaption = IRSwaption('Receive', '10y', 'EUR', strike='ATM+20', expiration_date='1m', name='ITM swaption')
otm_swaption = IRSwaption('Receive', '10y', 'EUR', strike='ATM-20', expiration_date='1m', name='OTM swaption')
port = Portfolio([itm_swaption, otm_swaption])
port.resolve()

# roll daily for 44 business days assuming both forward curve is realised and spot curve is realised
fwd_results = []
spot_results = []
r = range(0, 44, 4)
# by wrapping all the scenarios into one PricingContext we package all the requests into one call to GS
with PricingContext():
    for bus_days in r:
        with PricingContext(is_async=True), RollFwd(date=f'{bus_days}b', holiday_calendar='LDN', realise_fwd=True):
            fwd_results.append(port.price())

# In[ ]:


df = pd.DataFrame.from_records([[p['ITM swaption'], p['OTM swaption']] for p in fwd_results], index=r, columns=['ITM', 'OTM'])
df.plot(figsize=(10,6), secondary_y=['OTM'], title='Swaption Carry', xlabel='Business Days', ylabel='PV')
# note that the OTM swaption prices at 0 post expiry whereas the ITM swaption prices at the value of the swap.
