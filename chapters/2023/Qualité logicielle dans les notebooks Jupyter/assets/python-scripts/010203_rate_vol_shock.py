#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.common import PayReceive, Currency
from gs_quant.instrument import IRSwaption
from gs_quant.session import Environment, GsSession
from gs_quant.risk import MarketDataShockBasedScenario, MarketDataPattern, MarketDataShock, MarketDataShockType

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


ir_vol_scenario = MarketDataShockBasedScenario(
    shocks={MarketDataPattern('IR Vol'): MarketDataShock(MarketDataShockType.Absolute, 1e-4)}
)

# In[ ]:


swaption = IRSwaption(PayReceive.Receive, '5y', Currency.EUR, expiration_date='13m', strike='atm')
swaption.resolve()

with ir_vol_scenario:
    swaption_scenario_price = swaption.price()

# Look at the difference between scenario and base prices
print('Base price:     {:,.2f}'.format(swaption.price()))
print('Scenario price: {:,.2f}'.format(swaption_scenario_price))
