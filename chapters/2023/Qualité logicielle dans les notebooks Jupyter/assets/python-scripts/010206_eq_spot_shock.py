#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.common import PayReceive, Currency
from gs_quant.instrument import EqOption
from gs_quant.session import Environment, GsSession
from gs_quant.risk import MarketDataShockBasedScenario, MarketDataPattern, MarketDataShock, MarketDataShockType
from gs_quant.risk import EqDelta, EqVega, EqSpot, Price

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


o = EqOption('.SPX')
o.resolve()
p = o.price()
d1 = o.calc(EqDelta)
print(f'Base price: {p:,.2f}')
print(f'Equity Delta: {d1:,.2f}')

# In[ ]:


# shock equity spot up and down
with MarketDataShockBasedScenario(shocks={MarketDataPattern(mkt_type='Eq', mkt_class='Spot', mkt_quoting_style='PropSpread'): MarketDataShock(MarketDataShockType.Override, .005)}):
    price_up = o.price()
with MarketDataShockBasedScenario(shocks={MarketDataPattern(mkt_type='Eq', mkt_class='Spot', mkt_quoting_style='PropSpread'): MarketDataShock(MarketDataShockType.Override, -.005)}):
    price_down = o.price()
d2 = (price_up - price_down) / .01

print(f'Equity Delta via shocking: {d2:,.2f}')


# In[ ]:



