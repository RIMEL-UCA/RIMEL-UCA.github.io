#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.risk import RollFwd, MarketDataPattern, MarketDataShock, MarketDataShockBasedScenario,MarketDataShockType
from gs_quant.session import Environment, GsSession
from gs_quant.common import PayReceive, Currency
from gs_quant.instrument import IRSwap

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


# define swap
swap = IRSwap(termination_date='10y', notional_currency=Currency.USD, pay_or_receive=PayReceive.Pay, fixed_rate=0.007)

# In[ ]:


# 1bp Spot Shock
ir_1bp_scenario = MarketDataShockBasedScenario(shocks={
    MarketDataPattern('IR', 'USD') : MarketDataShock(shock_type=MarketDataShockType.Absolute, value=1e-4),
    MarketDataPattern('IR Reset', 'USD-3m') : MarketDataShock(shock_type=MarketDataShockType.Absolute, value=1e-4)
})

# Roll Forward 22b
rollfwd_22b_scenario = RollFwd(date='22b', holiday_calendar='NYC')

# In[ ]:


# Roll Forward 22b and apply 1bp Spot Shock
with rollfwd_22b_scenario, ir_1bp_scenario:
    rollfwd_22b_then_ir_1bp_price = swap.price()
    
# Look at the difference between scenario and base prices
print('Base price:     {:,.2f}'.format(swap.price()))
print('Scenario price: {:,.2f}'.format(rollfwd_22b_then_ir_1bp_price))
