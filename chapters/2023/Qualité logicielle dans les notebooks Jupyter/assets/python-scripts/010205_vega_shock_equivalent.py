#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.common import PayReceive, Currency
from gs_quant.instrument import IRSwaption
from gs_quant.markets import PricingContext
from gs_quant.session import Environment, GsSession
from gs_quant.risk import MarketDataShockBasedScenario, MarketDataPattern, MarketDataShock, MarketDataShockType, IRVegaParallel

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# Risk measures are short-hand for a set of predefined scenarios. Let's look at how at to get the `IRVegaParallel` value using a `MarketDataShockBasedScenario`. We calculate delta as a 2-sided 1bp bump.

# In[ ]:


swaption = IRSwaption(PayReceive.Pay, '10y', Currency.USD, notional_amount=10e6)
swaption.resolve()

# In[ ]:


r_shock_bp = 1/10000
ir_scenario_up = MarketDataShockBasedScenario({MarketDataPattern('IR Vol'): 
                                               MarketDataShock(MarketDataShockType.Absolute, r_shock_bp)})
ir_scenario_down = MarketDataShockBasedScenario({MarketDataPattern('IR Vol'): 
                                                 MarketDataShock(MarketDataShockType.Absolute, -r_shock_bp)})
with PricingContext():
    vega = swaption.calc(IRVegaParallel)
    with ir_scenario_up:
        up = swaption.dollar_price()
    with ir_scenario_down:
        down = swaption.dollar_price()

# In[ ]:


combined = (up.result() - down.result())/2
# should give the same value
combined - delta.result()

print(f'Vega direct={vega.result():.0f}, Vega by shocks={((up.result() - down.result())/2):.0f}')
