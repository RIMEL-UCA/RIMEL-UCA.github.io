#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.common import PayReceive, Currency
from gs_quant.instrument import IRSwap
from gs_quant.markets import PricingContext
from gs_quant.session import Environment, GsSession
from gs_quant.risk import MarketDataShockBasedScenario, MarketDataPattern, MarketDataShock, MarketDataShockType, IRDeltaParallel

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# Risk measures are short-hand for a set of predefined scenarios. Let's look at how at to get the `IRDeltaParallel` value using a `MarketDataShockBasedScenario`. We calculate delta as a 2-sided 1bp bump.

# In[ ]:


swap = IRSwap(PayReceive.Pay, '10y', Currency.USD, notional_amount=10e6)
swap.resolve()

# In[ ]:


r_shock_bp = 1/10000
ir_scenario_up = MarketDataShockBasedScenario({MarketDataPattern('IR', mkt_asset=Currency.USD): 
                                               MarketDataShock(MarketDataShockType.Absolute, r_shock_bp)})
ir_scenario_down = MarketDataShockBasedScenario({MarketDataPattern('IR', mkt_asset=Currency.USD): 
                                                 MarketDataShock(MarketDataShockType.Absolute, -r_shock_bp)})
with PricingContext(market_data_location='NYC'):
    delta = swap.calc(IRDeltaParallel)
    with ir_scenario_up:
        up = swap.dollar_price()
    with ir_scenario_down:
        down = swap.dollar_price()

# In[ ]:


combined = (up.result() - down.result())/2
# should give the same value
print(f'Delta direct={delta.result():.0f}, Delta by shocks={((up.result() - down.result())/2):.0f}')
