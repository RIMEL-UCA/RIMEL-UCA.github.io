#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.instrument import IRSwaption
from gs_quant.risk import CurveScenario, MarketDataPattern, MultiScenario
from gs_quant.session import Environment, GsSession
from gs_quant_internal.boltweb import valuation

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


swaption = IRSwaption('Receive', '5y', 'USD', expiration_date='13m', strike='atm')
swaption.resolve()

# In[ ]:


original_price = swaption.price()
market_data = swaption.market().market_data_dict
print('Base price:     {:,.2f}'.format(original_price))

# In[ ]:


# Create multiple curve scenarios (Multi Scenario only supports a list of the same type of scenario)
shifts = [5, 10, 15]
list_of_scenarios = [CurveScenario(market_data_pattern=MarketDataPattern('IR', 'USD'), parallel_shift=p, 
                                   name=f"parallel shift {p}") for p in shifts]
multi_scenario = MultiScenario(scenarios = list_of_scenarios)

# In[ ]:


# Price the swaption under a multiple curve scenarios
with multi_scenario:
    scenario_price = swaption.price()

# In[ ]:


scenario_price

# In[ ]:



