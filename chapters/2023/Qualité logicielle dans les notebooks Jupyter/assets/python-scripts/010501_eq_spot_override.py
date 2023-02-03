#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.common import OptionType, OptionStyle
from gs_quant.instrument import EqOption
from gs_quant.session import Environment, GsSession
from gs_quant.risk import MarketDataShockBasedScenario, MarketDataPattern, MarketDataShock, MarketDataShockType

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


eq_spot_scenario = MarketDataShockBasedScenario(
    shocks={MarketDataPattern('Eq', '.STOXX50E', 'Reference Spot', mkt_quoting_style='Price'): MarketDataShock(MarketDataShockType.Override, 3800)}
)

# In[ ]:


option = EqOption('.STOXX50E', expirationDate='3m', strikePrice='ATM', optionType=OptionType.Call, optionStyle=OptionStyle.European)

with eq_spot_scenario:
    option_spot_scenario_price = option.price()    

# Look at the difference between scenario and base prices
print('Base price:     {:,.2f}'.format(option.price()))
print('Spot scenario price: {:,.2f}'.format(option_spot_scenario_price))
