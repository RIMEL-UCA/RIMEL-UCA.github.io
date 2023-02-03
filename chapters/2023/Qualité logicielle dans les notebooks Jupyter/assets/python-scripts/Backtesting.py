#!/usr/bin/env python
# coding: utf-8

# Examples require an initialized GsSession and relevant entitlements. External clients need to substitute thier own client id and client secret below. Please refer to [Sessions](https://developer.gs.com/docs/gsquant/Authentication/gs-session/) for details.

# In[ ]:


from gs_quant.session import GsSession
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics',))

# ## Backtesting
# 
# The backtesting framework is a language for describing backtests that's (1) intuitive (2) cross-asset and instrument agnostic (3) available natively in python for users to change and extend. It's compatible with a variety of calculation engines - eliminating need for specific syntax knowledge - and uses the gs risk API as the default calculation engine.
# 
# The language consists of several basic components:
# * [Trigger](#Trigger)
# * [Action](#Action)
# * [Strategy](#Strategy)
# * [Calculation Engine](#Calculation-Engine)
# * [Backtest](#Backtest)
# 
# In this tutorial, we will examine the components in the context of looking at a basic vol selling strategy where we sell a 1m10y USD straddle daily.

# ### Trigger
# 
# A trigger can be used to amend a strategy based on a set of requirements and the current state of the backtest. Each trigger has an associated action or set of actions that are applied when the trigger is active. A backtest can have multiple triggers.
# 
# Below are the triggers we currently offer:
# * `PeriodicTrigger`: triggers at a regular interval (time/date). e.g.: every month end
# * `MktTrigger`: triggers when market data, absolute or change, hits some target e.g. whenever SPX 1m vol crosses 10. Check out our [data catalog](https://marquee.gs.com/s/discover/data-services/catalog) for sources of market data that can be leveraged here
# * `StrategyRiskTrigger`: triggers when the current risk of the strategy hits some target e.g. delta crosses $10m
# 
# Since in our example, we are selling an instrument periodically e.g. (daily), we will use a single `PeriodicTrigger.` Let's define the `PeriodicTriggerRequirements` first and then talk about actions, since we need to specify at least a single action to happen when this trigger is triggered.

# In[ ]:


from gs_quant.backtests.triggers import PeriodicTrigger, PeriodicTriggerRequirements
from datetime import date

start_date, end_date = date(2020, 1, 1), date(2020, 12, 1)
# define dates on which actions will be triggered (1b=every business day here)
trig_req = PeriodicTriggerRequirements(start_date=start_date, end_date=end_date, frequency='1b')

# ### Action
# 
# Actions define how the portfolio is updated when a trigger is triggered. Below are the actions we currently offer:
# * `AddTradeAction`: adds trade to a strategy
# * `ExitTradeAction`: removes trade from a strategy
# * `HedgeAction`: add trade that's a hedge for a risk
# 
# Since in our example we are selling a straddle, we only want a single `AddTradeAction` to supply to our trigger. Let's define it.

# In[ ]:


from gs_quant.backtests.actions import AddTradeAction
from gs_quant.common import PayReceive, Currency
from gs_quant.instrument import IRSwaption

# straddle is the position we'll be adding (note it's a short position since buy_sell='Sell')
straddle = IRSwaption(PayReceive.Straddle, '10y', Currency.USD, expiration_date='1m', notional_amount=1e8, buy_sell='Sell')

# this action specifies we will add the straddle when this action is triggered and hold it until expiration_date
action = AddTradeAction(straddle, 'expiration_date')

# we will now combine our trigger requirement and action to produce a PeriodicTrigger
# Note, action can be a list if there are multiple actions
trigger = PeriodicTrigger(trig_req, action)

# ### Strategy
# 
# A `Strategy` combines all the information needed to run a backtest. It has 2 components: an initial trade or portfolio and a trigger or set of triggers. 
# 
# In our example, we don't have a starting portfolio and we have a single trigger we defined in the previous step. Let's put these together.

# In[ ]:


from gs_quant.backtests.strategy import Strategy

# in this case we have a single trigger but this can be a list of triggers as well
strategy = Strategy(None, trigger)

# ### Calculation engine
# 
# The final piece is calculate engine - the underlying engine/source of calculations you want to run your backtest. 
# 
# Once you have defined your backtest in the steps above, you can view which available engines support your strategy. There may be multiple as different engines have been developed and optimized for certain types of backtesting. For example, the `EquityVolEngine` is optimized for fast volatility backtesting for a specific set of equity underlyers. That said, `GenericEngine` will support majority of use cases, running calculations using gs-quant's risk apis. You can check which available engines support your strategy using the function below:

# In[ ]:


strategy.get_available_engines()

# Alternatively, you check whether a particular engine supports your strategy:

# In[ ]:


from gs_quant.backtests.generic_engine import GenericEngine

ge = GenericEngine()
ge.supports_strategy(strategy)

# As a final step, let's put everything together to run the backtest. The frequency here indicates the frequency of calculations. You can also supply additional risks you may want to calculate as part of your backtest (greeks or dollar price, for example).

# In[ ]:


backtest = ge.run_backtest(strategy, start=start_date, end=end_date, frequency='1b', show_progress=True)
backtest

# ### Backtest
# 
# * to view the results of `backtest.risks` (by default, `Price`) (all in local ccy) and the cumulative cashflows evaluated on each backtested date: `backtest.results_summary`
# * to view status for each trade: `backtest.trade_ledger()`
# * to view and further evaluate the portfolio on any given backtested date: `backtest.portfolio_dict`
# * to view the number of total api calls and number of calculations performed: `backtest.calc_calls` and `backtest.calculations`
# 
# Here we'll use `backtest.results_summary` to plot mark to market and performance of the strategy

# In[ ]:


backtest.trade_ledger()

# In[ ]:


backtest.result_summary

# In[ ]:


import pandas as pd
from gs_quant.risk import Price
pd.DataFrame({'Generic backtester': backtest.result_summary[Price]}).plot(figsize=(10, 6), title='Mark to market')

# In[ ]:


pd.DataFrame({'Generic backtester': backtest.result_summary['Cumulative Cash'] + backtest.result_summary[Price]}).plot(figsize=(10, 6), title='Performance')

# If you have feedback on further expanding this framework, please reach out to `gs-quant-dev@gs.com`
