#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.session import GsSession, Environment
from gs_quant.instrument import EqOption, OptionType, OptionStyle
from gs_quant.backtests.strategy import Strategy
from gs_quant.backtests.triggers import *
from gs_quant.backtests.actions import *
from gs_quant.backtests.equity_vol_engine import *
from gs_quant.target.common import UnderlierType
from datetime import date
import pandas as pd

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


# Define backtest dates
start_date = date(2019, 9, 4)
end_date = date(2020, 9, 4)

# In[ ]:


# Define option instrument for strategy
option = EqOption('SX5E', underlierType=UnderlierType.BBID, expirationDate='3m', strikePrice='ATM', optionType=OptionType.Call, optionStyle=OptionStyle.European)

# In[ ]:


# Define a periodic trade action.  Trade 1000 units of the option instrument and hold for 1m
trade_action = EnterPositionQuantityScaledAction(priceables=option, trade_duration='1m', trade_quantity=1000, trade_quantity_type=BacktestTradingQuantityType.quantity)

# Define an entry signal trigger.  This is a list of dates when the entry signal is true.
# When entry signal is true, the strategy will enter a position (only if no position is already held)
entry_dates = (date(2019,9,20), date(2019,10,18), date(2019,11,15), date(2019,12,20), date(2020,1,17), date(2020,2,21))
entry_trigger = AggregateTrigger(triggers=[
    DateTrigger(trigger_requirements=DateTriggerRequirements(dates=entry_dates), actions=trade_action),
    PortfolioTrigger(trigger_requirements=PortfolioTriggerRequirements('len', 0, TriggerDirection.EQUAL))
])

# Define an exit signal trigger.  This is a list of dates when the exit signal is true.
# When the exit signal is true the strategy will exit the full position (only if already holding a position)
exit_dates = (date(2019,11,4), date(2020,3,2))
exit_trigger = AggregateTrigger(triggers=[
    DateTrigger(trigger_requirements=DateTriggerRequirements(dates=exit_dates), actions=ExitTradeAction()),
    PortfolioTrigger(trigger_requirements=PortfolioTriggerRequirements('len', 0, TriggerDirection.ABOVE))
])

# In[ ]:


# Create strategy
strategy = Strategy(initial_portfolio=None, triggers=[entry_trigger, exit_trigger])

# In[ ]:


# Run backtest
backtest = EquityVolEngine.run_backtest(strategy, start=start_date, end=end_date)

# In[ ]:


# Plot the performance
pnl = backtest.get_measure_series(FlowVolBacktestMeasure.PNL)
pnl.plot(legend=True, label='PNL Total')
