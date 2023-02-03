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

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


# Define backtest dates
start_date = date(2019, 9, 4)
end_date = date(2020, 9, 4)

# In[ ]:


# Define option instrument for strategy
option = EqOption('SX5E', underlierType=UnderlierType.BBID, expirationDate='3m', strikePrice='ATM', 
                  optionType=OptionType.Call, optionStyle=OptionStyle.European)

# In[ ]:


# Define a periodic trade trigger and action.  Trade and roll 1000 units of the option instrument every 1m
trade_action = EnterPositionQuantityScaledAction(priceables=option, trade_duration='1m', trade_quantity=1000, 
                                                 trade_quantity_type=BacktestTradingQuantityType.quantity)
trade_trigger = PeriodicTrigger(trigger_requirements=PeriodicTriggerRequirements(start_date=start_date, end_date=end_date, frequency='1m'), 
                                actions=trade_action)

# In[ ]:


# Create strategy
strategy = Strategy(initial_portfolio=None, triggers=[trade_trigger])

# In[ ]:


# Run backtest
backtest = EquityVolEngine.run_backtest(strategy, start=start_date, end=end_date)

# In[ ]:


# Plot the performance
pnl = backtest.get_measure_series(FlowVolBacktestMeasure.PNL)
pnl.plot(legend=True, label='PNL Total')

pnl_carry = backtest.get_measure_series(FlowVolBacktestMeasure.PNL_carry)
pnl_carry.plot(legend=True, label='PNL Carry')

pnl_vol = backtest.get_measure_series(FlowVolBacktestMeasure.PNL_vol)
pnl_vol.plot(legend=True, label='PNL Vol')
