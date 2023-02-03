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


# Define option instruments for strategy
option = EqOption('SX5E', underlierType=UnderlierType.BBID, expirationDate='3m', strikePrice='ATM', optionType=OptionType.Call, optionStyle=OptionStyle.European)

long_call = EqOption('SX5E', underlierType=UnderlierType.BBID, expiration_date='3m', strike_price='ATM', option_type=OptionType.Call,
                     option_style=OptionStyle.European, buy_sell=BuySell.Buy)
short_put = EqOption('SX5E', underlierType=UnderlierType.BBID, expiration_date='3m', strike_price='ATM', option_type=OptionType.Put,
                     option_style=OptionStyle.European, buy_sell=BuySell.Sell)

hedge_portfolio = Portfolio(name='SynFwd', priceables=[long_call, short_put])

# In[ ]:


# Define a periodic trade trigger and action.  

# Trade and roll 1000 units of the option instrument every 1m
trade_action = EnterPositionQuantityScaledAction(priceables=option, trade_duration='1m', trade_quantity=1000, trade_quantity_type=BacktestTradingQuantityType.quantity)
trade_trigger = PeriodicTrigger(trigger_requirements=PeriodicTriggerRequirements(start_date=start_date, end_date=end_date, frequency='1m'), 
                               actions=trade_action)

# Hedge option delta every day using the synthetic forward
hedge_action = HedgeAction(EqDelta, priceables=hedge_portfolio, trade_duration='1b')
hedge_trigger = PeriodicTrigger(trigger_requirements=PeriodicTriggerRequirements(start_date=start_date, end_date=end_date, frequency='1b'),
                               actions=hedge_action)


# In[ ]:


# Create strategy
strategy = Strategy(initial_portfolio=None, triggers=[trade_trigger, hedge_trigger])

# In[ ]:


# Run backtest
backtest = EquityVolEngine.run_backtest(strategy, start=start_date, end=end_date)

# In[ ]:


# Plot the performance
pnl = backtest.get_measure_series(FlowVolBacktestMeasure.PNL)
pnl.plot(legend=True, label='PNL')
