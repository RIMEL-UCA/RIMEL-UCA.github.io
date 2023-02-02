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
instrument = EqOption('SX5E', underlierType=UnderlierType.BBID, expirationDate='3m', strikePrice='ATM', optionType=OptionType.Call, optionStyle=OptionStyle.European)

# In[ ]:


# Define a periodic trigger and action.  Trade and roll the option instrument every 1m
trade_action = EnterPositionQuantityScaledAction(priceables=instrument, trade_duration='1m', trade_quantity=1000, trade_quantity_type=BacktestTradingQuantityType.quantity)
trade_trigger = PeriodicTrigger(trigger_requirements=PeriodicTriggerRequirements(start_date=start_date, end_date=end_date, frequency='1m'), actions=trade_action)

# In[ ]:


# Create strategy
strategy = Strategy(initial_portfolio=None, triggers=[trade_trigger])

# In[ ]:


# Run backtest with SFK market model (the default). Other options are SD and SVR
backtest = EquityVolEngine.run_backtest(strategy, start=start_date, end=end_date, market_model='SFK')
pnl_sfk = backtest.get_measure_series(FlowVolBacktestMeasure.PNL)
delta_sfk = backtest.get_measure_series(FlowVolBacktestMeasure.delta)
gamma_sfk = backtest.get_measure_series(FlowVolBacktestMeasure.gamma)
vega_sfk = backtest.get_measure_series(FlowVolBacktestMeasure.vega)

# In[ ]:


# Run backtest with SVR market model
backtest = EquityVolEngine.run_backtest(strategy, start=start_date, end=end_date, market_model='SVR')
pnl_svr = backtest.get_measure_series(FlowVolBacktestMeasure.PNL)
delta_svr = backtest.get_measure_series(FlowVolBacktestMeasure.delta)
gamma_svr = backtest.get_measure_series(FlowVolBacktestMeasure.gamma)
vega_svr = backtest.get_measure_series(FlowVolBacktestMeasure.vega)

# In[ ]:


# Market model does not affect performance calculation
relative_pnl_diff = (pnl_svr - pnl_sfk)/pnl_sfk
relative_pnl_diff.plot(legend=True, label='Relative PnL diff')

# In[ ]:


# Market model affects delta and gamma calculations
relative_delta_diff = (delta_svr - delta_sfk)/delta_sfk
relative_delta_diff.plot(legend=True, label='Relative delta diff')

# In[ ]:


relative_gamma_diff = (gamma_svr - gamma_sfk)/gamma_sfk
relative_gamma_diff.plot(legend=True, label='Relative gamma diff')

# In[ ]:


# Market model does not affect the calculation of any other risk measures
relative_vega_diff = (vega_svr - vega_sfk)/vega_sfk
relative_vega_diff.plot(legend=True, label='Relative vega diff')
