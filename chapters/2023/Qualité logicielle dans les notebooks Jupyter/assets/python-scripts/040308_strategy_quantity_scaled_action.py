#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.session import Environment
from gs_quant.instrument import OptionStyle
from gs_quant.backtests.strategy import Strategy
from gs_quant.backtests.triggers import *
from gs_quant.backtests.actions import *
from gs_quant.backtests.equity_vol_engine import *
from gs_quant.backtests.generic_engine import GenericEngine
from gs_quant.target.measures import Price
from datetime import datetime, date

import gs_quant.risk as risk

# In[ ]:


# Initialize session
from gs_quant.session import GsSession
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics',)) 

# In[ ]:


# Define backtest dates
start_date = date(2021, 4, 1)
end_date = datetime.today().date()

# In[ ]:


# Define instruments for strategy
# Portfolio of two eq options
call = EqOption('.STOXX50E', expiration_date='1m', strike_price='ATM', option_type=OptionType.Call, option_style=OptionStyle.European, name='call')
put = EqOption('.STOXX50E', expiration_date='1m', strike_price='ATM', option_type=OptionType.Put, option_style=OptionStyle.European, name='put')
portfolio = Portfolio(name='portfolio', priceables=[call, put])

# ### Entering a position monthly

# In[ ]:


# Trade the position monthly without any scaling
trade_action = EnterPositionQuantityScaledAction(priceables=portfolio, trade_duration='1m', name='act')
trade_trigger = PeriodicTrigger(trigger_requirements=PeriodicTriggerRequirements(start_date=start_date, end_date=end_date, frequency='1m'),
                                actions=trade_action)

strategy = Strategy(None, trade_trigger)

# Run backtest daily
GE = GenericEngine()
backtest = GE.run_backtest(strategy, start=start_date, end=end_date, frequency='1b', show_progress=True)

# In[ ]:


# View results summary
backtest.result_summary

# In[ ]:


# View backtest trade ledger
backtest.trade_ledger()

# In[ ]:


# View Mark to Market
pd.DataFrame({'Generic backtester': backtest.result_summary[Price]}).plot(figsize=(10, 6), title='Mark to market')

# In[ ]:


# View Performance
pd.DataFrame({'Generic backtester': backtest.result_summary['Cumulative Cash'] + backtest.result_summary[Price]}).plot(figsize=(10, 6), title='Performance')

# ### Scaling quantity

# In[ ]:


# Scale the position quantity (no. of options) by 2
trade_action_scaled = EnterPositionQuantityScaledAction(priceables=portfolio, trade_duration='1m', trade_quantity=2, name='scaled_act')
trade_trigger_scaled = PeriodicTrigger(trigger_requirements=PeriodicTriggerRequirements(start_date=start_date, end_date=end_date, frequency='1m'),
                                actions=trade_action_scaled)

strategy = Strategy(None, trade_trigger_scaled)

# Run backtest daily
GE = GenericEngine()
backtest = GE.run_backtest(strategy, start=start_date, end=end_date, frequency='1b', show_progress=True)

# In[ ]:


# View results summary. Note that MtM and cash payments are double
backtest.result_summary

# In[ ]:


# View backtest trade ledger. Note that all values are double
backtest.trade_ledger()

# In[ ]:


# View Mark to Market
pd.DataFrame({'Generic backtester': backtest.result_summary[Price]}).plot(figsize=(10, 6), title='Mark to market')

# In[ ]:


# View Performance
pd.DataFrame({'Generic backtester': backtest.result_summary['Cumulative Cash'] + backtest.result_summary[Price]}).plot(figsize=(10, 6), title='Performance')

# ### Scaling a risk measure

# In[ ]:


# Initial and final Vega of the portfolio
with PricingContext(start_date):
    start_vega = portfolio.calc(risk.EqVega)

with PricingContext(end_date):
    end_vega = portfolio.calc(risk.EqVega)

print(start_vega.aggregate())
print(end_vega.aggregate())

# In[ ]:


# Scale the position vega to 100
trade_action_scaled = EnterPositionQuantityScaledAction(priceables=portfolio, trade_duration='1m', trade_quantity=100, trade_quantity_type=BacktestTradingQuantityType.vega, name='scaled_vega_act')
trade_trigger_scaled = PeriodicTrigger(trigger_requirements=PeriodicTriggerRequirements(start_date=start_date, end_date=end_date, frequency='1m'),
                                actions=trade_action_scaled)

strategy = Strategy(None, trade_trigger_scaled)

# Run backtest daily
GE = GenericEngine()
backtest = GE.run_backtest(strategy, start=start_date, end=end_date, frequency='1b', show_progress=True)

# In[ ]:


# View results summary. Note that MtM and cash payments are around 10x larger, as vega was scaled by around 10x
backtest.result_summary

# In[ ]:


# View backtest trade ledger. Note that all values are around 10x larger than without scaling
backtest.trade_ledger()

# In[ ]:


# View Mark to Market
pd.DataFrame({'Generic backtester': backtest.result_summary[Price]}).plot(figsize=(10, 6), title='Mark to market')

# In[ ]:


# View Performance
pd.DataFrame({'Generic backtester': backtest.result_summary['Cumulative Cash'] + backtest.result_summary[Price]}).plot(figsize=(10, 6), title='Performance')

# ### NAV Trading

# In[ ]:


# Create a NAV-based action that starts with 100 available to spend
start_date = date(2022, 1, 1)
end_date = datetime.today().date()

trade_action_scaled = EnterPositionQuantityScaledAction(priceables=portfolio, trade_duration='1m', trade_quantity=100, trade_quantity_type=BacktestTradingQuantityType.NAV, name='nav_act')
trade_trigger_scaled = PeriodicTrigger(trigger_requirements=PeriodicTriggerRequirements(start_date=start_date, end_date=end_date, frequency='1m'),
                                actions=trade_action_scaled)

strategy = Strategy(None, trade_trigger_scaled)

# Run backtest daily
GE = GenericEngine()
backtest = GE.run_backtest(strategy, start=start_date, end=end_date, frequency='1b', show_progress=True)

# In[ ]:


# View results summary. The action trades the starting cash and only uses proceeds from selling options after that: note that Cumulative Cash is always the initial trade quantity
backtest.result_summary

# In[ ]:


# View backtest trade ledger
backtest.trade_ledger()

# In[ ]:


# View Mark to Market. If zero right before an order, future orders are all zero
pd.DataFrame({'Generic backtester': backtest.result_summary[Price]}).plot(figsize=(10, 6), title='Mark to market')

# In[ ]:


# View Performance. Maximum loss is 100 if there is no short selling
pd.DataFrame({'Generic backtester': backtest.result_summary['Cumulative Cash'] + backtest.result_summary[Price]}).plot(figsize=(10, 6), title='Performance')
