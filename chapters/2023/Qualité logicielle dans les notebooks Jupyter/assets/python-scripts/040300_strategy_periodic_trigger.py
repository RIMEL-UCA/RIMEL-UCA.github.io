#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import datetime, date
import pandas as pd
from gs_quant.instrument import FXOption
from gs_quant.common import BuySell, OptionType
from gs_quant.backtests.triggers import PeriodicTrigger, PeriodicTriggerRequirements
from gs_quant.backtests.actions import AddTradeAction
from gs_quant.backtests.generic_engine import GenericEngine
from gs_quant.backtests.strategy import Strategy
from gs_quant.risk import Price

# In[ ]:


# Initialize session
from gs_quant.session import GsSession
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics',)) 

# ## Buy $100k 2y USDJPY ATMF call, roll monthly

# In[ ]:


# Define backtest dates
start_date = date(2021, 6, 1)
end_date = datetime.today().date()

# In[ ]:


# Define instrument for strategy

# FX Option
call = FXOption(buy_sell=BuySell.Buy, option_type=OptionType.Call, pair='USDJPY', strike_price='ATMF', expiration_date='2y', name='2y_call')

# In[ ]:


# Periodic trigger: based on frequency
freq = '1m'
trig_req = PeriodicTriggerRequirements(start_date=start_date, end_date=end_date, frequency=freq)

# hold the trade for 1m
actions = AddTradeAction(call, freq)

# starting with empty portfolio (first arg to Strategy), apply actions on trig_req
triggers = PeriodicTrigger(trig_req, actions)
strategy = Strategy(None, triggers)

# run backtest daily
GE = GenericEngine()
backtest = GE.run_backtest(strategy, start=start_date, end=end_date, frequency='1b', show_progress=True)

# In[ ]:


# View backtest trade ledger
backtest.trade_ledger()

# In[ ]:


# View results summary
backtest.result_summary

# In[ ]:


# View Mark to Market
pd.DataFrame({'Generic backtester': backtest.result_summary[Price]}).plot(figsize=(10, 6), title='Mark to market')

# In[ ]:


# View Performance
pd.DataFrame({'Generic backtester': backtest.result_summary['Cumulative Cash'] + backtest.result_summary[Price]}).plot(figsize=(10, 6), title='Performance')

# In[ ]:



