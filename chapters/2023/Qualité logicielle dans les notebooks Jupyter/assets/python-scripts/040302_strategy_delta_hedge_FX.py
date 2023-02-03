#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import datetime, date
import pandas as pd
from gs_quant.instrument import FXOption, FXForward
from gs_quant.common import BuySell, OptionType, AggregationLevel
from gs_quant.backtests.triggers import PeriodicTrigger, PeriodicTriggerRequirements
from gs_quant.backtests.actions import AddTradeAction, HedgeAction
from gs_quant.backtests.generic_engine import GenericEngine
from gs_quant.backtests.strategy import Strategy
from gs_quant.risk import Price, FXDelta

# In[ ]:


# Initialize session
from gs_quant.session import GsSession
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics',)) 

# ## Buy $100k 2y USDJPY ATMF call, roll monthly, delta hedge monthly

# In[ ]:


# Define backtest dates
start_date = date(2021, 6, 1)
end_date = datetime.today().date()

# In[ ]:


# Define instrument for strategy

# FX Option
call = FXOption(buy_sell=BuySell.Buy, option_type=OptionType.Call, pair='USDJPY', strike_price='ATMF', expiration_date='2y', 
                notional_amount=1e6, name='2y_call')

# In[ ]:


# Risk Trigger: based on frequency threshold, delta hedge by Forward trade

# Define frequency for adding trade 
freq_add = '1m'
trig_req = PeriodicTriggerRequirements(start_date=start_date, end_date=end_date, frequency=freq_add)
action_add = AddTradeAction(call, freq_add)

# Define trade to hedge FX Delta
freq_hedge = '1m'
fwd_hedge = FXForward(pair='USDJPY', settlement_date='2y', name='2y_forward')
hedge_risk = FXDelta(currency='USD', aggregation_level='Type')
action_hedge = HedgeAction(hedge_risk, fwd_hedge, freq_hedge)

# starting with empty portfolio (first arg to Strategy), apply actions in order on trig_req
triggers = PeriodicTrigger(trig_req, [action_add, action_hedge])
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


# View FX Delta risk
pd.DataFrame({'Generic backtester': backtest.result_summary[FXDelta(aggregation_level=AggregationLevel.Type, currency='USD')]}).plot(figsize=(10, 6), title='FX Delta')

# In[ ]:



