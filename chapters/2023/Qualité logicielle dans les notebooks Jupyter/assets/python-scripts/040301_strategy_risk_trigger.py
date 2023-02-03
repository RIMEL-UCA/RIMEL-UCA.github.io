#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import datetime, date
import pandas as pd
from gs_quant.instrument import FXOption, FXForward
from gs_quant.common import BuySell, OptionType, AggregationLevel
from gs_quant.backtests.triggers import PeriodicTrigger, PeriodicTriggerRequirements, StrategyRiskTrigger, RiskTriggerRequirements, TriggerDirection
from gs_quant.backtests.actions import AddTradeAction, HedgeAction
from gs_quant.backtests.generic_engine import GenericEngine
from gs_quant.backtests.strategy import Strategy
from gs_quant.risk import Price, FXDelta

# In[ ]:


# Initialize session
from gs_quant.session import GsSession
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics',)) 

# ## Buy $100k 1y EURUSD ATMF call, roll monthly, delta hedge when breach of +50k

# In[ ]:


# Define backtest dates
start_date = date(2021, 6, 1)
end_date = datetime.today().date()

# In[ ]:


# Define instrument for strategy

# FX Option
call = FXOption(buy_sell=BuySell.Buy, option_type=OptionType.Call, pair='EURUSD', strike_price='ATMF',
                expiration_date='1y', notional_amount=1e5, name='1y_call')

# In[ ]:


# Let's look at FX Delta for strategy without hedging

# Define frequency for adding trade 
freq_add = '1m'
trig_req = PeriodicTriggerRequirements(start_date=start_date, end_date=end_date, frequency=freq_add)
action_add = AddTradeAction(call, freq_add)

# Starting with empty portfolio (first arg to Strategy), apply actions in order on trig_req
triggers = PeriodicTrigger(trig_req, action_add)
strategy = Strategy(None, triggers)

# Run backtest daily
GE = GenericEngine()
backtest = GE.run_backtest(strategy, start=start_date, end=end_date, frequency='1b', show_progress=True, 
                           risks=[Price, FXDelta(aggregation_level=AggregationLevel.Type, currency='USD')])

# In[ ]:


# View FX Delta risk
b1 = pd.DataFrame({'No Hedge': backtest.result_summary[FXDelta(aggregation_level=AggregationLevel.Type, currency='USD')]})
b1.plot(figsize=(10, 6), title='FX Delta')

# In[ ]:


# View results summary
backtest.result_summary

# In[ ]:


# Let's hedge delta when it is above +50k

# Define frequency for adding trade 
freq_add = '1m'
trig_req_add = PeriodicTriggerRequirements(start_date=start_date, end_date=end_date, frequency=freq_add)
action_add = AddTradeAction(call, freq_add)
trigger_add = PeriodicTrigger(trig_req_add, action_add)

# Define risk to hedge
hedge_risk = FXDelta(aggregation_level='Type', currency='USD')

# Define FX Forward to hedge with
fwd_hedge = FXForward(pair='EURUSD', settlement_date='1y', notional_amount=1e5, name='1y_forward')
action_hedge = HedgeAction(hedge_risk, fwd_hedge)

# Define hedge triggers
trig_req_hedge = RiskTriggerRequirements(risk=hedge_risk, trigger_level=50e3, direction=TriggerDirection.ABOVE)
trigger_risk = StrategyRiskTrigger(trig_req_hedge, action_hedge)

# Starting with empty portfolio (first arg to Strategy), apply actions in order on trig_req
strategy = Strategy(None, [trigger_add, trigger_risk])

# Run backtest daily
GE = GenericEngine()
backtest = GE.run_backtest(strategy, start=start_date, end=end_date, frequency='1b', show_progress=True)

# In[ ]:


# View FX Delta risk
b2 = pd.DataFrame({'Hedge above +50k': backtest.result_summary[FXDelta(aggregation_level=AggregationLevel.Type,
                                                                   currency='USD')]})
b2.plot(figsize=(10, 6), title='FX Delta')

# In[ ]:


# View backtest trade ledger
backtest.trade_ledger()

# In[ ]:


# Let's compare the two results
b1.columns = ['No Hedge']
b2.columns = ['Hedge above +50k']
pd.concat([b1, b2], axis=1).plot(figsize=(10, 6))

# In[ ]:



