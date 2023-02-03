#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import date, datetime
from gs_quant.instrument import IRSwaption
from gs_quant.common import BuySell, AggregationLevel, Currency
from gs_quant.backtests.triggers import StrategyRiskTrigger, RiskTriggerRequirements, TriggerDirection
from gs_quant.backtests.actions import AddTradeAction, ExitTradeAction
from gs_quant.backtests.generic_engine import GenericEngine
from gs_quant.backtests.strategy import Strategy
from gs_quant.risk import Price, IRDelta
import pandas as pd

# In[ ]:


# Initialize session
from gs_quant.session import GsSession
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics',)) 

# In[ ]:


# Define backtest dates
start_date = date(2021, 6, 1)
end_date = datetime.today().date()

# In[ ]:


# Define instruments for strategy

# USD swaption
swaption = IRSwaption(expiration_date='6m', termination_date='10y', notional_currency=Currency.USD, 
                      buy_sell=BuySell.Sell, strike='=solvefor(25e3,bp)', name='swaption10y')

# ### Start with USD swaption on start_date, rebalance when IR Delta breaches +75k

# In[ ]:


# Define risk to hedge
rebal_risk = IRDelta(aggregation_level=AggregationLevel.Type, currency='USD')

# Define trade actions when breach
action_exit = ExitTradeAction()
action_add = AddTradeAction(swaption)

# Define rebalance triggers
trig_req = RiskTriggerRequirements(risk=rebal_risk, trigger_level=1e3, direction=TriggerDirection.ABOVE)
# Order is important here, we first want to exit all positions in portfolio, then add the new trade
trigger_risk = StrategyRiskTrigger(trig_req, [action_exit, action_add])

# Starting with a swaption (first arg to Strategy), apply actions in order
strategy = Strategy(swaption, trigger_risk)

# Run backtest daily
GE = GenericEngine()
backtest = GE.run_backtest(strategy, start=start_date, end=end_date, frequency='1b', show_progress=True)

# In[ ]:


# View results summary
backtest.result_summary

# In[ ]:


# View backtest trade ledger - this includes all trades that were added and removed excluding starting portfolio
backtest.trade_ledger()

# In[ ]:


# View Mark to Market
pd.DataFrame({'Generic backtester': backtest.result_summary[Price]}).plot(figsize=(10, 6), title='Mark to market')

# In[ ]:


# View Performance
pd.DataFrame({'Generic backtester': backtest.result_summary['Cumulative Cash'] + backtest.result_summary[Price]}).plot(figsize=(10, 6), title='Performance')
