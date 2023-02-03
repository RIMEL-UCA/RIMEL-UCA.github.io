#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import datetime, date
import pandas as pd
from gs_quant.instrument import IRSwap, IRSwaption
from gs_quant.backtests.triggers import PeriodicTrigger, PeriodicTriggerRequirements
from gs_quant.backtests.actions import AddTradeAction, HedgeAction
from gs_quant.backtests.generic_engine import GenericEngine
from gs_quant.backtests.strategy import Strategy
from gs_quant.risk import Price, IRDelta, DollarPrice
from gs_quant.common import Currency

# In[ ]:


# Initialize session
from gs_quant.session import GsSession
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics',)) 

# ## Sell swaption monthly, delta hedge monthly

# In[ ]:


# Define backtest dates
start_date = date(2021, 6, 1)
end_date = datetime.today().date()

# In[ ]:


# Define instrument for strategy

# IR Swaption
option = IRSwaption(expiration_date='1m', termination_date='30y', notional_currency=Currency.EUR, buy_sell='Sell')

# In[ ]:


# Risk Trigger: based on frequency threshold, delta hedge by swap trade

# Define frequency for adding trade 
freq_add = '1m'
trig_req = PeriodicTriggerRequirements(start_date=start_date, end_date=end_date, frequency=freq_add)
action_add = AddTradeAction(option, 'expiration_date')

# Define trade to hedge Delta
freq_hedge = '1m'
swap_hedge = IRSwap(termination_date='30y', notional_currency=Currency.EUR, notional_amount=100e6, name='30yhedge')
hedge_risk = IRDelta(aggregation_level='Type', currency=Currency.USD)
action_hedge = HedgeAction(hedge_risk, swap_hedge)

# Starting with empty portfolio (first arg to Strategy), apply actions in order on trig_req
triggers = PeriodicTrigger(trig_req, [action_add, action_hedge])
strategy = Strategy(None, triggers)

# Run backtest daily
GE = GenericEngine()
# The results will be in local ccy, EUR in this case. To change them to USD, specify result_ccy in the backtest args
backtest = GE.run_backtest(strategy, start=start_date, end=end_date, frequency='1b', show_progress=True)

# In[ ]:


# View backtest trade ledger
backtest.trade_ledger()

# In[ ]:


# View results summary
backtest.result_summary

# In[ ]:


# Plot results
df = backtest.result_summary
df['Performance'] = df[Price] + df['Cumulative Cash']
df.plot(figsize=(10, 10)).legend(bbox_to_anchor=(1, 1))

# In[ ]:



