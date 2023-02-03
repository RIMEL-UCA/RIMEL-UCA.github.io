#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import datetime, date
import pandas as pd
from gs_quant.instrument import IRSwap
from gs_quant.backtests.triggers import MeanReversionTrigger, MeanReversionTriggerRequirements
from gs_quant.backtests.actions import AddTradeAction
from gs_quant.backtests.generic_engine import GenericEngine
from gs_quant.backtests.strategy import Strategy
from gs_quant.backtests.data_sources import GenericDataSource, MissingDataStrategy
from gs_quant.risk import Price
from gs_quant.common import Currency, PayReceive
from gs_quant.data import Dataset

# In[ ]:


# Initialize session
from gs_quant.session import GsSession
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics','read_product_data')) 

# ## Buy 10y EUR payers

# In[ ]:


# Define backtest dates
start_date = date(2021, 6, 1)
end_date = datetime.today().date()

# In[ ]:


# Define instrument for strategy

# IR Swap
swap = IRSwap(pay_or_receive=PayReceive.Pay, termination_date='10y', notional_currency=Currency.EUR, 
              notional_amount=1e4, fixed_rate='ATM', name='swap_10y')

# #### Dataset from Marquee Data Catalog: https://marquee.gs.com/s/developer/datasets/SWAPRATES_STANDARD

# In[ ]:


# Get data set for 10y EUR swap rates
ds = Dataset('SWAPRATES_STANDARD')
data = ds.get_data(start_date, assetId=['MA5WM2QWRVMYKDK0'])
data_10y = data.loc[data['tenor'] == '10y']
data_10y.head()

# In[ ]:


# Mean Reversion Trigger

# Define our actions: add the swap
action = AddTradeAction(swap)

# Define data source
s = pd.Series(data_10y['rate'].to_dict())
data_source = GenericDataSource(s, MissingDataStrategy.fill_forward) 

# Define bounds
z_score_bound = 2
rolling_mean_window = 30
rolling_std_window = 30

# Build trigger requirements
# Sell when value hits z_score_bound on up side, Buy when value hits z_score_bound on down side
# Close out position when value crosses mean for the rolling_mean_window
trig_req = MeanReversionTriggerRequirements(data_source, z_score_bound, rolling_mean_window, rolling_std_window)
trigger = MeanReversionTrigger(trig_req, action)
strategy = Strategy(None, trigger)

# Run backtest daily
GE = GenericEngine()
backtest = GE.run_backtest(strategy, start=start_date, end=end_date, frequency='1b', show_progress=True)

# In[ ]:


# View backtest trade ledger
backtest.trade_ledger()

# In[ ]:


# View results summary
backtest.result_summary

# In[ ]:


# View Performance
pd.DataFrame({'Generic backtester': backtest.result_summary['Cumulative Cash'] + backtest.result_summary[Price]}).plot(figsize=(10, 6), title='Performance')

# In[ ]:



