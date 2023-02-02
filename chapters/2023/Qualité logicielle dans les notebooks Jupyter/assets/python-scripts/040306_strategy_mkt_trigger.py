#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import date
import pandas as pd
from gs_quant.instrument import FXOption
from gs_quant.common import BuySell, OptionType
from gs_quant.backtests.triggers import MktTrigger, MktTriggerRequirements, TriggerDirection
from gs_quant.backtests.actions import AddTradeAction
from gs_quant.backtests.generic_engine import GenericEngine
from gs_quant.backtests.strategy import Strategy
from gs_quant.backtests.data_sources import GenericDataSource, MissingDataStrategy
from gs_quant.data import Dataset
from gs_quant.risk import Price

# In[ ]:


# Initialize session
from gs_quant.session import GsSession
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics','read_product_data')) 

# ### Buy 1m USDCNH Put if spot below 6.42

# In[ ]:


# Define backtest dates
start_date = date(2021, 8, 1)
end_date = date(2022, 2, 1)

# In[ ]:


# Define instrument for strategy

# FX Option
put = FXOption(buy_sell=BuySell.Buy, option_type=OptionType.Put, pair='USDCNH', strike_price='ATMF', 
               expiration_date='1m', name='1m_put')

# ##### Dataset from Marquee Data Catalog: https://marquee.gs.com/s/developer/datasets/FXSPOT_PREMIUM

# In[ ]:


# Get data set for USDCNH spot
ds = Dataset('FXSPOT_PREMIUM')
data = ds.get_data(start_date, assetId=['MAEFRJZ9NYGDDR41'])
s = pd.Series(data['spot'].to_dict())
data_source = GenericDataSource(s, MissingDataStrategy.fill_forward) 

# In[ ]:


# View historical spot data
s.plot(figsize=(10, 6), title='USDCNH Spot')

# In[ ]:


# Define trade to add
action_add = AddTradeAction(put)

# Define market trigger strategy
mkt_trigger = MktTrigger(MktTriggerRequirements(data_source, 6.42, TriggerDirection.BELOW), action_add)
strategy = Strategy(None, mkt_trigger)

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



