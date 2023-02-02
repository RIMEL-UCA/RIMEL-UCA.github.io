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
start_date = date(2022, 1, 1)
end_date = date(2022, 2, 1)

# In[ ]:


# Define instruments for strategy
# Portfolio of two eq options
spx_opt = EqOption('.SPX', expiration_date='2m', strike_price='ATM', option_type=OptionType.Call, option_style=OptionStyle.European, name='spx')
ndx_opt = EqOption('.NDX', expiration_date='2m', strike_price='ATM', option_type=OptionType.Call, option_style=OptionStyle.European, name='ndx')

# ## RebalanceAction rebalances a trade's quantity according to a custom function

# #### This function returns the no. of .NDX options held

# In[ ]:


def match_ndx_holding(state, backtest, info):
    port = backtest.portfolio_dict
    current_ndx_notional = sum([x.number_of_options for x in port[state].all_instruments if isinstance(x, EqOption) and x.underlier == '.NDX'])

    return current_ndx_notional

# #### AddTradeAction adds .NDX options daily, RebalanceAction rebalances the .SPX option according to the total number of .NDX options held weekly

# In[ ]:


# Add NDX options daily
add_trig_req = PeriodicTriggerRequirements(start_date=start_date, end_date=end_date, frequency='1b')
add_ndx = AddTradeAction(ndx_opt, '2m', name='Action1')
add_trigger = PeriodicTrigger(add_trig_req, add_ndx)

# Rebalance SPX option holdings weekly
rebalance_trig_req = PeriodicTriggerRequirements(start_date=start_date, end_date=end_date, frequency='1w')
# Need to resolve an option before rebalancing it. In this case we want to rebalance the option we are starting with
with PricingContext(start_date):
    spx_opt.resolve()
rebalance_spx = RebalanceAction(spx_opt, 'number_of_options', match_ndx_holding)
rebalance_trigger = PeriodicTrigger(rebalance_trig_req, rebalance_spx)

strategy = Strategy(spx_opt, [add_trigger, rebalance_trigger])

# #### Run the strategy

# In[ ]:


# run backtest daily
GE = GenericEngine()
backtest = GE.run_backtest(strategy, start=start_date, risks=[Price], end=end_date, frequency='1b',
                           show_progress=True)

# #### See total no. of options by underlier

# In[ ]:


ts = backtest.strategy_as_time_series()

# In[ ]:


ndx = ts[~ts.index.get_level_values('Instrument Name').str.contains('spx')]
ndx = ndx.groupby('Pricing Date').agg({('Static Instrument Data', 'number_of_options'): ['sum']})

# In[ ]:


spx = ts[~ts.index.get_level_values('Instrument Name').str.contains('ndx')]
spx = spx.groupby('Pricing Date').agg({('Static Instrument Data', 'number_of_options'): ['sum']})

# In[ ]:


import matplotlib.pyplot as plt
plt.plot(ndx['Static Instrument Data']['number_of_options']['sum'],label='NDX')
plt.plot(spx['Static Instrument Data']['number_of_options']['sum'],label='SPX')

plt.legend(prop={'size': 15})

# In[ ]:


# View Mark to Market
pd.DataFrame({'Generic backtester': backtest.result_summary[Price]}).plot(figsize=(10, 6), title='Mark to market')

# In[ ]:


# View Performance
pd.DataFrame({'Generic backtester': backtest.result_summary['Cumulative Cash'] + backtest.result_summary[Price]}).plot(figsize=(10, 6), title='Performance')
