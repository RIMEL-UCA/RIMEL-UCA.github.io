#!/usr/bin/env python
# coding: utf-8

# ## Systematic Selling
# 
# ### Summary 
# 
# In this note I look at running a simple backtest where I sell a 1m10y straddle each day. I examine premium collected at inception, payout on option expiry and mark-to-market over the life of the trade. 
# 
# Look out for future publications where I will build on this strategy with added delta hedging and add analytics for understanding strategy performance!
# 
# The content of this notebook is split into:
# * [1 - Let's get started with gs quant](#1---Let's-get-started-with-gs-quant)
# * [2 - Create portfolio](#2---Create-portfolio)
# * [3 - Evaluate portfolio historically](#3---Evaluate-portfolio-historically)
# * [4 - Putting it all together](#4---Putting-it-all-together)
# 
# Note, since this notebook was released, we have built a generic backtesting framework to support the same flexibility with significantly less code. See [how you can achieve the same results using this framework at the end of the notebook](#Generic-backtesting-framework).

# ### 1 - Let's get started with gs quant
# Start every session with authenticating with your unique client id and secret. If you don't have a registered app, create one [here](https://marquee.gs.com/s/developer/myapps/register). `run_analytics` scope is required for the functionality covered in this example. Below produced using gs-quant version 0.8.102.

# In[ ]:


from gs_quant.session import GsSession
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics',)) 

# ### 2 - Create portfolio
# Let's create a portfolio with a rolling strip of straddles. For each date in our date range (start of 2019 through today), we will construct a 1m10y straddle and include it in our portfolio.

# In[ ]:


from gs_quant.markets.portfolio import Portfolio
from gs_quant.common import Currency, PayReceive
from gs_quant.instrument import IRSwaption
from gs_quant.markets import HistoricalPricingContext, PricingContext
from datetime import datetime, date

start_date = date(2020, 12, 1)
end_date = datetime.today().date()
with HistoricalPricingContext(start=start_date, end=end_date, show_progress=True): 
    f = IRSwaption(PayReceive.Straddle, '10y', Currency.USD, expiration_date='1m', notional_amount=1e8,
                   buy_sell='Sell').resolve(in_place=False)

# put resulting swaptions in a portfolio
result = f.result().items()
portfolio = Portfolio([v[1] for v in sorted(result)])

# We can use `to_frame` to take a look at our portfolio and resolved instrument details as a dataframe. Let's remove any instruments with premium_payment_date larger than today.

# In[ ]:


frame = portfolio.to_frame()
frame.index = frame.index.droplevel(0)

# extend dataframe with trade dates
trade_dates = {value:key for key, value in result}
frame['trade_date'] = frame.apply(lambda x: trade_dates[x.name], axis=1)

frame = frame[frame.premium_payment_date < datetime.today().date()]
frame.head(3)

# ### 3 - Evaluate portfolio historically
# Let's now evaluate each instrument for the relevant date range (inception to option expiry).
# 
# Note that I use the async flag in pricing context - this is what makes computing 9000 points so fast (~300 instruments * ~30 days)! It sends off a request for 1 month of data for each instrument to be computed in parallel. I keep track of each future in our portfolio frame.
# 
# To learn more about async and other compute controls and how to use them, please see our [pricing context guide](https://developer.gs.com/docs/gsquant/guides/Pricing-and-Risk/pricing-context/). 

# In[ ]:


frame['future'] = len(frame) * [None]

with PricingContext(is_batch=True, show_progress=True):
    for inst, row in frame.iterrows():
        with HistoricalPricingContext(start=row.trade_date, 
                                      end=min(row.expiration_date, datetime.today().date()),
                                      is_async=True):
            pv = inst.price()
        frame.at[inst, 'future'] = pv

# We can now grab all the results and organize them into a dataframe. This call will wait for all the results to come back from the pool so it's as fast as the slowest single request out of the ~300 we sent in the previous step.

# In[ ]:


import pandas as pd

timeseries = pd.concat([pd.Series(row.future.result(), name=row.name) for _, row in frame.iterrows()], axis=1, sort=True)

# ### 4 - Putting it all together
# With the portfolio and historical PV's in hand, let's comb through the data to tease out components we want to track: premium collected, payout on expiry and mark-to-mark of the swaption.

# In[ ]:


import matplotlib.pyplot as plt
from collections import defaultdict
from gs_quant.datetime import business_day_offset

def get_p(df, first=True):
    p_p = df.apply(lambda series: series.first_valid_index() if first else series.last_valid_index())
    g = defaultdict(float)
    for i, r in p_p.items():
        g[r]+=df[i][r]
    return pd.Series(g)

premia = get_p(timeseries)
payoffs = get_p(timeseries, first=False).reindex(timeseries.index).fillna(0)
mtm = timeseries.fillna(0).sum(axis=1)-payoffs

overview = pd.concat([premia.cumsum(), payoffs.cumsum(), mtm], axis=1, sort=False)
overview.columns = ['Premium Received at Inception', 'Paid at Expiry', 'Mark to Market']
overview = overview.sort_index()
overview = overview.fillna(method='ffill')[:business_day_offset(datetime.today().date(), -2)]

overview.plot(figsize=(12, 8), title='Cumulative Payoff, Premium and Mark-to-Market')

# From the above, we can see that premium received at inception varies relative to payout at expiry. We can see when this is the case more clearly by looking at the difference between the two.

# In[ ]:


(overview['Paid at Expiry'] - overview['Premium Received at Inception']).plot(figsize=(12, 8), title='Realized Performance')

# But looking at only premium collected and amount paid out doesn't speak to the volatility of this strategy - let's add mark-to-market in to see that. 

# In[ ]:


p = (overview['Paid at Expiry'] - overview['Premium Received at Inception'] + overview['Mark to Market'])
p.plot(figsize=(12, 8), title='Performance Including Mark to Market')

# As we can see above, since beginning of 2019 this has mostly been a losing strategy although it has worked well 4Q19.
# 
# Stay tuned to futher editions of gs quant for ways to modify this strategy (delta hedging, for example) and to analyze performance of strategies like this one.

# ### Generic Backtesting Framework

# In[ ]:


from gs_quant.backtests.triggers import PeriodicTrigger, PeriodicTriggerRequirements
from gs_quant.backtests.actions import AddTradeAction, HedgeAction
from gs_quant.backtests.generic_engine import GenericEngine
from gs_quant.backtests.strategy import Strategy

# dates on which actions will be triggered
trig_req = PeriodicTriggerRequirements(start_date=start_date, end_date=end_date, frequency='1b')

# instrument that will be added on AddTradeAction
irswaption = IRSwaption(PayReceive.Straddle, '10y', Currency.USD, expiration_date='1m', notional_amount=1e8,
                        buy_sell='Sell', name='1m10y')
actions = AddTradeAction(irswaption, 'expiration_date')

# starting with empty portfolio (first arg to Strategy), apply actions on trig_req
triggers = PeriodicTrigger(trig_req, actions)
strategy = Strategy(None, triggers)

# run backtest
GE = GenericEngine()
backtest = GE.run_backtest(strategy, start=start_date, end=end_date, frequency='1b', show_progress=True)

# In[ ]:


# summarized results
backtest.result_summary

# In[ ]:


pd.DataFrame({'Original series': overview['Paid at Expiry'] - overview['Premium Received at Inception'], 
              'Generic backtester': backtest.result_summary['Cash'].cumsum()}).plot(figsize=(10, 6), 
                                                                                    title='Realized performance comparison')

# In[ ]:


from gs_quant.risk import DollarPrice

pd.DataFrame({'Original series': overview['Mark to Market'], 
              'Generic backtester': backtest.result_summary[DollarPrice]}).plot(figsize=(10, 6),
                                                                          title='Mark to market comparison')

# In[ ]:


pd.DataFrame({'Generic backtester': backtest.result_summary['Cash'].cumsum() + backtest.result_summary[DollarPrice],
              'Original series': p}).plot(figsize=(10, 6), title='Backtest comparison')

# ### Disclaimer
# This website may contain links to websites and the content of third parties ("Third Party Content"). We do not monitor, review or update, and do not have any control over, any Third Party Content or third party websites. We make no representation, warranty or guarantee as to the accuracy, completeness, timeliness or reliability of any Third Party Content and are not responsible for any loss or damage of any sort resulting from the use of, or for any failure of, products or services provided at or from a third party resource. If you use these links and the Third Party Content, you acknowledge that you are doing so entirely at your own risk.
