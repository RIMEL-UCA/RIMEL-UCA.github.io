#!/usr/bin/env python
# coding: utf-8

# ## Delta Hedging
# 
# ### Summary 
# 
# Being short volatility hasn't been profitable in this period of extreme implied and realized volatility movements but may be an interesting entry point for some. 
# 
# In this note I take a further look at this strategy and extend it with delta hedging to understand how it can impact performance.
# 
# Each day I sell a 1m10y straddle (like last time) - but this time I also trade a swap with a matched effective date and termination date to hedge my delta. Each day I unwind the previous day's swap and trade into a new one.
# 
# I examine premium collected at inception, payout on option expiry and mark-to-market over the life of the trade to compare the two strategies.
# 
# Look out for future publications where I will build on this analysis further by adding transaction costs and analyzing performance accross strategies.
# 
# The content of this notebook is split into:
# * [1 - Let's get started with gs quant](#1---Let's-get-started-with-gs-quant)
# * [2 - Create portfolio](#2---Create-portfolio)
# * [3 - Grab the data](#3---Grab-the-data)
# * [4 - Putting it all together](#4---Putting-it-all-together)

# ### 1 - Let's get started with gs quant
# Start every session with authenticating with your unique client id and secret. If you don't have a registered app, create one [here](https://marquee.gs.com/s/developer/myapps/register). `run_analytics` scope is required for the functionality covered in this example. Below produced using gs-quant version 0.8.108.

# In[ ]:


from gs_quant.session import GsSession
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics',))

use_batch = True

# ### 2 - Create portfolio
# Just like in our last analysis, let's start by creating a portfolio with a rolling strip of straddles. For each date in our date range (start of 2019 through today), we will construct a 1m10y straddle and include it in our portfolio.

# In[ ]:


from gs_quant.markets import HistoricalPricingContext, PricingContext
from gs_quant.markets.portfolio import Portfolio
from gs_quant.common import Currency, PayReceive
from gs_quant.instrument import IRSwaption
import datetime as dt

start_date = dt.datetime(2020, 12, 1).date()
end_date = dt.datetime.today().date()

# create and resolve a new straddle on every day of the pricing context
with HistoricalPricingContext(start=start_date, end=end_date, show_progress=True): 
    f = IRSwaption(PayReceive.Straddle, '10y', Currency.USD, expiration_date='1m', 
                   notional_amount=1e8, buy_sell='Sell').resolve(in_place=False)

# put resulting swaptions in a portfolio
result = f.result().items()
portfolio = Portfolio([v[1] for v in sorted(result)])

# I will now convert the portfolio to a dataframe, extend it with trade dates and remove any instruments with a premium payment date after today.

# In[ ]:


frame = portfolio.to_frame()
frame.index = frame.index.droplevel(0)

# extend dataframe with trade dates
trade_dates = {value:key for key, value in result}
frame['trade_date'] = frame.apply(lambda x: trade_dates[x.name], axis=1)

# filter any swaptions with premium date larger than today
frame = frame[frame.premium_payment_date < dt.datetime.today().date()]
frame.head(2)

# ### 3 - Grab the Data
# 
# Now the fun part - we need to calculate a lot of datapoints for this backtest. 
# 
# For each straddle, we need to define a new swap every day and price it the following day when we unwind it. This means about 36,000 points (~300 instruments * 30 days * 4 measures (swaption price, swaption delta, swap price, swap delta)).
# 
# Like last time I will compute as much as I can asyncrously and keep track of the futures for each measure. 
# 
# Introducing a high-level `PricingContext` to batch requests can improve speed as well. Note just using `PricingContext` will improve speed but `batch=True` can  add efficiency.
# 
# To learn more about async and other compute controls and how to use them, please see our [pricing context guide](https://developer.gs.com/docs/gsquant/guides/Pricing-and-Risk/pricing-context/). 

# I'll start by getting the prices and delta for the swaptions first.

# In[ ]:


from gs_quant.risk import IRDeltaParallel

# insert columns in our frame to track the futures
frame['so_price_f'] = len(frame) * [None]
frame['so_delta_f'] = len(frame) * [None]

with PricingContext(is_batch=use_batch, show_progress=True):
    for inst, row in frame.iterrows():
        with HistoricalPricingContext(start=row.trade_date, 
                                      end=min(row.expiration_date, dt.datetime.today().date()), 
                                      is_async=True):
            so_price = inst.price()
            so_delta = inst.calc(IRDeltaParallel)        

        frame.at[inst, 'so_price_f'] = so_price
        frame.at[inst, 'so_delta_f'] = so_delta

# Easy enough. I will now do the same for the swaps which I will use to delta hedge. Note instead of pricing the same already resolved swaption each day, here I create and price a new swap each day which will reflect that's day's ATM rate and matches the effective date and termination date of the corresponding swaption.

# In[ ]:


from gs_quant.instrument import IRSwap

# insert columns in our frame to track the futures
frame['s_f'] = len(frame) * [None]
frame['s_delta_f'] = len(frame) * [None]

with PricingContext(is_batch=use_batch, show_progress=True):
    for inst, row in frame.iterrows():
        swap = IRSwap(PayReceive.Pay, row.termination_date, Currency.USD, 
                      effective_date=row.effective_date, fixed_rate='ATMF', notional_amount=1e8)

        with HistoricalPricingContext(start=row.trade_date, 
                                      end=min(row.expiration_date, dt.datetime.today().date()), 
                                      is_async=True):
            # track the resolved swap - we will need to price it when we unwind following day
            s = swap.resolve(in_place=False)
            s_delta = swap.calc(IRDeltaParallel)

        frame.at[inst, 's_f'] = s
        frame.at[inst, 's_delta_f'] = s_delta

# In the above request, we created a new resolved swaption for each day but we still need to price it the following day when we unwind it. In the below, I collect the resolved swaps from the previous request and price lagged 1 day - that is, the following day.

# In[ ]:


from gs_quant.markets import PricingContext
import pandas as pd

swaps = pd.concat([pd.Series(row.s_f.result(), name=row.name) for _, row in frame.iterrows()], 
                      axis=1, sort=True).shift(periods=1)
g = {}

with PricingContext(is_batch=use_batch, show_progress=True):
    for date, row in swaps.iterrows():
        with PricingContext(date, is_async=True):
            prices = {k: p if isinstance(p, float) else p.price() for k, p in row.iteritems()}
        g[date] = prices
        
swap_prices = pd.DataFrame(g).T

# Finally, let's collect all the points and do some arithmetic to create a timeseries for each swaption. I will create two frames - one for the simple vol selling strategy and one taking into account the changing delta hedge.

# In[ ]:


from gs_quant.timeseries import *

not_delta_hedged = []
delta_hedged = []

for inst, row in frame.iterrows():
    # collect all the results
    total_result = pd.concat([row.so_price_f.result(), row.so_delta_f.result(), 
                              pd.Series({k: v.result() for k, v in swap_prices[inst].iteritems() 
                                         if not isinstance(v, float)}), 
                              row.s_delta_f.result()], axis=1, sort=True)
    total_result.columns = ['swaption_prices', 'swaption_delta', 'swap_bought_prices', 'swap_sold_delta']
    
    # today's hedge notional will be the ratio of prior day's swaption/swap delta ratio - that's
    # how much of the swap we bought to hedge so will use it to scale unwind PV of the swap today
    total_result['hedge_notional'] = -(total_result.swaption_delta/total_result.swap_sold_delta).shift(periods=1)
    total_result = total_result.fillna(0)
    
    # scale the umwind PV of prior day's swap hedge
    total_result['swap_pos'] = total_result['hedge_notional'] * total_result['swap_bought_prices']
    
    # add to swaption price to get total performance cutting off last time due to the lag
    swaption_pl = diff(total_result['swaption_prices'], 1).fillna(0)
    total_result['total_pv'] = swaption_pl + total_result['swap_pos']
    
    not_delta_hedged.append(pd.Series(swaption_pl[:-1], name=inst))
    delta_hedged.append(pd.Series(total_result['total_pv'][:-1], name=inst))

# ### 4 - Putting it all together
# Now, let's combine all the results to look at the impact delta hedging makes on the strategy. Unsurprisingly, the delta hedged version provides protection to tail events like March 2020.

# In[ ]:


not_dh = pd.concat(not_delta_hedged, axis=1, sort=True).fillna(0).sum(axis=1).cumsum()
dh = pd.concat(delta_hedged, axis=1, sort=True).fillna(0).sum(axis=1).cumsum()

# In[ ]:


comp = pd.DataFrame([dh, not_dh]).T
comp.columns = ['Delta Hedged', 'Not Delta Hedged']
comp.plot(figsize=(10, 6), title='Hedged vs Not Hedged')

# Note that this backtesting doesn't include transaction costs and the implementation is different from how one might hedge in practice (unwinding and trading a new swap every day) but is economically equivalent to layering the hedges (and is cleaner from a calculation perspective).
# 
# Look out for future publications for added transaction costs and ways to quantitatively compare these strategies.

# ### Generic Backtesting Framework

# In[ ]:


from gs_quant.backtests.triggers import PeriodicTrigger, PeriodicTriggerRequirements
from gs_quant.backtests.actions import AddTradeAction, HedgeAction
from gs_quant.backtests.generic_engine import GenericEngine
from gs_quant.backtests.strategy import Strategy
from gs_quant.common import AggregationLevel
from gs_quant.risk import IRDelta, Price

# dates on which actions will be triggered
trig_req = PeriodicTriggerRequirements(start_date=start_date, end_date=end_date, frequency='1b')

# instrument that will be added on AddTradeAction
irswaption = IRSwaption('Straddle', '10y', 'USD', expiration_date='1m', notional_amount=1e8,
                        buy_sell='Sell', name='1m10y')
swap_hedge = IRSwap(PayReceive.Pay, '10y', 'USD', fixed_rate='ATMF', notional_amount=1e8, name='10y_swap')
action_trade = AddTradeAction(irswaption, 'expiration_date')
action_hedge = HedgeAction(IRDelta(aggregation_level=AggregationLevel.Type, currency='local'), swap_hedge, '1b')

# starting with empty portfolio (first arg to Strategy), apply actions on trig_req
triggers = PeriodicTrigger(trig_req, [action_trade, action_hedge])
strategy = Strategy(None, triggers)

# run backtest
GE = GenericEngine()
backtest = GE.run_backtest(strategy, start=start_date, end=end_date, frequency='1b', show_progress=True)

# Note, these strategies won't be exactly the same since the original series. This is because the original series is matching each swaption with a swap and holding both positions while resizing the swap each day. In the generic backtester snippet above, we passed '1b' for hedge holding period. This means that the delta from ALL swaption positions on a given date will be hedged with a single swap which will be replaced each day. Note if instead of '1b', we put '10y' for example (to match swaption tail), the swap hedge will be 'layered' - that is, we will add positive or negative notional each day but not unwind the existing swap hedges.

# In[ ]:


pd.DataFrame({'Generic backtester': backtest.result_summary['Cash'].cumsum() + backtest.result_summary[Price],
              'Original series': dh}).plot(figsize=(10, 6), title='Delta hedged backtest comparison')
