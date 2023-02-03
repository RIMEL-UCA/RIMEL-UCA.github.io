#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime as dt

from dateutil.relativedelta import relativedelta

from gs_quant.datetime.date import prev_business_date
from gs_quant.instrument import EqOption, OptionType, OptionStyle
from gs_quant.markets import PricingContext, HistoricalPricingContext
from gs_quant.markets.portfolio import Portfolio
from gs_quant.risk import EqDelta, EqGamma, EqVega, EqSpot, DollarPrice
from gs_quant.session import Environment, GsSession

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


# This example uses instrument resolving.  For external users this requires a reference spot override using a market scenario when resolving
from gs_quant.risk import MarketDataPattern, MarketDataShock, MarketDataShockType, MarketDataShockBasedScenario
eq_spot_scenario = MarketDataShockBasedScenario(
    shocks={
        MarketDataPattern('Eq', '.SPX', 'Reference Spot', mkt_quoting_style='Price'): MarketDataShock(MarketDataShockType.Override, 3900)
    }
)

# ### Create Eq Options Portfolios
# 
# Let's now create a put-spread and a butterfly using `EqOption`. The underlier is entered as Reuters Instrument Code and 
# can be one of .SPX, .STOXX50E, .FTSE, .N225. 
# Strike can be specified as value, percent or at-the-money e.g. 62.5, 95%, ATM-25, ATMF.
# Other parameters we can instantiate can be uncovers by pressing `shift+tab` on `EqOption`. 

# In[ ]:


put_1 = EqOption('.SPX', expiration_date = '3m', strike_price = '95%', option_type = OptionType.Put, 
                 option_style = OptionStyle.European, buy_sell = 'Buy')
put_2 = EqOption('.SPX', expiration_date = '3m', strike_price = '105%', option_type = OptionType.Put, 
                 option_style = OptionStyle.European, buy_sell = 'Sell')
put_spread = Portfolio((put_1, put_2))

itm_call = EqOption('.SPX', expiration_date = '3m', strike_price = '90%', option_type = OptionType.Call, 
                    option_style=OptionStyle.European, buy_sell = 'Buy')
otm_call = EqOption('.SPX', expiration_date = '3m', strike_price = '110%', option_type = OptionType.Call, 
                    option_style = OptionStyle.European, buy_sell = 'Buy')
atm_call = EqOption('.SPX', expiration_date = '3m', strike_price = 'atm', option_type = OptionType.Call, 
                    number_of_options = 2, option_style = OptionStyle.European, buy_sell = 'Sell')
butterfly = Portfolio((itm_call, otm_call, atm_call))

# ### Compute greeks for both our portfolios at inception and at expiry
# 
# Let's now compute greeks on our portfolios. For an exhaustive list of supported metrics for `EqOption` please refer to 
# the [Measures](https://developer.gs.com/docs/gsquant/guides/Pricing-and-Risk/measures/) guide.
# 

# In[ ]:


trade_date = dt.date.today() - relativedelta(months = 3)
trade_date = prev_business_date(trade_date)

# apply spot override when resolving"
with PricingContext(pricing_date = trade_date), eq_spot_scenario:
    put_spread.resolve(in_place = True)
    ps_greeks = put_spread.calc((EqDelta, EqGamma, EqVega))
    butterfly.resolve(in_place = True)
    b_greeks = butterfly.calc((EqDelta, EqGamma, EqVega))

expiry = prev_business_date(put_1.expiration_date)

with PricingContext(pricing_date = expiry):
    ps_exp = put_spread.calc((EqDelta, EqGamma, EqVega))
    b_exp = butterfly.calc((EqDelta, EqGamma, EqVega))
    

# In[ ]:


print(f'Put Spread Delta at inception: {ps_greeks[EqDelta].aggregate():.0f}')
print(f'Put Spread Gamma at inception: {ps_greeks[EqGamma].aggregate():.0f}')
print(f'Put Spread Vega at inception: {ps_greeks[EqVega].aggregate():.0f}')

print(f'Butterfly Delta at inception: {b_greeks[EqDelta].aggregate():.0f}')
print(f'Butterfly Gamma at inception: {b_greeks[EqGamma].aggregate():.0f}')
print(f'Butterfly Vega at inception: {b_greeks[EqVega].aggregate():.0f}')

# ### Now let's look at our portfolios' PV vs Spot price
# 
# Using `EqSpot` we can track the performance of the index vs our portfolio.

# In[ ]:


with HistoricalPricingContext(start = trade_date, end = expiry):
    ps_perf = put_spread.calc((EqSpot, DollarPrice))
    b_perf = butterfly.calc((EqSpot, DollarPrice))
    

# In[ ]:


import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

ps_perf = ps_perf.aggregate()
fig, ax1 = plt.subplots(figsize = (10, 6))
plt.figure(figsize = (24, 20))
color='tab:blue'
ax1.set_ylabel('Spot', color = color)
ax1.plot(ps_perf[EqSpot], color = color)
ax1.set_title('Put Spread')
ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('$PV', color = color)
ax2.plot(ps_perf[DollarPrice], color = color)

plt.show()

# In[ ]:


b_perf = b_perf.aggregate()
fig, ax1 = plt.subplots(figsize=(10, 6))
plt.figure(figsize=(24, 20))
color='tab:blue'
ax1.set_ylabel('Spot', color=color)
ax1.plot(b_perf[EqSpot], color=color)
ax1.set_title('Butterfly')
ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('$PV', color=color)
ax2.plot(b_perf[DollarPrice], color=color)

plt.show()
