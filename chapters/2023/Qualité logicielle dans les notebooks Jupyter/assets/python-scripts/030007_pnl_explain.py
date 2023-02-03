#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.session import Environment, GsSession

client_id = None # Supply your application id
client_secret = None # Supply your client secret

GsSession.use(Environment.PROD, client_id, client_secret, ('run_analytics',))

# #### Create a dummy portfolio

# In[ ]:


from gs_quant.instrument import IRSwap, IRSwaption
from gs_quant.markets.portfolio import Portfolio

swap = IRSwap(notional_currency='EUR', termination_date='10y', pay_or_receive='Pay')
swaption = IRSwaption(notional_currency='EUR', termination_date='10y', expiration_date='1y', pay_or_receive='Receive')

portfolio = Portfolio((swap, swaption))
portfolio.resolve()

# #### Calculate PnlExplain from 1 week ago (5 business days) to today, compare with the dollar price difference and examine the breakdown

# In[ ]:


from gs_quant.datetime.date import business_day_offset
from gs_quant.markets import CloseMarket, PricingContext, close_market_date
from gs_quant.risk import DollarPrice, PnlExplain

to_date = close_market_date()

# 5 business days ago
from_date = business_day_offset(to_date, -5)

# A risk measure for calculating PnlExplain from that date
explain = PnlExplain(CloseMarket(date=to_date))

# Calculate PnlExplain and dollar price from 1 week ago
with PricingContext(pricing_date=from_date):
    result = portfolio.calc((DollarPrice, explain))
    
# Calculate dollar price with the "to" market but "from" pricing date
with PricingContext(pricing_date=from_date, market=CloseMarket(date=to_date)):
    to_market_price = portfolio.dollar_price()

# Calculate dollar price with the "to" market and pricing date
with PricingContext(pricing_date=to_date):
    to_price = portfolio.dollar_price()

# Compute the time component (PnlExplain does not do this)
time_value = to_price.aggregate() - to_market_price.aggregate()
    
print(f'Dollar price difference: {to_price.aggregate() - result[DollarPrice].aggregate():.0f}')
print(f'Pnl explain + time value total: {result[explain].aggregate().value.sum() + time_value:.0f}')
print(f'Pnl explain total: {result[explain].aggregate().value.sum():.0f}')
print(f'Time value total: {time_value:.0f}')

# Show the PnlExplain breakdown
explain_all = result[explain].aggregate()
explain_all[explain_all.value.abs() > 1.0].round(0)

# In[ ]:



