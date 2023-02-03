#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.common import Currency, PayReceive
from gs_quant.instrument import IRSwap
from gs_quant.markets import PricingContext, OverlayMarket, MarketDataCoordinate
from gs_quant.session import GsSession

# In[ ]:


GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics',))

# ### Retrieve market data 

# Let's create a swaption and retrieve the market data our instrument is sensitive to. To do so we can call `market()` on 
# our instrument. This will give us the `OverlayMarket` object which contains the market data used to price our 
# instrument. 

# In[ ]:


swap = IRSwap(PayReceive.Receive, '10y', Currency.EUR, fixed_rate=-0.025)
swap.resolve()
market = swap.market()

print(f'Base price: {swap.price()}')

# Then, using the `market_data` attribute, we can access the market data coordinates and values directly

# In[ ]:


print(f'The value of the coordinate, {market.market_data[0].coordinate} is {market.market_data[0].value}')

# ### Overwrite market data

# We can also amend the market data of our instrument's `OverlayMarket` to pass-in our own market data value. 
# To do so, we simply overwrite the `MarketDataCoordinate` of the instrument `OverlayMarket` to a given value.

# In[ ]:


c_10y = MarketDataCoordinate.from_string('IR_EUR_SWAP_10Y.ATMRATE')

print(f'Current value of the EUR 10yr swap point is {market[c_10y]}')

market[c_10y] = -0.02

print(f'New value of the EUR 10yr swap point is {market[c_10y]}')

with PricingContext(market=market):
    price_f = swap.price()

print(f'New price: {price_f.result()}')

# ... or pass in an new `OverlayMarket` all together! Here we create a bespoke market with our own values for the 3m5y 
# implied volatility and 10y swap rate. Note that the values that are not overwritten will be defaulted to their original 
# value.

# In[ ]:


from gs_quant.instrument import IRSwaption
swaption = IRSwaption(PayReceive.Receive, '5y', Currency.EUR, expiration_date='3m')
swaption.resolve()

print(f'Base price: {swaption.price()}')

vol_3m5y = MarketDataCoordinate.from_string('IR VOL_EUR-EURIBOR-TELERATE_SWAPTION_5Y,3M')
market_data = {c_10y: 0.01, vol_3m5y: 40 / 1e4}
new_market = OverlayMarket(market_data)

with PricingContext(market=new_market):
    price_f = swaption.price()

print(f'Price from new market data: {price_f.result()}')
