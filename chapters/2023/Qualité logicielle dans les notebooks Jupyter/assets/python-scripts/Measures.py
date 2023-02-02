#!/usr/bin/env python
# coding: utf-8

# Examples require an initialized GsSession and relevant entitlements. `run_analytics` scope is required for the functionality covered in this tutorial. External clients need to substitute thier own client id and client secret below. Please refer to <a href="https://developer.gs.com/docs/gsquant/guides/Authentication/2-gs-session/"> Sessions</a> for details.

# In[ ]:


from gs_quant.session import GsSession
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics',))

# ## Measures
# 
# A measure is a metric that can be calculated on an instrument, like a dollar price. Below is a table of supported measures and their definitions.
# 
# | Measure                                                                                                              | Definition                                                                                                                                                                                            |
# | -------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | [Annuity*](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.Annuity)                                  | Annuity of instrument in local currency (unless currency specified in parameters)                                                                                                                   |
# | [Dollar Price](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.DollarPrice)                        | Price of the instrument in US Dollars                                                                                                                                                                 |
# | [Price*](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.Price)                                     | Price of the instrument  (In local currency unless currency specified in parameters)                                                                                                                                                              |
# | [ForwardPrice](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.ForwardPrice)                       | Price of the instrument at expiry in the local currency                                                                                                                                               |
# | [CDDelta](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.CDDelta)                                 | Change in option Dollar Price relative to the change in underlying index Dollar Price, due to a 1bp shift in the underlying index spread                                                              |
# | [CDGamma](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.CDGamma)                                 | Change in option Delta relative to the change in underlying index Delta, due to a 1bp shift in the underlying index spread                                                                            |
# | [CDTheta](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.CDTheta)                                 | Change in option Dollar Price over one day                                                                                                                                                            |
# | [CDVega](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.CDVega)                                   | Change in option Dollar Price due to a 1bp shift in the implied volatility of the underlying index                                                                                                    |
# | [EqDelta](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.EqDelta)                                 | Change in Dollar Price (USD present value) due to individual 1% move in the spot price of underlying equity security                                                                                |
# | [EqGamma](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.EqGamma)                                 | Change in EqDelta for a 1% move in the price of the underlying equity security                                                                                                                      |
# | [EqVega](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.EqVega)                                   | Change in Dollar Price (USD present value) due to individual 1bp moves in the implied volatility of the underlying equity security                                                                    |
# | [FXDelta*](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.FXDelta)                                 | Dollar Price sensitivity of the instrument to a move in the underlying spot such that dSpot \* FXDelta = PnL                                                                                          |
# | [FXGamma*](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.FXGamma)                                 | FXDelta sensitivity of the instrument to a move in the underlying spot such that dSpot \* FXGamma = dDelta                                                                                            |
# | [FXAnnualImpliedVol](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.FXAnnualImpliedVol)                                   | FX daily implied volatility (in percent)                                                               |
# | [FXAnnualATMImpliedVol](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.FXAnnualATMImpliedVol)                                   | FX daily implied volatility (in basis points)                                                               |
# | [FXSpot](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.FXSpot)                                   | FX spot reference                                                                                                                                                                                     |
# | [FXVega*](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.FXVega)                                   | Change in Dollar Price due to a 1 vol move in the implied volatility of ATM instruments used to build the volatility surface                                                                |
# | [IRBasis*](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.IRBasis)                                 | Change in Dollar Price (USD present value) due to individual 1bp moves in the interest rate instruments used to build the basis curve(s)                                                    |
# | [IRDelta*](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.IRDelta)                                 | Change in Dollar Price (USD present value) due to individual 1bp moves in the interest rate instruments used to build the underlying discount curve                                         |
# | [IRGamma*](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.IRGamma)                                 | Change in aggregated IRDelta for a aggregated 1bp shift in the interest rate instruments used to build the underlying discount curve                                                               |
# | [IRVega*](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.IRVega)                                   | Change in Dollar Price (USD present value) due to individual 1bp moves in the implied volatility (IRAnnualImpliedVol) of instruments used to build the volatility surface                   |
# | [IRAnnualImpliedVol](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.IRAnnualImpliedVol)           | Interest rate annual implied volatility (in percent)                                                                                                                                                  |
# | [IRAnnualATMImpliedVol](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.IRAnnualATMImpliedVol)     | Interest rate annual implied at-the-money volatility (in percent)                                                                                                                                     |
# | [IRDailyImpliedVol](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.IRDailyImpliedVol)             | Interest rate daily implied volatility (in basis points)                                                                                                                                              |
# | [IRSpotRate](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.IRSpotRate)                           | Interest rate at-the-money spot rate (in percent)                                                                                                                                                     |
# | [IRFwdRate](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.IRFwdRate)                             | Interest rate par rate (in percent)                                                                                                                                                                   |
# | [IRXccyDelta*](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.IRXccyDelta)                             | Change in Price due to 1bp move in cross currency rates.                                                                                                                                                                |
# | [InflationDelta*](https://developer.gs.com/docs/gsquant/api/risk.html#gs_quant.risk.InflationDelta)                             | Change in Price due to 1bp move in inflation curve.                                                                                                                                                                  |
# 
# Note - * indicates parameterised risk measures. See Parameterised Risk Measure section below.
# 

# 
# ## Calculating Price Measures
# 
# Let's price an instrument. For information on how to define an instrument, please refer to the [Instruments](https://developer.gs.com/docs/gsquant/guides/Pricing-and-Risk/instruments/) guide.
# 
# Note, below we resolve the swaption parameters that will be used to price the swaption, thereby mutating the swaption object. If [`resolve()`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.base.Priceable.html#gs_quant.base.Priceable.resolve), is
# not called prior to calling [`price()`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.base.Priceable.html#gs_quant.base.Priceable.price), the object will be copied and resolved on the fly without mutating the original swaption object.
# The preferred behavior may depend on the [`PricingContext`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.markets.PricingContext.html) - more on this in the [Pricing Context](https://developer.gs.com/docs/gsquant/guides/Pricing-and-Risk/pricing-context/) guide.
# 
# [`price()`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.base.Priceable.html#gs_quant.base.Priceable.price) method will return price in the local currency.
# 

# ### IR Swaps

# In[ ]:


from gs_quant.instrument import IRSwap
from gs_quant.common import PayReceive, Currency
# Creating Swaps - spot starting, forward starting
swap = IRSwap(PayReceive.Receive, '10y', 'GBP', fixed_rate='atm+50', notional_amount=1e8)  # running
swap_fwd_start = IRSwap(PayReceive.Pay, '5y', 'EUR', fixed_rate='atm+20', effective_date='3y')  # fwd starting

# In[ ]:


# View Instrument w/ specified relative parameters
swap_fwd_start.as_dict()

# In[ ]:


# Resolve Instrument, View fixed parameters
swap_fwd_start.resolve()
swap_fwd_start.as_dict()

# ### IR Swaptions

# In[ ]:


from gs_quant.instrument import IRSwaption
from gs_quant.common import PayReceive, Currency

swaption = IRSwaption(PayReceive.Receive, '5y', Currency.USD, expiration_date='13m', strike='atm+40', notional_amount=1e8)
swaption.resolve()
swaption.price() # local is USD

# All instruments can also priced in dollars.

# In[ ]:


swaption.dollar_price() # USD price

# ## Calculating Risk Measures
# 
# We can also calculate risk measures for the defined instrument. Please refer to [the Measures Guide](https://developer.gs.com/docs/gsquant/guides/Pricing-and-Risk/measures/) for the supported risk measures.
# Calling [`calc(risk_measure)`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.base.Priceable.html#gs_quant.base.Priceable.calc) calculates the value of the risk measure and can return a float, a dataframe or a future thereof, depending on how [`PricingContext`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.markets.PricingContext.html) is being used - more on this in the [Pricing Context guide](https://developer.gs.com/docs/gsquant/guides/Pricing-and-Risk/pricing-context/).
# 
# Calculate a scalar value like implied volatility:

# In[ ]:


import gs_quant.risk as risk

swaption.calc(risk.IRAnnualImpliedVol) * 10000

# Calculate a structured value like vega:

# In[ ]:


swaption.calc(risk.IRVega)

# Calculate IR Delta (Aggregated Scalar and Ladder) for Swaps

# In[ ]:


from gs_quant.common import AggregationLevel

ir_risk = swap.calc((risk.IRDelta(aggregation_level=AggregationLevel.Type), risk.IRDelta(aggregation_level=AggregationLevel.Type, currency='local'), risk.IRDelta))

# Print Risks
print(ir_risk[risk.IRDelta(aggregation_level=AggregationLevel.Type)])
print(ir_risk[risk.IRDelta(aggregation_level=AggregationLevel.Type, currency='local')])
print(ir_risk[risk.IRDelta])

# Calculate a conditional risk measure. Show IRDelta Ladder only where exposure >1e-2

# In[ ]:


ird_ladder=ir_risk[risk.IRDelta]
print(ird_ladder[abs(ird_ladder.value)>1e-2])

# See [measures](#Measures) table for information on units.

# ### Parameterised Risk Measures
# 
# Some risk measures now support extra parameters
# 
# You can now specify which currency you want the price of an instrument to be expressed in

# In[ ]:


eur_swap = IRSwap(PayReceive.Pay, '5y', 'EUR', fixed_rate='atm+20') 
price = eur_swap.price(currency='PLN')
print(f'{price} {list(price.unit.keys())[0]}')

# For some finite difference risk measures (noted * in the table above), you can now pass in the specifics of the calculation methodology.
# 
# 
# **Parameters supported in finite difference risk measures**
# 
# |Parameter name	          |Type	                                             |Description|
# |-------------------------|--------------------------------------------------|------------------------------------------------------------
# |currency	              |string	                                         |Currency of risk result|
# |aggregation_level        |gs_quant.target.common.AggregationLevel	         |Level of aggregate shift|
# |local_curve	          |bool                                  	         |Change in Price (present value in the denominated currency)|
# |finite_difference_method |gs_quant.target.common.FiniteDifferenceParameter	 |Direction and dimension of finite difference|
# |mkt_marking_options      |gs_quant.target.common.MktMarkingOptions          |Market marking mode|
# |bump_size	              |float	                                         |Bump size|
# |scale_factor	          |float	                                         |Scale factor|

# In[ ]:


print(swap.calc(risk.IRDelta))

# In[ ]:


# Calculate delta using an aggregate 1bp shift on the asset level and using change in Price only in the denominated currency
print(swap.calc(risk.IRDelta(aggregation_level=AggregationLevel.Type, currency='local')))

# #### Disclaimer
# This website may contain links to websites and the content of third parties ("Third Party Content"). We do not monitor, review or update, and do not have any control over, any Third Party Content or third party websites. We make no representation, warranty or guarantee as to the accuracy, completeness, timeliness or reliability of any Third Party Content and are not responsible for any loss or damage of any sort resulting from the use of, or for any failure of, products or services provided at or from a third party resource. If you use these links and the Third Party Content, you acknowledge that you are doing so entirely at your own risk.
