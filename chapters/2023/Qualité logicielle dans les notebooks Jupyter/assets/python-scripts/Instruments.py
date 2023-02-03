#!/usr/bin/env python
# coding: utf-8

# Examples require an initialized GsSession and relevant entitlements. `run_analytics` scope is required for the functionality covered in this tutorial. External clients need to substitute thier own client id and client secret below. Please refer to <a href="https://developer.gs.com/docs/gsquant/guides/Authentication/2-gs-session/"> Sessions</a> for details.

# In[1]:


from gs_quant.session import GsSession
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics',))

# ## What is an Instrument
# 
# [`Instrument`](https://developer.gs.com/docs/gsquant/api/instrument.html) is a class that inherits from the [`Priceable`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.base.Priceable.html)
# class and is used to represent financial objects that can be priced, such as derivative instruments.
# 
# [`Priceable`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.base.Priceable.html#gs_quant.base.Priceable/) exposes several methods common to all instruments, such as [`as_dict()`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.base.Priceable.html#gs_quant.base.Priceable.as_dict) which returns a dictionary of all the public, non-null properties and values and
# [`calc(risk_measure)`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.base.Priceable.html#gs_quant.base.Priceable.calc) which is used to evaluate various risk measures. More about the latter in the [Measures](https://developer.gs.com/docs/gsquant/guides/Pricing-and-Risk/measures/) guide.
# 
# `gs-quant` offers a number of [`Instrument`](https://developer.gs.com/docs/gsquant/api/instrument.html#instruments/) implementations, such as [equity options](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.instrument.EqOption.html#gs_quant.instrument.EqOption/), [interest rate swaptions](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.instrument.IRSwaption.html#gs_quant.instrument.IRSwaption/) and [commodity swaps](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.instrument.CommodSwap.html).
# Please refer to [supported instruments](#supported-instruments) for a list of externally supported instruments.

# ## How to Create an Instrument
# 
# Let's now create an instance of an instrument implementation.
# For this example, we will create an Interest Rate Swaption, which the [`IRSwaption`](/gsquant/api/classes/gs_quant.instrument.IRSwaption.html#gs_quant.instrument.IRSwaption/) class implements.
# 
# We will start by importing [`IRSwaption`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.instrument.IRSwaption.html#gs_quant.instrument.IRSwaption/) from the [`Instrument`](https://developer.gs.com/docs/gsquant/api/instrument.html#instruments/) package as well as `PayReceive`, `Currency` to represent commonly used constants.
# 

# In[1]:


from gs_quant.instrument import IRSwaption
from gs_quant.common import PayReceive, Currency

# We will now instantiate an [`IRSwaption`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.instrument.IRSwaption.html#gs_quant.instrument.IRSwaption/) object. Note that all [`Instruments`](https://developer.gs.com/docs/gsquant/api/instrument.html#instruments/), including [`IRSwaption`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.instrument.IRSwaption.html#gs_quant.instrument.IRSwaption/), take some non-keyworded required arguments (first 3 in this example),
# and some optional keyworded arguments (\*\*kwargs). If the optional arguments are not supplied, default market conventions will be used as described in each
# instrument's signature. Signatures can be found in the [the Instrument Package](/gsquant/api/instrument.html#instruments/) by clicking
# on the desired instrument.

# In[2]:


swaption = IRSwaption(PayReceive.Receive, '5y', Currency.USD, expiration_date='13m', strike='atm+40', notional_amount=1e8)

# We can now use the [`as_dict()`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.base.Priceable.html#gs_quant.base.Priceable.as_dict) method inherited from [`Priceable`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.base.Priceable.html#) to view all public non-null properties and values of this swaption instance.

# In[5]:


swaption.as_dict()

# ## Instrument Resolution
# 
# The above output shows only the inputs specified, many of which are relative (i.e. expiration date, strike). Calling the [`resolve()`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.base.Priceable.html#gs_quant.base.Priceable.resolve) method will resolve these
# parameters to absolute values as well as fill in any defaulted parameters using the [`PricingContext`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.markets.PricingContext.html). Please refer to the above mentioned [Instrument Package](/gsquant/api/instrument.html#instruments/) for each instrument's available parameters and to the [Pricing Context guide](/gsquant/guides/Pricing-and-Risk/pricing-context) for further details on [`PricingContext`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.markets.PricingContext.html).

# In[ ]:


swaption.resolve()
swaption.as_dict()

# Note [`resolve()`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.base.Priceable.html#gs_quant.base.Priceable.resolve) will change the state of the instrument object. In the code snippet above, calling [`resolve()`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.base.Priceable.html#gs_quant.base.Priceable.resolve) mutates several specified relative parameters, for example:
# 
# - `expiration_date`, specified as '13m', was resolved to '2020-11-02'
# - `strike` specified as 'atm+40', was resolved to '0.017845989434194357'
# 
# [`resolve()`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.base.Priceable.html#gs_quant.base.Priceable.resolve) will also add any unspecified default parameters - note the additions when calling [`as_dict()`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.base.Priceable.html#gs_quant.base.Priceable.as_dict) before and after [`resolve()`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.base.Priceable.html#gs_quant.base.Priceable.resolve). For example:
# 
# - 'fixed_rate_frequency': '6m'
# - 'premium_payment_date': '2019-10-04'
# 
# Accessing any of the unspecified parameters on the unresolved swaption will resolve the swaption in place.
# 
# Additionally, as discussed in the [measures guide](/gsquant/guides/Pricing-and-Risk/measures), if [`resolve()`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.base.Priceable.html#gs_quant.base.Priceable.resolve) is not called prior to calling [`price()`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.base.Priceable.html#gs_quant.base.Priceable.price) or calculating risk, the instrument object will be copied and resolved on the fly without mutating the original swaption object.
# 
# The preferred behavior may depend on the [`PricingContext`](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.markets.PricingContext.html#gs_quant.markets.PricingContext/) - more on this in the [Pricing Context](/gsquant/guides/Pricing-and-Risk/pricing-context) guide.

# 
# ## Supported Instruments
# 
# Below are the instruments covered, names they are referred to as in gs_quant and brief definitions as well as links to
#  the technical documentation for each. Each instrument corresponds to a model maintained by Goldman Sachs Securities Division.
# 
# | Instrument                            | gs_quant name                                                                                                               | Description                                                                                                                                 |
# | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------|
# | Eq Option                             | [EqOption](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.instrument.EqOption.html)                             | An option on an underlying equity security                                                                                                  |
# | FX Forward                            | [FXForward](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.instrument.FXForward.html)                           | An exchange of cashflows in different currencies at a determined future time                                                                |
# | FX Option                             | [FXOption](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.instrument.FXOption.html)                             | An option on an FX Forward                                                                                                                  |
# | FX Binary                             | [FXBinary](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.instrument.FXBinary.html)                             | An option where the buyer receives a fixed amount if a certain currency pair fixes above or below a specified level on a specified date.    |
# | FX Multi Cross Binary                 | [FXMultiCrossBinary](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.instrument.FXMultiCrossBinary.html)         | An option where the buyer receives a fixed amount if each of the currency pairs fixes above or below a specified level on a specified date. |
# | FX Volatility Swap                    | [FXVolatilitySwap](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.instrument.FXVolatilitySwap.html)             | An exchange of cashflows based on the realized volatility of the underlying FX cross and a pre-determined fixed volatility level             |
# | Inflation Swap                        | [InflationSwap](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.instrument.InflationSwap.html)                   | A zero coupon vanilla inflation swap of fixed vs floating cashflows adjusted to inflation rate                                              |
# | Interest Rate Swap                    | [IRSwap](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.instrument.IRSwap.html)                                 | A vanilla interest rate swap of fixed vs floating cashflows in the same currency                                                            |
# | Interest Rate Basis Swap              | [IRBasisSwap](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.instrument.IRBasisSwap.html)                       | An exchange of cashflows from different interest rate indices in the same currency                                                          |
# | Interest Rate Xccy Swap               | [IRXccySwap](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.instrument.IRXccySwap.html)                         | An exchange of cashflows from different interest rate indices in different currencies                                                       |
# | Interest Rate Xccy Swap Fix Fix       | [IRXccySwapFixFix](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.instrument.IRXccySwapFixFix.html)             | An exchange of fixed cashflows in different currencies                                                                                      |
# | Interest Rate Xccy Swap Fix Float     | [IRXccySwapFixFlt](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.instrument.IRXccySwapFixFlt.html)             | A vanilla interest rate swap of fixed vs floating cashflows in different currencies                                                         |
# | Interest Rate Swaption                | [IRSwaption](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.instrument.IRSwaption.html)                         | An option to enter into a vanilla interest rate swap of fixed vs floating cashflows                                                         |
# | Interest Rate Cap                     | [IRCap](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.instrument.IRCap.html)                                   | An instrument in which the buyer receives payments at the end of each period in which the interest rate exceeds the agreed strike price     |
# | Interest Rate Floor                   | [IRFloor](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.instrument.IRFloor.html)                               | An instrument in which the buyer receives payments at the end of each period in which the interest rate is below the agreed strike price    |
# | Interest Rate CMS Option              | [IRCMSOption](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.instrument.IRCMSOption.html)                       | An option on a single date where the payoff is based on the CMS rate                                                                        |
# | Interest Rate CMS Option Strip        | [IRCMSOptionStrip](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.instrument.IRCMSOptionStrip.html)             | A strip of CMS Options                                                                                                                      |
# | Interest Rate CMS Spread Option       | [IRCMSSpreadOption](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.instrument.IRCMSSpreadOption.html)           | An option on a single date where the payoff is dependent on the spread of two CMS rates compared to the strike of the option.               |
# | Interest Rate CMS Spread Option Strip | [IRCMSSpreadOptionStrip](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.instrument.IRCMSSpreadOptionStrip.html) | A strip of CMS Spread Options                                                                                                               |
# 
# Note that `IRDelta` is additional available for Interest Rate Futures and Bond Futures upon request.
# 

# #### Disclaimer
# This website may contain links to websites and the content of third parties ("Third Party Content"). We do not monitor, review or update, and do not have any control over, any Third Party Content or third party websites. We make no representation, warranty or guarantee as to the accuracy, completeness, timeliness or reliability of any Third Party Content and are not responsible for any loss or damage of any sort resulting from the use of, or for any failure of, products or services provided at or from a third party resource. If you use these links and the Third Party Content, you acknowledge that you are doing so entirely at your own risk.
# 
