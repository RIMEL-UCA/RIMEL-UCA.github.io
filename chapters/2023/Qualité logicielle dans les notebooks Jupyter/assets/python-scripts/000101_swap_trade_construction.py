#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.session import Environment, GsSession
from gs_quant.common import PayReceive, Currency, DayCountFraction, BusinessDayConvention
from gs_quant.target.common import SwapClearingHouse
from gs_quant.instrument import IRSwap
from gs_quant.markets.portfolio import Portfolio
from datetime import date

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


# get list of properties of an interest rate swap
IRSwap.properties()

# In[ ]:


swaps = Portfolio()

# you don't need to specify any parameters to get a valid trade.  All properties have defaults
swaps.append(IRSwap())

# In[ ]:


# pay_or_receive can be a string of 'pay' or 'receive' or the PayReceive enum 
# and relates to paying or receiving the fixed leg.  defaults to a receiver swap
swaps.append(IRSwap(pay_or_receive=PayReceive.Pay))
swaps.append(IRSwap(pay_or_receive='Pay'))

# In[ ]:


# termination_date is the end date of the swap.  It may be a tenor relative to effective_date or a datetime.date.
# defaults to 10y
swaps.append(IRSwap(termination_date=date(2025, 11, 12)))
swaps.append(IRSwap(termination_date='1y'))

# In[ ]:


# notional currency may be a string or the Currency enum.  defaults to USD
swaps.append(IRSwap(notional_currency=Currency.USD))
swaps.append(IRSwap(notional_currency='EUR'))

# In[ ]:


# the effective date is the start date of the swap and may be a tenor relative 
# to the active PricingContext.pricing_date or a datetime.date, default is pricing date
swaps.append(IRSwap(effective_date='1y'))
swaps.append(IRSwap(effective_date=date(2019, 11, 12)))

# In[ ]:


# fixed_rate is the interest rate on the fixed leg of the swap.  Defaults to Par Rate (ATM).  
# Can be expressed as 'ATM', 'ATM+25' for 25bp above par, a-100 for 100bp below par, 0.01 for 1%
swaps.append(IRSwap(fixed_rate='ATM'))
swaps.append(IRSwap(fixed_rate='ATM+50'))
swaps.append(IRSwap(fixed_rate='a-100'))
swaps.append(IRSwap(fixed_rate=0.01))

# In[ ]:


# floating_rate_for_the_initial_calculation_period sets the first fixing on the trade.  
# It should be a float in absolute terms so 0.0075 is 75bp. Defaults to the value derived from fwd curve
swaps.append(IRSwap(floating_rate_for_the_initial_calculation_period=0.0075))

# In[ ]:


# floating rate option is the index that is being observed, defaults to LIBOR style index for each ccy, 
# 'OIS' will give the default overnight index for the notional ccy
swaps.append(IRSwap(notional_currency='USD', floating_rate_option='USD-ISDA-SWAP RATE'))
swaps.append(IRSwap(notional_currency='USD', floating_rate_option='USD-LIBOR-BBA'))
swaps.append(IRSwap(notional_currency='EUR', floating_rate_option='EUR-EONIA-OIS-COMPOUND'))
swaps.append(IRSwap(notional_currency='GBP', floating_rate_option='OIS'))

# In[ ]:


# floating_rate_designated_maturity is the index term.  defaults to the frequency of the floating leg
swaps.append(IRSwap(notional_currency='GBP', floating_rate_designated_maturity='3m'))

# In[ ]:


# floating_rate_spread is a float spread over the index. eg. pay euribor + 1%. defaults to 0
swaps.append(IRSwap(pay_or_receive='receive', notional_currency='EUR', floating_rate_spread=0.01))

# In[ ]:


# floating_rate_frequency is the accrual frequency of the floating leg defined as a tenor.  
# It will drive the floating_rate_designated_maturity if that has not been independently set.  
# Defaults to ccy/tenor market standard defaults
swaps.append(IRSwap(floating_rate_frequency='6m'))

# In[ ]:


# floating_rate_day_count_fraction can be the enum used here or a string.  defaults to ccy market standard defaults
swaps.append(IRSwap(floating_rate_day_count_fraction=DayCountFraction.ACT_OVER_365_ISDA))
swaps.append(IRSwap(floating_rate_day_count_fraction=DayCountFraction._30E_OVER_360))
swaps.append(IRSwap(floating_rate_day_count_fraction='30/360'))
swaps.append(IRSwap(floating_rate_day_count_fraction='ACT/360'))

# In[ ]:


# floating_rate_business_day_convention can be the enum used here a the equivalent string
swaps.append(IRSwap(floating_rate_business_day_convention=BusinessDayConvention.Following))
swaps.append(IRSwap(floating_rate_business_day_convention='Modified Following'))

# In[ ]:


# fee is an amount paid.  A positive fee will have a negative impact on the PV.  Defaults to 0
swaps.append(IRSwap(fee=50000))

# In[ ]:


# you can specify fee currency and fee date.  trades where the fee is paid in a different currency to the
# notional currency are supported.  Default fee currency is notional currency
# fee date can be a datetime.date or a tenor. Default fee date is spot dates from the PricingContext.pricing_date
swaps.append(IRSwap(notional_currency=Currency.GBP, fee=50000, fee_currency=Currency.GBP, fee_payment_date='1y'))
swaps.append(IRSwap(notional_currency=Currency.GBP, fee=1e5, fee_currency=Currency.USD, fee_payment_date=date(2020, 1, 30)))

# In[ ]:


# valid clearinghouses are held in the SwapClearingHouse enum
swaps.append(IRSwap(clearing_house=SwapClearingHouse.LCH))
swaps.append(IRSwap(clearing_house=SwapClearingHouse.EUREX))
swaps.append(IRSwap(clearing_house='CME'))

# In[ ]:


# you can specify a name for a trade.  This has no economic effect but is useful when extracting results
# from a portfolio object
swaps.append(IRSwap(PayReceive.Receive, '5y', 'gbp', name='GBP5y'))

# In[ ]:


swaps.price()

# In[ ]:


# you can express a swap as a dictionary
swap = IRSwap(termination_date='10y', notional_currency='EUR', fixed_rate='ATM+50')
swap_dict = swap.as_dict()
swap_dict

# In[ ]:


# and you can construct a swap from a dictionary
new_swap = IRSwap.from_dict(swap_dict)

# In[ ]:


swap = IRSwap(effective_date='1y')
swap.resolve()
swap.as_dict()
