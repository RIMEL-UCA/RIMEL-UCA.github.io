#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import date

from gs_quant.common import PayReceive
from gs_quant.instrument import IRSwap, IRSwaption
from gs_quant.markets.portfolio import Portfolio
from gs_quant.session import Environment, GsSession
from gs_quant.common import SwapSettlement

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None,
              client_secret=None, scopes=('run_analytics',))

# In[ ]:


swaptions = Portfolio()

# you don't need to specify any parameters to get a valid trade.  All properties have defaults
swaptions.append(IRSwaption())

# In[ ]:


# get list of properties of an interest rate swaption
# Many of these properties overlap with the IRSwap properties (outlined in example '010001_swap_trade_construction')
IRSwaption.properties()

# In[ ]:


# pay_or_receive can be a string of 'pay', 'receive', 'straddle' (an option strategy
# where you enter into both a payer and receiver) or the PayReceive enum
# relates to whether you expect to pay/receive fixed for the underlying. default is 'straddle'
swaptions.append(IRSwaption(pay_or_receive=PayReceive.Receive))
swaptions.append(IRSwaption(pay_or_receive='Receive'))

# In[ ]:


# expiration_date is the date the option expires and may be a tenor relative
# to the active PricingContext.pricing_date or a datetime.date, default is '10y'
swaptions.append(IRSwaption(expiration_date='6m'))
swaptions.append(IRSwaption(expiration_date=date(2022, 2, 12)))

# In[ ]:


# strike is the rate at which the option can be exercised
# It also represents the interest rate on the fixed leg of the swap if the swaption expires ITM. Defaults to Par Rate (ATM).
# Can be expressed as 'ATM', 'ATM+25' for 25bp above par, a-100 for 100bp below par, 0.01 for 1%
swaptions.append(IRSwaption(strike='ATM'))
swaptions.append(IRSwaption(strike='ATM+50'))
swaptions.append(IRSwaption(strike='a-100'))
swaptions.append(IRSwaption(strike=.02))

# In[ ]:


# effective_date is the start date of the underlying swap and may be a tenor relative
# to the expiration_date or a datetime.date. Default is spot dates from expiration
# For example, for a swaption w/ notional_currency as GBP, spot date is T+0, so effective_date = expiration_date.
# for a swaption w/ notional_currency USD, spot is T+2 days and the effective_date is 2b after expiration_date
swaptions.append(IRSwaption(effective_date='5b'))
swaptions.append(IRSwaption(effective_date=date(2031, 2, 12)))

# In[ ]:


# An IRSwaption's strike will resolve to an IRSwap's fixed_rate if the swaps' paramaters match and
# the swaption's effective_date is equivalent to the swap's effective_date
s = IRSwap(notional_currency='GBP', effective_date='10y')
swp = IRSwaption(notional_currency='GBP', expiration_date = '10y', effective_date='0b')
print(s.fixed_rate*100)
print(swp.strike*100)

# In[ ]:


# settlement is the settlement convention of the swaption and can be a string of:
# 'Phys.CLEARED' (enter into cleared swap), 'Cash.PYU' (PYU - Par Yield Unadjusted,
# cash payment calculated with PYU), 'Physical' (enter into a uncleared swap), 
# 'Cash.CollatCash' (collateralized, cash settled at expiry) or the SwapSettlement enum
swaptions.append(IRSwaption(settlement=SwapSettlement.Phys_CLEARED))
swaptions.append(IRSwaption(settlement='Cash.PYU'))

# In[ ]:


# premium is the amount to be exchanged for the option contract. A positive premium will have a 
# negative impact on the PV. premium is a default is  0.
swaptions.append(IRSwaption(premium=1e4))

# In[ ]:


# premium_payment_date is the date when premium is exchanged.
# premium_payment_date can be a datetime.date or a tenor. defaulted to spot dates from the
# PricingContext.pricing_date
swaptions.append(IRSwaption(premium_payment_date='5d'))
swaptions.append(IRSwaption(premium_payment_date=date(2020, 2, 13)))

# In[ ]:


# in some markets, the convention is for premium to be exchanged at expiration
# this can be expressed by changing the premium_payment_date to the swaption's expiration_date
swaptions.append(IRSwaption(premium_payment_date='5y', expiration_date='5y'))

# In[ ]:


swaptions.price()
