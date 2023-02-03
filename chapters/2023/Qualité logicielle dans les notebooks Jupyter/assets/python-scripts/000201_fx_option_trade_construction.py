#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.session import Environment, GsSession
from gs_quant.instrument import FXOption
from gs_quant.markets.portfolio import Portfolio
from datetime import date
import pandas as pd

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))
# get list of properties of an fx option
FXOption.properties()
# In[ ]:


# in this example we will construct and price a portfolio of FXOptions
fx_options = Portfolio()

# you don't need to specify any parameters to get a valid trade.  All properties have defaults
fx_options.append(FXOption())

# In[ ]:


# buy_sell indicates whether the option is bought (long) or sold (short) 
# It can be represented by the BuySell enum or a string 
fx_options.append(FXOption(buy_sell='Buy'))

from gs_quant.common import BuySell
fx_options.append(FXOption(buy_sell=BuySell.Sell))

# In[ ]:


# put_call indicates whether the option is a put or call; it can be represented by the OptionType enum or a string
fx_options.append(FXOption(option_type='Put'))

from gs_quant.common import OptionType
fx_options.append(FXOption(option_type=OptionType.Call))

# In[ ]:


# pair is the FXOption's underlying currency pair. It is a string of two ccy iso codes, optionally separated 
# with a space (' '). The first currency is the base (transaction) currency and the second is the quote currency

# In this case, base currency is 'EUR' and quote currency is 'GBP'
# This option gives the purchasor the option to buy EUR at expiration
fx_options.append(FXOption(buy_sell=BuySell.Buy, option_type=OptionType.Call, pair='EUR GBP'))

# Here, base currency is 'GBP' and quote currency is 'USD'
# This option gives the purchasor the option to sell GBP at expiration
fx_options.append(FXOption(buy_sell=BuySell.Buy, option_type=OptionType.Put, pair='GBPUSD'))

# In[ ]:


# notional_currency is the notional amount's denominated currency
# It can be a Currency enum or a string. By default, notional_currency will match the base currency

# In this case, notional_currency of a EURGBP Call is EUR. 
from gs_quant.common import Currency
fx_options.append(FXOption(option_type = OptionType.Call, pair='EUR GBP', notional_currency = Currency.EUR ))

# In[ ]:


# notional_amount is the quantity of the base currency to be exchanged in the future; it can be a string (eg: '100mm')
# or a double (10e8). By definition, notional_amount_other_currency is the notional amount in the quoted currency. It can be a Currency enum or a string 
# by default, notional_amount_other_currency will be the product of 'notional_amount' and 'strike_price'

# Here, a EURGBP Call has a notional of 10mm EUR
fx_options.append(FXOption(option_type = OptionType.Call, pair='EUR GBP', notional_currency = Currency.EUR, 
                           notional_amount='10m'))

# In[ ]:


"""
strike_price is the exchange rate stuck for the FXOption's underlying currency pair, which can be specified by a double 
or a string. If a string is used, it represents a relative value. When the trade is resolved, we solve for the strike_price 

The specific solver keys are: 
    - 'S'    - current spot rate
    - 'F'    - forward
    - 'ATM'  - At the Money 
    - 'ATMF' - At the Money Forward
    - 'D'    - Delta Strikes
    - 'P'    - Premium

You can use these keys to strike_price with the following formats: 
    - For S, F, ATM, ATMF: 's*1.1', 'F+10%', '1.05*ATMF+.01'
    - For Delta Strikes, specify the option delta: '25D', '20D-.01', etc.
    - You can also solve for Premium: P=<target premium>, P=<target premium> P=,<target>%, PPct=<target> 

"""

# Here the option is bought to purchase 95mm EUR at expiration  
fx_options.append(FXOption(buy_sell=BuySell.Buy, pair='EURGBP', option_type='Call', notional_amount=10e8, 
                           strike_price=.95))

# Here the option is sold to sell 100k of EUR at the current spot rate on expiration
fx_options.append(FXOption(buy_sell=BuySell.Sell, pair='EURGBP', option_type=OptionType.Put, notional_amount='100k', 
                           strike_price='S'))

# The option is sold to purchase AUD at the Forward rate + 1%
fx_options.append(FXOption(buy_sell=BuySell.Sell, pair='AUDJPY', option_type=OptionType.Call, strike_price='F+1%'))

# In[ ]:


# notional_amount is the quantity of the base currency to be exchanged in the future; it can be a string (eg: '100mm')
# or a double (10e8). By definition, notional_amount_other_currency is the notional amount in the quoted currency. It 
# can be a Currency enum or a string 
# by default, notional_amount_other_currency will be the product of 'notional_amount' and 'strike_price'

# Here, a EURGBP Call has a notional of 10mm EUR
fx_options.append(FXOption(option_type = OptionType.Call, pair='EUR GBP', notional_currency = Currency.EUR, 
                           notional_amount='10m'))

# In[ ]:


# method_of_settlement indicates whether the option is cash or physically settled and can be either the 
# OptionSettlementMethod Enum or a string 
from gs_quant.common import OptionSettlementMethod
fx_options.append(FXOption(method_of_settlement=OptionSettlementMethod.Cash))

# In[ ]:


# premium_currency is the premium amount's denominated currency
# It can be a Currency enum or a string. By default, premium_currency will match the base currency

# In this case, premium_currency of a EURGBP Put is EUR 
fx_options.append(FXOption(option_type = OptionType.Put, pair='EUR GBP', premium_currency = Currency.EUR ))

# premium_payment_date is the date the premium is exchanged. It can either be a date or string
# It can be set to 's' which indicates spot premium
fx_options.append(FXOption(option_type = OptionType.Call, pair='EUR GBP', premium_payment_date='spot' ))

# or set to 'fwd' or 'forward' to indicate forward premium
fx_options.append(FXOption(option_type = OptionType.Call, pair='EUR GBP', premium_payment_date='fwd' ))

# premium is the price of the option contract. It can be a float or a string
fx_options.append(FXOption(option_type = OptionType.Call, premium= -5e6))

# It is possible to solve for the strike_price based on a certain Premium
# The below resolves the strike_price such that the option premium is -£5mm
fx_options.append(FXOption(pair='AUD JPY', option_type = OptionType.Call, strike_price= 'P=-5mm', premium_currency=Currency.AUD))

# The below resolves the strike_price such that the option premium is 5% of the total amount exchanged if the option
# is exercised
fx_options.append(FXOption(pair = 'AUD JPY', option_type = OptionType.Call, strike_price= 'Premium=5%', premium_currency=Currency.JPY))

# In[ ]:


pd.DataFrame(fx_options.price())
