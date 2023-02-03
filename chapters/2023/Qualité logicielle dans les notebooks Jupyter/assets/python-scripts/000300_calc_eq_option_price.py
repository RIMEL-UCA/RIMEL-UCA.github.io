#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import date
from gs_quant.instrument import EqOption, OptionType, OptionStyle, UnderlierType
from gs_quant.session import Environment, GsSession

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))


# In[ ]:


# create a .STOXX50E 3m call option striking at-the-money spot
eq_option = EqOption('.STOXX50E', expiration_date='3m', strike_price='ATMS', option_type=OptionType.Call,
                     option_style=OptionStyle.European)

# In[ ]:


# calculate local price and dollar price
print('Local price:      {:,.4f}'.format(eq_option.price()))
print('Dollar price:     {:,.4f}'.format(eq_option.dollar_price()))

# #### Underlier Syntax
# 
# The underlier accepts an underlier as a RIC or BBID identifier. The default is RIC.
# 
# | Syntax  | Defintion           |
# |---------|---------------------|
# |  'RIC'  | Reuters identifier      |
# |  'BBID'  | Bloomberg identifier        |

# In[ ]:


# resolve using a Bloomberg ID
eq_option_bbid = EqOption('SX5E', underlier_type=UnderlierType.BBID, expiration_date='3m', strike_price='ATMS', option_type=OptionType.Call,
                     option_style=OptionStyle.European)

eq_option_bbid.resolve()
eq_option_bbid.as_dict()

# #### Strike Syntax
# The strike_price syntax allows for an int or a string. The absolute level can be specified using an integer.
# 
# The following solver keys using a string format are accepted: 
# 
# | Syntax  | Defintion           |
# |---------|---------------------|
# |  '%'   | Percent of Spot      |
# |  'ATMS'  | At the Money        |
# |  'ATMF' | At the Money Forward|
# |  'D'    | Delta Strikes       |
# |  'P'    | Premium             |
# 
#     - For ATM, ATMF: '1.05*ATMF+.01'
#     - For Delta Strikes, specify the option delta: '25D', '20D-.01', etc.
#     - You can also solve for Premium: P=,<target>% 

# In[ ]:


# resolve with strike at 110% of spot
eq_atm_solver = EqOption('.STOXX50E', expiration_date='3m', strike_price='ATMS+10%', option_type=OptionType.Put,
                     option_style=OptionStyle.European)

eq_atm_solver.resolve()
eq_atm_solver.strike_price

# In[ ]:


# resolve with strike at 94.5% of spot
eq_spot_pct = EqOption('.STOXX50E', expiration_date='3m', strike_price='94.5%', option_type=OptionType.Put,
                     option_style=OptionStyle.European)

eq_spot_pct.resolve()
eq_spot_pct.strike_price

# In[ ]:


# resolve with strike at spot minus 10
eq_atmf_solver = EqOption('.STOXX50E', expiration_date='1m', strike_price='ATMF-10', option_type=OptionType.Put,
                     option_style=OptionStyle.European)

eq_atmf_solver.resolve()
eq_atmf_solver.strike_price

# In[ ]:


# resolve with strike solving for 10% premium 
eq_10x = EqOption('.STOXX50E', expiration_date='6m', strike_price='P=10%', option_type=OptionType.Put,
                     option_style=OptionStyle.European)

eq_10x.resolve()
eq_10x.strike_price
