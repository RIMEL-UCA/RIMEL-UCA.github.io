#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.instrument import FXVolatilitySwap

# In[ ]:


from gs_quant.session import GsSession
# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics'))

# In[ ]:


# get list of properties of an fx vol swap
FXVolatilitySwap.properties()

# In[ ]:


swap = FXVolatilitySwap(pair='GBPUSD', last_fixing_date='3m', buy_sell='Buy')
swap.resolve()
print(swap.strike_vol*100)

# In[ ]:


#forward starting vol swap
fwd_6m6m_swap = FXVolatilitySwap(pair='USDJPY', first_fixing_date='6m', last_fixing_date='1y', buy_sell='Buy')
fwd_6m6m_swap.resolve()
print(fwd_6m6m_swap.as_dict())

# **Fixing frequency** defaults to 'Daily/Business Days' but many other conventions are supported such as:
# * 'Daily/All Days'
# * 'Daily/Holiday'
# * 'Daily/MonToSat'
# * 'Daily/Weighted Lagging'
# 

# In[ ]:


weighted_swap = FXVolatilitySwap(pair='GBPUSD', last_fixing_date='1y', buy_sell='Buy', fixing_frequency='Daily/Weighted Lagging')

# **Fixing source** defaults to 'WM Company LDN 4pm Mid' but other options include:
# * 'GS NYC 3PM'
# * 'WM Company NYC 10am Mid'
# * 'BFIX LDN 4PM'
# * 'WMC Company LDN 1pm Mid'
# 
# We also have bloomberg options as well such as 'BFIX TKO 3pm-m'. 
# We're happy to confirm the syntax if you are looking for a fixing source that is not mentioned here.
# 

# In[ ]:


bfix = FXVolatilitySwap(pair='GBPUSD', last_fixing_date='1y', buy_sell='Buy', fixing_source='BFIX LDN 4PM')
print(bfix.strike_vol)

# **Strike vol** defaults to ATM but the following solvers are supported:
# 
# * zero cost vol : price=0
# * % Vol = {%}N (i.e. '2N')
# 

# In[ ]:


#zero cost vol
zero_swap = FXVolatilitySwap(pair='USDNOK', last_fixing_date='3m', buy_sell='Sell', strike_vol='price=0')
zero_swap.resolve()
zero_swap.price()

# In[ ]:


#1.25*atm
swap_25 = FXVolatilitySwap(pair='EURUSD', last_fixing_date='3m', buy_sell='Sell', strike_vol='1.25N')
swap_25.resolve()
print(swap_25.strike_vol*100)

# In[ ]:


# current spot
swap_spot = FXVolatilitySwap(pair='EURUSD', last_fixing_date='3m', buy_sell='Sell', strike_vol='S')
swap_spot.resolve()
print(swap_spot.strike_vol*100)

# In[ ]:


# current spot
swap_spot = FXVolatilitySwap(pair='EURUSD', last_fixing_date='3m', buy_sell='Sell', strike_vol='S')
swap_spot.resolve()
print(swap_spot.strike_vol*100)

# In[ ]:


# current spot
swap_spot = FXVolatilitySwap(pair='EURUSD', last_fixing_date='3m', buy_sell='Sell', strike_vol='S')
swap_spot.resolve()
print(swap_spot.strike_vol*100)
