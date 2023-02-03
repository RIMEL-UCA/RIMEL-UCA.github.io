#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.instrument import IRFixedLeg, IRFloatLeg, IRSwap
from gs_quant.markets import PricingContext
from gs_quant.session import Environment, GsSession

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics',)) 

# In[ ]:


fixed_leg = IRFixedLeg('Buy', 
                       fixed_rate=0.05, 
                       notional_currency='USD', 
                       termination_date='5y', 
                       notional_amount=100e6)

float_leg = IRFloatLeg('Sell', 
                       notional_currency='USD', 
                       termination_date='5y', 
                       floating_rate_spread=40/1e4, 
                       notional_amount=100e6)

swap = IRSwap(pay_or_receive='Receive', 
              termination_date='5y', 
              notional_currency='USD', 
              fixed_rate=0.05, 
              floating_rate_spread=40/1e4, 
              principal_exchange='None', 
              notional_amount=100e6)

# In[ ]:


with PricingContext(csa_term='USD-SOFR'):
    fixed_price = fixed_leg.price()
    float_price = float_leg.price()
    swap_price = swap.price()
    
print(fixed_price.result() + float_price.result())
print(swap_price.result())
