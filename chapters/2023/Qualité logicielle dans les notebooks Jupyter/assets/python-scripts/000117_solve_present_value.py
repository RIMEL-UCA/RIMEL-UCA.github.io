#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.instrument import IRSwaption
from gs_quant.markets.portfolio import Portfolio
from gs_quant.session import GsSession

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics',)) 

# ### Solve for Present Value
# Using 1x2 receiver structure - solve for strike on second leg such that pv of leg2 = pv of leg1
# 
# This example takes advantage of intra-portfolio formulae. You could do this in two API requests with:
# ```python
# swaption_1 = IRSwaption( ... )
# price = swaption_1.price()
# swaption_2 = IRSwaption( ..., strike=f'{price}/pv', ... )
# zero_cost_portfolio = Portfolio((swaption_1, swaption_2))
# ```

# In[ ]:


zero_cost_portfolio = Portfolio((
    IRSwaption('Receive', '30y', 'USD', notional_amount=10e6, expiration_date='3m',strike='atmf', buy_sell='Buy', name='30y_buy'),
    IRSwaption('Receive', '30y', 'USD', notional_amount=20e6, expiration_date='3m', strike='=solvefor([30y_buy].risk.Price,pv)', buy_sell='Sell')
))

# see the strikes and prices
print([s.strike * 1e4 for s in zero_cost_portfolio.resolve(in_place=False)])
print([s for s in zero_cost_portfolio.price()])
