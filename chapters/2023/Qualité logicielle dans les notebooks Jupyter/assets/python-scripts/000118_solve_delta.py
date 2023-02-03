#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.instrument import IRSwaption, IRSwap
from gs_quant.markets.portfolio import Portfolio
from gs_quant.risk import IRDeltaParallel
from gs_quant.session import Environment, GsSession

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# ### Solve for Delta
# Size swap to match swaption delta

# In[ ]:


port = Portfolio((
    IRSwaption('Pay', '5y', 'USD', expiration_date='1y', buy_sell='Sell', name='payer_swaption'), 
    IRSwap('Receive', '5y', 'USD', fixed_rate='atm', notional_amount='=solvefor([payer_swaption].risk.IRDeltaParallel,bp)', name='hedge1')
))
port.resolve()

# In[ ]:


port_delta = port.calc(IRDeltaParallel)

# In[ ]:


# check that delta is approximately equal
port_delta['payer_swaption'] - port_delta['hedge1']
