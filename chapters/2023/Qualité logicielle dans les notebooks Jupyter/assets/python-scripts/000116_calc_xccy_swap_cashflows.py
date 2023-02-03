#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.instrument import IRXccySwap, IRXccySwapFixFix
from gs_quant.risk import Cashflows
from gs_quant.session import Environment, GsSession

# external users should substitute their client id and secret
client_id = None # Supply your application id
client_secret = None # Supply your client secret

GsSession.use(Environment.PROD, client_id=client_id, client_secret=client_secret, scopes=('run_analytics',))

# ### Create a Xccy float vs float Swap and a Xccy fix vs fix swap

# In[ ]:


mtm_swap = IRXccySwap(payer_currency='EUR', receiver_currency='USD', effective_date='3m', termination_date='10y')
fix_swap = IRXccySwapFixFix(payer_currency='EUR', receiver_currency='USD', termination_date='10y', payer_rate=0.01, receiver_rate=0.015)

# ### Compute cashflows for the 10y EURUSD fix swap 

# In[ ]:


cf_fix = fix_swap.calc(Cashflows)
cf_fix.head()

# ### Compute cashflows for the forward starting 10y EURUSD float float swap

# In[ ]:


mtm_swap.calc(Cashflows).head()

# ### Clone the float float swap keeping spread constant but modifying the rate applied at initiation

# In[ ]:


mtm_swap = mtm_swap.clone(initial_fx_rate=1.2, payer_spread=mtm_swap.payer_spread)
mtm_swap.calc(Cashflows).head()
