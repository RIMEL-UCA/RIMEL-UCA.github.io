#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.common import PayReceive
from gs_quant.instrument import IRSwaption
from gs_quant.session import Environment, GsSession

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None,
              client_secret=None, scopes=('run_analytics',))

# In[ ]:


# A swaption straddle represents the purchase of a payer swaption and
# receiver swaption with the same strike and expiration_date
straddle = IRSwaption(pay_or_receive='Straddle')

# In[ ]:


# The price of a straddle is sum of the price of the payer and receiver swaptions
payer = IRSwaption(pay_or_receive=PayReceive.Pay)
receiver = IRSwaption(pay_or_receive=PayReceive.Receive)

print(straddle.price())
print(payer.price() + receiver.price())
