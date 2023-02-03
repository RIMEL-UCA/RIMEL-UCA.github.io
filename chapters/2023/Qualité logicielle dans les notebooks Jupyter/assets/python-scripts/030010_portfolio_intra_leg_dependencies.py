#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.instrument import IRSwaption, IRSwap
from gs_quant.markets.portfolio import Portfolio
from gs_quant.session import Environment, GsSession
from gs_quant.common import PayReceive, Currency

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# ### Reference other leg properties in instrument fields
# This would be the equivalent of making two requests to resolve the legs serialially
# ```python
# swaption_1y5y = IRSwaption( ... )
# swaption_1y4y = IRSwaption( ... )
# 
# swaption_1y5y.resolve()
# swaption_1y4y.strike = swaption_1y5y.strike
# swaption_1y4y.resolve()
# ```

# In[ ]:


swaption_1y5y = IRSwaption(PayReceive.Pay, '5y', Currency.EUR, expiration_date='1y', strike="atm", name="foo")
swaption_1y4y = IRSwaption(PayReceive.Pay, '4y', Currency.EUR, expiration_date='1y', strike="=[foo].strike + 5bp", name="bar")
port = Portfolio((swaption_1y5y, swaption_1y4y))

# In[ ]:


port.resolve()

# In[ ]:


print(port['foo'].strike * 1e4)
print(port['bar'].strike * 1e4)
