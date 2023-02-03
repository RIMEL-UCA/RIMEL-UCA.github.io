#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.session import Environment, GsSession
from gs_quant.instrument import IRSwaption

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None,
              client_secret=None, scopes=('run_analytics',))

# In[ ]:


# Construct a midcurve swaption: 6m 2y1y
# expiration_date='6m' - option expires in 6m
# effective_date='2y' - swap starting 2y after expiry (2.5y after trade)
# termination_date='1y' - swap tenor is 1y (swap matures 3.5y post trade)
midcurve = IRSwaption(pay_or_receive='Pay', expiration_date='6m',
                      effective_date='2y', termination_date='1y')
midcurve.price()

# In[ ]:


# expiration_date can also be specified as IMM date
midcurve = IRSwaption(pay_or_receive='Pay', expiration_date='IMM1',
                      effective_date='2y', termination_date='1y')
midcurve.price()
