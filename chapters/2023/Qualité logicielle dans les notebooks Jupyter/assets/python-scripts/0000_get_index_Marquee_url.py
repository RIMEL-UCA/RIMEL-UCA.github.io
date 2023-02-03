#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.markets.index import Index
from gs_quant.session import Environment, GsSession

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None)

# In[ ]:


index = Index.get('GSXXXXXX')  # substitute input with any identifier for an Index
index.get_url()                # url to view index on Marquee

# *Have any other questions? Reach out to the [Marquee STS team](mailto:gs-marquee-sts-support@gs.com)!*
