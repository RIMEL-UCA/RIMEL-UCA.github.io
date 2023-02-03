#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.models.risk_model import FactorRiskModel, FactorType
from gs_quant.session import Environment, GsSession

import datetime as dt

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('read_product_data run_analytics',))

# In[ ]:


model = FactorRiskModel.get("BARRA_USFAST")

# ### Getting factors and factor categories
# Once you have the risk model, you can fetch the factors and factor categories defined for it

# In[ ]:


start_date = dt.date(2020,1, 1)
end_date   = dt.date(2021,1, 1)

# In[ ]:


model_factor_data = model.get_factor_data(start_date=start_date, end_date=end_date)

# In[ ]:


# Fetch only factors
model.get_factor_data(start_date=start_date, end_date=end_date, factor_type=FactorType.Factor)

# In[ ]:


# Fetch only factor categories
model.get_factor_data(start_date=start_date, end_date=end_date, factor_type=FactorType.Category)

# You can also fetch factors of specific categories

# In[ ]:


model.get_factor_data(category_filter=['Style'], start_date=start_date, end_date=end_date)
