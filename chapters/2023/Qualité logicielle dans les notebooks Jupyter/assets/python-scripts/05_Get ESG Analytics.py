#!/usr/bin/env python
# coding: utf-8

# # Pull Marquee ESG Analytics with GS Quant
# 
# First get your portfolio and choose the date for which you'd like ESG data:

# In[ ]:


import datetime as dt
import warnings

from gs_quant.api.gs.esg import ESGMeasure
from gs_quant.markets.portfolio_manager import PortfolioManager
from gs_quant.session import GsSession, Environment
from IPython.display import display

GsSession.use(Environment.PROD)
warnings.filterwarnings("ignore", category=RuntimeWarning)

portfolio_id= 'MPWQQ8B05FKPCCH6'
date = dt.date(2021, 9, 1)

pm = PortfolioManager(portfolio_id)

# Then get any of the following ESG Analytics:

# In[ ]:


print('Portfolio Summmary:')
display(pm.get_esg_summary(pricing_date=date))

# In[ ]:


print('Quintile Breakdown:')
display(pm.get_esg_quintiles(
    measure=ESGMeasure.ES_PERCENTILE,
    pricing_date=date
))

# In[ ]:


print('Sector Breakdown:')
display(pm.get_esg_by_sector(
    measure=ESGMeasure.ES_PERCENTILE,
    pricing_date=date
))

# In[ ]:


print('Region Breakdown:')
display(pm.get_esg_by_region(
    measure=ESGMeasure.ES_PERCENTILE,
    pricing_date=date
))

# In[ ]:


print(f'Top Ten Ranked:')
display(pm.get_esg_top_ten(
    measure=ESGMeasure.ES_PERCENTILE,
    pricing_date=date
))

# In[ ]:


print(f'Bottom Ten Ranked:')
display(pm.get_esg_bottom_ten(
    measure=ESGMeasure.ES_PERCENTILE,
    pricing_date=date
))

