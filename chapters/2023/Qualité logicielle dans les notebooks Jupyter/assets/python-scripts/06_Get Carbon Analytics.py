#!/usr/bin/env python
# coding: utf-8

# ### Pull Marquee Carbon Analytics Data with GS Quant
# 
# First get your portfolio:

# In[ ]:


import warnings

from IPython.display import display
from gs_quant.markets.portfolio_manager import PortfolioManager
from gs_quant.session import GsSession, Environment

GsSession.use(Environment.PROD)
warnings.filterwarnings("ignore", category=RuntimeWarning)

pm = PortfolioManager('MPMFMQP333M81MAR')

# Then get any of the following Carbon Analytics:

# In[ ]:


print('Data Coverage:')
display(pm.get_carbon_coverage(include_estimates=True))

# In[ ]:


print('Science Based Target and Net Zero Emissions Target Coverage:')
display(pm.get_carbon_sbti_netzero_coverage(include_estimates=True))

# In[ ]:


print('Financed Emissions and Intensity Profile:')
display(pm.get_carbon_emissions(include_estimates=True))

# In[ ]:


print('Financed Emissions by Sector:')
display(pm.get_carbon_emissions_allocation(include_estimates=True))

# In[ ]:


print('Attribution Analysis:')
display(pm.get_carbon_attribution_table(benchmark_id='MA4B66MW5E27U8P32SB', include_estimates=True))

# It's that simple! Compare with your portfolio's [Marquee](https://marquee.gs.com/s/portfolios/MPMFMQP333M81MAR/carbon?benchmark=MA4B66MW5E27U8P32SB&currency=USD&dateFrom=11-Oct-2020&dateTo=11-Oct-2021&includeEstimates=true&normalizeEmissions=false&reportingYear=Latest&useHistoricalData=false) page
