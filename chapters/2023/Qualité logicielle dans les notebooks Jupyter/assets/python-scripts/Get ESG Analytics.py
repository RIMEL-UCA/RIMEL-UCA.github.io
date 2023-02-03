#!/usr/bin/env python
# coding: utf-8

# # Pull Marquee ESG Analytics with GS Quant
# 
# Portfolio ESG Analytics are driven by the Goldman Sachs GIR SUSTAIN Headline Metrics dataset. Using data from third-party ESG data providers, the GIR SUSTAIN team has identified what it believes to be the most relevant Environmental & Social (E&S) exposures for a company’s sector and Governance (G) exposure relative to its region and the global SUSTAIN ESG universe. These metrics form the framework through which to evaluate corporate ESG engagement. We caution at the outset that this data should not be viewed as signalling 'good' or 'bad' ESG or overall performance. Instead, the framework highlights a company’s peer-relative performance on the key ESG metrics that the team believes to be the most relevant for that sector as a basis for further analysis.
# 
# Please note that for companies with multiple share classes, SUSTAIN ESG data will only be provided for the primary share class. If other share classes are included in your portfolio, the ESG metrics will be represented as N/A and aggregate metrics will not include them.

# ## Step 1: Authenticate and Initialize Your Session
# 
# First you will import the necessary modules and add your client id and client secret.

# In[ ]:


import datetime as dt

from gs_quant.api.gs.esg import ESGMeasure, ESGCard
from gs_quant.markets.portfolio_manager import PortfolioManager
from gs_quant.session import GsSession, Environment
from IPython.display import display

client = None
secret = None
scopes = None

## External users must fill in their client ID and secret below and comment out the line below

#client = 'ENTER CLIENT ID'
#secret = 'ENTER CLIENT SECRET'
#scopes = ('read_product_data')

GsSession.use(
    Environment.PROD,
    client_id=client,
    client_secret=secret,
    scopes=scopes
)

# ## Step 2: Define Your Entity
# 
# ESG can be pulled from any object that inherits from the `PositionedEntity` class, such as `PortfolioManager`, `Index` or `Basket`. In this example, we will get ESG data for a Marquee portfolio.

# In[ ]:


pm = PortfolioManager('ENTER PORTFOLIO ID')

# Now we will walk through some ESG analytics you can pull from your portfolio. For all of the following examples, if no date is provided, the results will be as of the last previous business day.
# 
# ## Get Portfolio's Weight Averaged ESG Percentile Values
# 
# Easily pull your weighted average percentile values of your portfolio on a given day:

# In[ ]:


print('Portfolio Summmary:')
display(pm.get_esg_summary(pricing_date=dt.date(2021, 9, 1)))

# ### Quintile Breakdown
# 
# If you want to see what percent of your portfolio has an ESG percentile value between 0-20%, 20%-40%,
# and so on, it's possible to pull that information for your requested ESG measure:

# In[ ]:


print('Quintile Breakdown:')
display(pm.get_esg_quintiles(
    measure=ESGMeasure.ES_PERCENTILE,
    pricing_date=dt.date(2021, 9, 1)
))

# ### Breakdown by Region and Sector
# 
# View your portfolio's weighted average ESG percentile value by GIR SUSTAIN subsector...

# In[ ]:


print('Sector Breakdown:')
display(pm.get_esg_by_sector(
    measure=ESGMeasure.ES_PERCENTILE,
    pricing_date=dt.date(2021, 9, 1)
))

# ### Breakdown by Region and Sector
# 
# and by GIR SUSTAIN region:

# In[ ]:


print('Region Breakdown:')
display(pm.get_esg_by_region(
    measure=ESGMeasure.ES_PERCENTILE,
    pricing_date=dt.date(2021, 9, 1)
))

# ### Top Ten and Bottom Ten Ranked
# 
# Get a list of your ten positions with the highest ESG percentile values...

# In[ ]:


print(f'Top Ten Ranked:')
display(pm.get_esg_top_ten(
    measure=ESGMeasure.ES_PERCENTILE,
    pricing_date=dt.date(2021, 9, 1)
))

# and a list of your ten positions with the lowest ESG percentile values:

# In[ ]:


print(f'Bottom Ten Ranked:')
display(pm.get_esg_bottom_ten(
    measure=ESGMeasure.ES_PERCENTILE,
    pricing_date=dt.date(2021, 9, 1)
))

# ### Quick Tip!
# If you would like to pull data for multiple measures at once, you can leverage the `get_all_esg_data` function:

# In[ ]:


aggregated_esg_results = pm.get_all_esg_data(
    measures=[ESGMeasure.ES_PERCENTILE, ESGMeasure.G_PERCENTILE],
    cards=[ESGCard.QUINTILES, ESGCard.MEASURES_BY_SECTOR],
    pricing_date=dt.date(2021, 9, 1)
)
print(aggregated_esg_results)

# ### You're all set, Congrats!
