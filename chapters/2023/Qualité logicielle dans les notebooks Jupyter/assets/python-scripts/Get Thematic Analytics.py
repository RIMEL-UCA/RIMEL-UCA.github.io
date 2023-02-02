#!/usr/bin/env python
# coding: utf-8

# # Pull Portfolio Thematic Analytics with GS Quant
# 
# ## Permission Prerequisites
# 
# To execute all the code in this tutorial, you will need the following application scopes:
# - **read_product_data**
# - **read_financial_data**
# - **run_analytics** (must be requested)
# 
# ## Authenticate and Initialize Your Session
# 
# First you will import the necessary modules and add your client id and client secret.

# In[ ]:



import datetime as dt

from IPython.display import display

from gs_quant.markets.portfolio_manager import PortfolioManager
from gs_quant.markets.securities import SecurityMaster, AssetIdentifier
from gs_quant.session import GsSession, Environment
import pandas as pd

client = None
secret = None
scopes = None

## External users must fill in their client ID and secret below and comment out the line below

#client = 'ENTER CLIENT ID'
#secret = 'ENTER CLIENT SECRET'
#scopes = ('read_product_data read_financial_data run_analytics',)

GsSession.use(
    Environment.PROD,
    client_id=client,
    client_secret=secret,
    scopes=scopes
)

print('GS Session initialized.')

# ## Get Portfolio Thematic Report

# In[ ]:


thematic_report = PortfolioManager('ENTER PORTFOLIO ID').get_thematic_report()

print(f'Thematic report found with ID: {thematic_report.id}')

# ## Current Thematic Exposure to All Baskets
# 
# Once your thematic report is scheduled as of the latest business day, you can view your portfolio's current exposure and
# beta to every flagship basket in our thematic factor model in just a few lines of code:

# In[ ]:


thematic_exposures = thematic_report.get_all_thematic_exposures(start_date=thematic_report.latest_end_date,
                                                                end_date=thematic_report.latest_end_date)

pd.set_option('display.max_colwidth', 0)

display(thematic_exposures)

# ## Get Thematic Exposure Breakdown for a Basket
# 
# Interested in a more granular breakdown of your exposure to a particular basket? Pull your thematic breakdown by asset
# on a desired date:

# In[ ]:


date = dt.date(2022, 4, 6)

basket = SecurityMaster.get_asset('GSXUSTAY', AssetIdentifier.TICKER)

thematic_breakdown = thematic_report.get_thematic_breakdown(date, basket.get_marquee_id())

display(thematic_breakdown)

# ## Historical Thematic Exposure
# 
# You can also pull the historical change in your thematic exposure to a basket:

# In[ ]:


historical_exposures = thematic_report.get_all_thematic_exposures(start_date=thematic_report.earliest_start_date,
                                                                end_date=thematic_report.latest_end_date,
                                                                basket_ids=[basket.get_marquee_id()])[['Date', 'Thematic Exposure']]


historical_exposures.plot(title='Historical Exposure to GSXUSTAY')

# *Other questions? Reach out to the [Portfolio Analytics team](mailto:gs-marquee-analytics-support@gs.com)!*
