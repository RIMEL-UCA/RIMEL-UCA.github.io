#!/usr/bin/env python
# coding: utf-8

# # Updating a Historical Marquee Portfolio
# 
# If you already have a portfolio in Marquee, the GS Quant SDK provides a simple and intuitive workflow to update positions and rerun reports.
# 
# ## Permission Prerequisites
# 
# To execute all the code in this tutorial, you will need the following application scopes:
# - **read_product_data**
# - **read_financial_data**
# - **modify_financial_data** (must be requested)
# - **run_analytics** (must be requested)
# 
# If you are not yet permissioned for these scopes, please request them on your [My Applications Page](https://developer.gs.com/go/apps/view). If you have any other questions please reach out to the [Marquee sales team](mailto:gs-marquee-sales@gs.com).
# 
# You will also need to be an admin on the portfolio you would like to update. If you are not an admin, please ask a portfolio admin to [edit the portfolio's entitlements](../examples/marquee/01_edit_portfolio_entitlements.ipynb) to include you.

# ## Step 1: Authenticate and Initialize Your Session
# 
# First you will import the necessary modules and add your client id and client secret.

# In[ ]:


import datetime as dt

from gs_quant.common import PositionSet
from gs_quant.markets.portfolio_manager import PortfolioManager
from gs_quant.session import GsSession, Environment

client = None
secret = None
scopes = None

## External users must fill in their client ID and secret below and comment out the line below

#client = 'ENTER CLIENT ID'
#secret = 'ENTER CLIENT SECRET'
#scopes = ('read_product_data read_financial_data modify_financial_data run_analytics',)

GsSession.use(
    Environment.PROD,
    client_id=client,
    client_secret=secret,
    scopes=scopes
)

print('GS Session initialized.')

# ## Step 2: Define Your Portfolio ID and the Positions You Would Like to Upload
# 
# Portfolio positions in Marquee are stored on a holding basis, when means you only upload positions for days where you are rebalancing your portfolio. Take the following set of positions:

# In[ ]:


portfolio_id = 'ENTER PORTFOLIO ID'
portfolio_position_sets = [
    PositionSet.from_dicts(
        date=dt.date(day=3, month=5, year=2021),
        positions=[
            {
                'identifier': 'AAPL UW',
                'quantity': 25
            }, {
                'identifier': 'GS UN',
                'quantity': 50
            }]
    ),
    PositionSet.from_dicts(
        date=dt.date(day=1, month=7, year=2021),
        positions=[
            {
                'identifier': 'AAPL UW',
                'quantity': 26
            }, {
                'identifier': 'GS UN',
                'quantity': 51
            }]
    )
]

# #### Have your positions as a dataframe?
# 
# If you have a day's positions in a dataframe with columns `identifer` (string values), `quantity` (float values),
# and optionally `tags` (dictionary values), you can turn them into a `PositionSet` object by using the
# `PositionSet.from_frame()` function:
# 
# `position_set = PositionSet.from_frame(positions_df, datetime_date)`
# 
# ## Step 3: Post Positions to the Marquee Portfolio

# In[ ]:


pm = PortfolioManager(portfolio_id)
pm.update_positions(portfolio_position_sets)

# ## Step 4: Reschedule All Portfolio Reports
# 
# Now that the portfolio has new positions, it's time to rerun all reports associated with the portfolio so your performance, risk, and other analytics reflect these new positions.

# In[ ]:


pm.schedule_reports()

print('All portfolio reports scheduled.')

# *Other questions? Reach out to the [Portfolio Analytics team](mailto:gs-marquee-analytics-support@gs.com)!*
