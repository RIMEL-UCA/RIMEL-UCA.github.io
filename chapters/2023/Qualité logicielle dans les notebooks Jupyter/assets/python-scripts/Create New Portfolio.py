#!/usr/bin/env python
# coding: utf-8

# # Create a Marquee Portfolio with GS Quant
# 
# The Marquee Portfolio Service provides a powerful framework for uploading portfolio positions and retrieving analytics including historical performance, factor risk exposure, ESG analytics, and more. GS Quant makes operating the suite of Portfolio Service API's intuitive and fast.
# 
# ## Permission Prerequisites
# 
# To execute all the code in this tutorial, you will need the following application scopes:
# - **read_product_data**
# - **read_financial_data**
# - **modify_product_data** (must be requested)
# - **modify_financial_data** (must be requested)
# - **run_analytics** (must be requested)
# - **read_user_profile**
# 
# If you are not yet permissioned for these scopes, please request them on your [My Applications Page](https://developer.gs.com/go/apps/view). If you have any other questions please reach out to the [Marquee sales team](mailto:gs-marquee-sales@gs.com).
# 

# ## Step 1: Authenticate and Initialize Your Session
# 
# First you will import the necessary modules and add your client id and client secret.

# In[ ]:


import datetime as dt

from gs_quant.entities.entitlements import Entitlements, EntitlementBlock, User
from gs_quant.markets.portfolio import Portfolio
from gs_quant.markets.portfolio_manager import PortfolioManager, CustomAUMDataPoint
from gs_quant.markets.position_set import PositionSet
from gs_quant.markets.report import FactorRiskReport, ThematicReport
from gs_quant.markets.securities import SecurityMaster, AssetIdentifier
from gs_quant.session import GsSession, Environment
from gs_quant.target.portfolios import RiskAumSource

client = None
secret = None
scopes = None

## External users must fill in their client ID and secret below and comment out the line below

#client = 'ENTER CLIENT ID'
#secret = 'ENTER CLIENT SECRET'
#scopes = ('read_product_data read_financial_data modify_product_data modify_financial_data run_analytics read_user_profile',)

GsSession.use(
    Environment.PROD,
    client_id=client,
    client_secret=secret,
    scopes=scopes
)

print('GS Session initialized.')

# ## Step 2: Create the Portfolio
# 
# The first step is to create a new, empty portfolio in Marquee.

# In[ ]:


portfolio = Portfolio(name='My New Portfolio')
portfolio.save()

print(f"Created portfolio '{portfolio.name}' with ID: {portfolio.id}")

# Once your portfolio has been saved to Marquee, the `PortfolioManager` class allows users to interact with their Marquee portfolios directly from GS Quant. We will be using `PortfolioManager` to update portfolio positions, entitlements, update custom AUM, and run reports.

# In[ ]:


pm = PortfolioManager(portfolio.id)

# ## Step 3: Define Portfolio Entitlements (Optional)
# 
# By default, an application will have all entitlement permissions to a portfolio it makes.
# However, if you would like to share the portfolio with others, either Marquee users or other
# applications, you will need to specify them in the entitlements parameter of the portfolio.
# Let's walk through how we convert a list of admin and viewer emails into an `Entitlements` object:
# 
# *Note: If you would like to see this portfolio on your Marquee webpage, you'll need to add your account
# email address into the `portfolio_admin_emails` list*

# In[ ]:


portfolio_admin_emails = ['LIST OF ADMIN EMAILS']
portfolio_viewer_emails = ['LIST OF VIEWER EMAILS']

admin_entitlements = EntitlementBlock(users=User.get_many(emails=portfolio_admin_emails))
view_entitlements = EntitlementBlock(users=User.get_many(emails=portfolio_viewer_emails))

entitlements = Entitlements(
    view=view_entitlements,
    admin=admin_entitlements
)

print(f'Entitlements:\n{entitlements.to_dict()}')

pm.set_entitlements(entitlements)

print(f"Updated entitlements for '{portfolio.name}'")

# ## Step 4: Define Portfolio Positions
# 
# Portfolio positions in Marquee are stored on a holding basis, when means you only upload positions for days where you are rebalancing your portfolio. Take the following set of positions:

# In[ ]:


portfolio_position_sets = [
    PositionSet.from_dicts(
        date=dt.date(day=3, month=5, year=2021),
        positions=[
            {
                'identifier': 'AAPL UW',
                'quantity': 25,
                'tags': {'Analyst': 'Jane Doe'}
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
                'quantity': 26,
                'tags': {'Analyst': 'Jane Doe'}
            }, {
                'identifier': 'GS UN',
                'quantity': 51
            }]
    )
]

# If these positions were to be uploaded correctly, this portfolio would hold 50 shares of GS UN and 25 shares of AAPL UW from May 3, 2021 to June 30, 2021, and it would hold 51 shares of GS UN and 26 shares of AAPL UW from July 1, 2021 to today.
# 
# #### Have your positions as a dataframe?
# 
# If you have a day's positions in a dataframe with columns `identifer` (string values), `quantity` (float values),
# and optionally `tags` (dictionary values), you can turn them into a `PositionSet` object by using the
# `PositionSet.from_frame()` function:
# 
# `position_set = PositionSet.from_frame(positions_df, datetime_date)`

# In[ ]:


pm.update_positions(portfolio_position_sets)

print(f"Updated positions for '{portfolio.name}'")

# ## Step 5: Create Factor Risk and/or Thematic Reports (Optional)
# 
# By default, creating a portfolio will automatically create a corresponding performance report for it as well.
# If you would like to create a factor risk and/or thematic report (more documentation on reports found [here](https://developer.gs.com/p/docs/services/portfolio/programmatic-access/reports/))
# for it as well, run the following:

# In[ ]:


risk_model_id = 'RISK_MODEL_ID_HERE'
benchmark = SecurityMaster.get_asset(id_value='BENCHMARK TICKER HERE', id_type=AssetIdentifier.TICKER)

# Add a factor risk report
risk_report = FactorRiskReport(risk_model_id=risk_model_id)
risk_report.set_position_source(portfolio.id)
risk_report.save()

# Add an active factor risk report with a benchmark of your choice
active_risk_report = FactorRiskReport(risk_model_id=risk_model_id, benchmark_id=benchmark.get_marquee_id())
active_risk_report.set_position_source(portfolio.id)
active_risk_report.save()

# Add a thematic report
thematic_report = ThematicReport()
thematic_report.set_position_source(portfolio.id)
thematic_report.save()

print('All portfolio reports created.')

# #### Quick Tip!
# 
# *Explore the different factor risk models available in Marquee in our [Data Catalog](https://marquee.gs.com/s/discover/data-services/catalog?Category=Factor+Risk+Model).*
# 
# 
# ## Step 6: Schedule Reports
# 
# Now, it's schedule all the portfolio reports. Once this is done and reports are completed, you can programmatically retrieve factor risk and attribution data for your portfolio.
# 
# When scheduling reports, you have two options:
# - Backcast the report: Take the earliest date with positions in the portfolio / basket and run the report on the positions held then with a start date before the earliest position date and an end date
#  of the earliest position date. This option is ideal for snapshot portfolios.
# - Do not backcast the report: Set the start date as a date that has positions in the portfolio or basket and an end date after that (best practice is to set it to T-1). In this case the
#  report will run on positions held as of each day in the date range. This option is ideal for historical portfolios.

# In[ ]:


pm.schedule_reports(backcast=False)

print('All portfolio reports scheduled.')

# ## Step 7: Update Custom AUM/NAV (Optional)
# The `CustomAUMDataPoint` class is used to represent custom AUM data for a specific date. A list of them can be posted to Marquee using our initialized `PortfolioManager`. If you do not upload custom AUM data for your portfolio and change your portfolio's AUM Source to `Custom AUM`, by default the "AUM" (which is used for calculating risk as percent values) will be your portfolio's long exposure.

# In[ ]:


pm.set_aum_source(RiskAumSource.Custom_AUM)
custom_aum = [
    CustomAUMDataPoint(date=dt.date(2021, 5, 1), aum=100000),
    CustomAUMDataPoint(date=dt.date(2021, 7, 1), aum=200000)
]
pm.upload_custom_aum(custom_aum, clear_existing_data=False)

print(f"Custom AUM for '{portfolio.name} successfully uploaded'")

# *Other questions? Reach out to the [Portfolio Analytics team](mailto:gs-marquee-analytics-support@gs.com)!*
# 
