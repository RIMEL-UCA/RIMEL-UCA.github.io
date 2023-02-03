#!/usr/bin/env python
# coding: utf-8

# # Thematic Reports
# 
# Thematic reports run historical analyses on the exposure of a portfolio to various Goldman Sachs Flagship Thematic baskets over a specified date range.
# 
# ### Prerequisite
# 
# To execute all the code in this tutorial, you will need the following application scopes:
# - **read_product_data**
# - **read_financial_data**
# - **modify_financial_data** (must be requested)
# - **run_analytics** (must be requested)
# 
# If you are not yet permissioned for these scopes, please request them on your [My Applications Page](https://developer.gs.com/go/apps/view).
# If you have any other questions please reach out to the [Marquee sales team](mailto:gs-marquee-sales@gs.com).
# 
# ## Step 1: Authenticate and Initialize Your Session
# 
# First you will import the necessary modules and add your client id and client secret.

# In[ ]:


import datetime as dt
from time import sleep

from gs_quant.markets.baskets import Basket
from gs_quant.markets.portfolio_manager import PortfolioManager
from gs_quant.markets.report import ThematicReport
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

# ## Step 2: Create a New Thematic Report
# 
# #### Already have a thematic report?
# 
# <i>If you want to skip creating a new report and continue this tutorial with an existing thematic report, run the following and skip to Step 3:</i>

# In[ ]:


portfolio_id = 'ENTER PORTFOLIO ID'

thematic_report = PortfolioManager(portfolio_id).get_thematic_report()

# The only parameter necessary in creating a new thematic report is the unique Marquee identifier of the portfolio on which you would like to run thematic analytics.

# In[ ]:


portfolio_id = 'ENTER PORTFOLIO ID'

thematic_report = ThematicReport()
thematic_report.set_position_source(portfolio_id)
thematic_report.save()

print(f'A new thematic report for portfolio "{portfolio_id}" has been made with ID "{thematic_report.id}".')

# ## Step 3: Schedule the Report
# 
# When scheduling reports, you have two options:
# - Backcast the report: Take the earliest date with positions in the portfolio / basket and run the report on the positions held then with a start date before the earliest position date and an end date
#  of the earliest position date
# - Do not backcast the report: Set the start date as a date that has positions in the portfolio or basket and an end date after that (best practice is to set it to T-1). In this case the
#  report will run on positions held as of each day in the date range
# 
# In this case, let's try scheduling the report without backcasting:

# In[ ]:


start_date = dt.date(2021, 1, 4)
end_date = dt.date(2021, 8, 4)

thematic_report.schedule(
    start_date=start_date,
    end_date=end_date,
    backcast=False
)

print(f'Report "{thematic_report.id}" has been scheduled.')

# ## Alternative Step 3: Run the Report
# 
# Depending on the size of your portfolio and the length of the schedule range, it usually takes anywhere from a couple seconds to half a minute for your report to finish executing.
# Only after that can you successfully pull the results from that report. If you would rather run the report and pull the results immediately after they are ready, you can leverage the `run`
# function.
# 
# You can run a report synchronously or asynchronously.
# - Synchronous: the Python script will stall at the `run` function line and wait for the report to finish. The `run` function will then return a dataframe with the report results
# - Asynchronously: the Python script will not stall at the `run` function line. The `run` function will return a `ReportJobFuture` object that will contain the report results when they are ready.
# 
# In this example, let's run the report asynchronously and wait for the results:

# In[ ]:


start_date = dt.date(2021, 1, 4)
end_date = dt.date(2021, 8, 4)

report_result_future = thematic_report.run(
    start_date=start_date,
    end_date=end_date,
    backcast=False,
    is_async=True
)

while not report_result_future.done():
    print('Waiting for report results...')
    sleep(5)

print('\nReport results done! Here they are...')
print(report_result_future.result())

# ### Step 3: Pull Report Results
# 
# Now that we have our factor risk report, we can leverage the unique functionalities of the `ThematicReport` class to pull exposure and PnL data. Let's get the historical changes in thematic exposure and beta to the GS Asia Stay at Home basket:

# In[ ]:


basket = Basket.get('GSXASTAY')
thematic_exposures = thematic_report.get_thematic_data(
    start_date=start_date,
    end_date=end_date,
    basket_ids=[basket.get_marquee_id()]
)

print(f'Thematic Exposures: \n{thematic_exposures.__str__()}')
thematic_exposures.plot(title='Thematic Data Breakdown')

# ### You're all set; Congrats!
# 
# *Other questions? Reach out to the [Portfolio Analytics team](mailto:gs-marquee-analytics-support@gs.com)!*
