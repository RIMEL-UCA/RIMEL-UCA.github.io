#!/usr/bin/env python
# coding: utf-8

# # Performance Reports
# 
# Performance reports run historical analyses on measures like exposure and PnL for your portfolio over a specified date range.

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
import pandas as pd
from time import sleep

from gs_quant.markets.portfolio_manager import PortfolioManager
from gs_quant.markets.report import PerformanceReport
from gs_quant.markets.securities import Asset, AssetIdentifier
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

# ## Step 2: Get Performance Report
# 
# When creating a Marquee portfolio, a corresponding performance report for it is automatically created. Thus, we can leverage the `PortfolioManager` class to pull all the reports associated with your portfolio and find the performance report.

# In[ ]:


portfolio_id = 'ENTER PORTFOLIO ID'

all_reports = PortfolioManager(portfolio_id).get_reports()
performance_report = list(filter(lambda report: isinstance(report, PerformanceReport), all_reports))[0]

print(f'Performance report for portfolio "{portfolio_id}" has been found with ID "{performance_report.id}".')

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

performance_report.schedule(
    start_date=start_date,
    end_date=end_date,
    backcast=False
)

print(f'A new performance report for portfolio "{performance_report.position_source_id}" has been made with ID "{performance_report.id}".')

# ## Alternative Step 3: Run the Report
# 
# Depending on the size of your portfolio and the length of the schedule range, it usually takes anywhere from a couple seconds to a couple minutes for your report to finish executing.
# Only after that can you successfully pull the results from that report. If you would rather run the report and pull the results immediately after they are ready, you can leverage the `run`
# function of the `Report` class.
# 
# You can run a report synchronously or asynchronously.
# - Synchronous: the Python script will stall at the `run` function line and wait for the report to finish. The `run` function will then return a dataframe with the report results
# - Asynchronously: the Python script will not stall at the `run` function line. The `run` function will return a `ReportJobFuture` object that will contain the report results when they are ready.
# 
# In this example, let's run the report asynchronously and wait for the results:

# In[ ]:


start_date = dt.date(2021, 1, 4)
end_date = dt.date(2021, 8, 4)

report_result_future = performance_report.run(start_date=start_date, end_date=end_date, backcast=False, is_async=True)

while not report_result_future.done():
    print('Waiting for report results...')
    sleep(5)

print('\nReport results done! Here they are...')
print(report_result_future.result())

# ## Step 4: Pull Report Results
# 
# Now that we have our performance report, we can leverage the unique functionalities of the `PerformanceReport` class to pull daily exposure and PnL data:

# In[ ]:


all_exposures = performance_report.get_many_measures(
    measures=['pnl', 'grossExposure', 'netExposure'],
    start_date=performance_report.earliest_start_date,
    end_date=performance_report.latest_end_date
)

print(all_exposures)
all_exposures.plot(title='Performance Breakdown')

# Leverage the Brinson Attribution function in the `PerformanceReport` class to compare the PnL of your portfolio to a benchmark, which can be any equity ETF, Index, or Basket in Marquee:

# In[ ]:


asset = Asset.get(id_value='ENTER BENCHMARK IDENTIFIER', id_type=AssetIdentifier.TICKER)

brinson_attribution_results = performance_report.get_brinson_attribution(
    benchmark=asset.get_marquee_id(),
    include_interaction=True,
    start_date=performance_report.earliest_start_date,
    end_date=performance_report.latest_end_date
)

display(pd.DataFrame(brinson_attribution_results))

# ### You're all set; Congrats!
# 
# *Other questions? Reach out to the [Portfolio Analytics team](mailto:gs-marquee-analytics-support@gs.com)!*
