#!/usr/bin/env python
# coding: utf-8

# # Factor Risk Reports
# 
# Factor risk reports run historical factor risk analyses for your portfolio or basket over a specified date range by leveraging a factor risk model of your choice.
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

import pandas as pd

from gs_quant.api.gs.risk_models import GsRiskModelApi
from gs_quant.markets.report import FactorRiskReport
from gs_quant.models.risk_model import FactorRiskModel
from gs_quant.session import GsSession, Environment
from gs_quant.markets.securities import SecurityMaster, AssetIdentifier

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


# ## Step 2: Create a New Factor Risk Report
# 
# #### Already have a factor risk report?
# 
# <i>If you want to skip creating a new report and continue this tutorial with an existing factor risk report, run the following and skip to Step 3:</i>

# In[ ]:


risk_report_id = 'ENTER FACTOR RISK REPORT ID'

risk_report = FactorRiskReport.get(risk_report_id)

# When creating a factor risk report, you must specify the risk model you would like to use.
# 
# 
# If you would like to see all available risk model IDs to choose from, run the following:

# In[ ]:


risk_models = FactorRiskModel.get_many()
for risk_model in risk_models:
    print(f'{risk_model.id}\n')

# In this tutorial, we'll create a factor risk report leveraging the Barra USSLOW Long model. If you would like to calculate
# risk in relation to a benchmark, you can add an index, basket, or ETF to your `FactorRiskReport` object:

# In[ ]:


entity_id = 'ENTER PORTFOLIO OR BASKET ID'
risk_model_id = 'BARRA_USSLOWL'

benchmark = SecurityMaster.get_asset(id_value='SPX', id_type=AssetIdentifier.BLOOMBERG_ID)

risk_report = FactorRiskReport(
    risk_model_id=risk_model_id,
    fx_hedged=True,
    benchmark_id=benchmark.get_marquee_id()
)

risk_report.set_position_source(entity_id)
risk_report.save()

print(f'A new factor risk report for entity "{entity_id}" has been made with ID "{risk_report.id}".')

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

risk_report.schedule(
    start_date=start_date,
    end_date=end_date,
    backcast=False
)

print(f'Report "{risk_report.id}" has been scheduled.')

# ## Alternative Step 3: Run the Report
# 
# Depending on the size of your portfolio and the length of the schedule range, it usually takes anywhere from a couple seconds to a couple minutes for your report to finish executing.
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

report_result_future = risk_report.run(
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

# ## Step 4: Pull Report Results
# 
# Now that we have our completed factor risk report, we can leverage the unique functionalities of the `FactorRiskReport` class to pull attribution and risk data. In this example, let's pull historical data on factor, specific, and total PnL:

# In[ ]:


pnl = risk_report.get_factor_pnl(
    factor_name=['Factor', 'Specific', 'Total'],
    start_date=start_date,
    end_date=end_date
)
pnl.set_index('date', inplace=True)
pnl.index = pd.to_datetime(pnl.index)

pnl.cumsum().plot(title='Risk Attribution Breakdown')

# Now let's pull the breakdown of proportion of risk among the different factor types over time:

# In[ ]:


prop_of_risk = risk_report.get_factor_proportion_of_risk(
    factor_names=['Market', 'Style', 'Industry', 'Country'],
    start_date=start_date,
    end_date=end_date
)
prop_of_risk.set_index('date', inplace=True)
prop_of_risk.index = pd.to_datetime(prop_of_risk.index)

prop_of_risk.plot(title='Factor Proportion of Risk Breakdown')

# ### Quick Tip!
# If you would like to pull all factor risk data for a list of different factors, you can use the `get_results` function:

# In[ ]:


factor_and_total_results = risk_report.get_results(factors=['Factor', 'Specific'], start_date=dt.date(2020, 1, 1), end_date=dt.date(2021, 1, 1))
print(factor_and_total_results)

# ### You're all set; Congrats!
# 
# *Other questions? Reach out to the [Portfolio Analytics team](mailto:gs-marquee-analytics-support@gs.com)!*
# 
