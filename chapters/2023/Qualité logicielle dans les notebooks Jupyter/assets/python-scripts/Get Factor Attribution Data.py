#!/usr/bin/env python
# coding: utf-8

# # Pull Portfolio Factor Attribution Data with GS Quant
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



import pandas as pd
from IPython.display import display

from gs_quant.markets.portfolio_manager import PortfolioManager
from gs_quant.markets.report import FactorRiskViewsMode, FactorRiskUnit, FactorRiskTableMode
from gs_quant.markets.securities import SecurityMaster, AssetIdentifier
from gs_quant.session import GsSession, Environment

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

# ## Get Portfolio Factor Risk Report

# In[ ]:


risk_report = PortfolioManager('ENTER PORTFOLIO ID').get_factor_risk_report(risk_model_id='ENTER RISK MODEL ID',
                                                                            benchmark_id=None)

print(f'Factor risk report found with ID: {risk_report.id}')

# Want to query data for an active risk report? Leverage the `SecurityMaster` class to retrieve the benchmark identifier and
# pass it into the `get_factor_risk_report` function:

# In[ ]:


benchmark = SecurityMaster.get_asset(id_value='SPX', id_type=AssetIdentifier.BLOOMBERG_ID)

risk_report = PortfolioManager('ENTER PORTFOLIO ID').get_factor_risk_report(risk_model_id='ENTER RISK MODEL ID',
                                                                            benchmark_id=benchmark.get_marquee_id())

print(f'Factor risk report found with ID: {risk_report.id}')

# ## Get Current Portfolio Attribution
# 
# Once your risk report is scheduled as of the latest business day, you can view updated attribution broken down by factor category:

# In[ ]:


attr_data = risk_report.get_view(
  mode=FactorRiskViewsMode.Attribution,
  start_date=risk_report.earliest_start_date,
  end_date=risk_report.latest_end_date,
  unit=FactorRiskUnit.Notional
)
category_table = attr_data.get('factorCategoriesTable')
category_df = pd.DataFrame(category_table).filter(items=['name', 'pnl', 'minExposure', 'maxExposure', 'avgExposure'])

display(category_df)

# It is also possible to get a similar table for all the factors in a factor category. In this case, let's drill down into the Style factors:

# In[ ]:


factor_tables = attr_data.get('factorCategoryTables')
factor_tables = [f for f in factor_tables if f.get('factorCategory') == 'Style']
factor_df = pd.DataFrame(factor_tables[0].get('factors')).filter(items=['name', 'pnl', 'minExposure', 'maxExposure', 'avgExposure'])

display(factor_df)

# You can also generate a table that shows you the factor PnL over a date range at the asset level. Let's see the factor PnL for each asset for the factors Country, Beta, and Earnings Quality:

# In[ ]:


pnl_table = risk_report.get_table(
    mode=FactorRiskTableMode.Pnl,
    start_date=risk_report.earliest_start_date,
    end_date=risk_report.latest_end_date,
    factors=["Country", "Beta", "Earnings Quality"]    # Skip passing in a value here to get a table with all model factors
)

display(pd.DataFrame(pnl_table))

# ## Historical Portfolio Factor Performance
# 
# `get_factor_pnl` allows you to pull historical factor performance for a list of factors, as well as aggregations like factor, specific, and total risk:

# In[ ]:


pnl = risk_report.get_factor_pnl(
  factor_names=['Factor', 'Specific', 'Total','Market', 'Country', 'Industry', 'Style'],
  start_date=risk_report.earliest_start_date,
  end_date=risk_report.latest_end_date,
  unit=FactorRiskUnit.Notional
)
pnl_overview = pnl.filter(items=['date', 'Total']).set_index('date')

pnl_overview.cumsum().plot(title='PnL')

# This makes it easy to break down PnL over time and how it was attributed to various systematic risk factors:

# In[ ]:


pnl_overview = pnl.filter(items=['date', 'Factor', 'Specific', 'Total']).set_index('date')
pnl_overview.cumsum().plot(title='PnL Overview')

# And dissect that further by factor attribution further by category...

# In[ ]:


pnl_by_type = pnl.filter(items=['date', 'Market', 'Country', 'Industry', 'Style', 'Specific']).set_index('date')

pnl_by_type.cumsum().plot(title='PnL by Factor Category')


# ## Historical Factor Exposure
# 
# For each day, it's possible to pull your portfolio's exposure to specific factors...

# In[ ]:


category_exposures = risk_report.get_factor_exposure(
  factor_names=['Market', 'Industry', 'Style'],
  start_date=risk_report.earliest_start_date,
  end_date=risk_report.latest_end_date,
  unit=FactorRiskUnit.Notional
).set_index('date')

category_exposures.plot(title='Exposures to Factor Categories')

# Or get the exposures to all factors in a given category:

# In[ ]:


category_exposures = risk_report.get_factor_exposure(
  factor_categories=['Style'],
  start_date=risk_report.earliest_start_date,
  end_date=risk_report.latest_end_date,
  unit=FactorRiskUnit.Notional
).set_index('date')

category_exposures.plot(title='Exposures to Style Factors')


# *Other questions? Reach out to the [Portfolio Analytics team](mailto:gs-marquee-analytics-support@gs.com)!*
