#!/usr/bin/env python
# coding: utf-8

# # Pull Portfolio Factor Risk Data with GS Quant
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


risk_report = PortfolioManager('ENTER PORTFOLIO ID').get_factor_risk_report(risk_model_id='ENTER RISK MODEL ID')

print(f'Factor risk report found with ID: {risk_report.id}')

# Want to query data for an active risk report? Leverage the `SecurityMaster` class to retrieve the benchmark identifier and
# pass it into the `get_factor_risk_report` function:

# In[ ]:


benchmark = SecurityMaster.get_asset(id_value='SPX', id_type=AssetIdentifier.BLOOMBERG_ID)

risk_report = PortfolioManager('ENTER PORTFOLIO ID').get_factor_risk_report(risk_model_id='ENTER RISK MODEL ID',
                                                                            benchmark_id=benchmark.get_marquee_id())

print(f'Factor risk report found with ID: {risk_report.id}')

# ## Current Portfolio Risk
# 
# Once your risk report is scheduled as of the latest business day, you can view updated risk data broken down by factor category:

# In[ ]:


category_table = risk_report.get_view(
  mode=FactorRiskViewsMode.Risk,
  start_date=risk_report.latest_end_date,
  end_date=risk_report.latest_end_date,
  unit=FactorRiskUnit.Notional
).get('factorCategoriesTable')

category_df = pd.DataFrame(category_table).filter(items=['name', 'proportionOfRisk', 'marginalContributionToRiskPercent', 'relativeMarginalContributionToRisk', 'exposure', 'avgProportionOfRisk'])
category_df.rename(
  columns={
    'proportionOfRisk': 'Prop. of Risk',
    'marginalContributionToRiskPercent': 'MCTR Percent',
    'relativeMarginalContributionToRisk': 'MCTR (USD)',
    'exposure': 'Exposure (USD)',
    'avgProportionOfRisk': 'Avg Prop. of Risk'
  },
  inplace=True
)

display(category_df)

# It is also possible to get a similar table for all the factors in a factor category. In this case, let's pull risk data for all the Style factors:

# In[ ]:


factor_table = risk_report.get_view(
  mode=FactorRiskViewsMode.Risk,
  factor_category='Style',
  start_date=risk_report.latest_end_date,
  end_date=risk_report.latest_end_date,
  unit=FactorRiskUnit.Notional
).get('factorsTable')

factor_df = pd.DataFrame(factor_table).filter(items=['name', 'proportionOfRisk', 'marginalContributionToRiskPercent', 'relativeMarginalContributionToRisk', 'exposure', 'avgProportionOfRisk'])
factor_df.rename(
  columns={
    'proportionOfRisk': 'Prop. of Risk',
    'marginalContributionToRiskPercent': 'MCTR %',
    'relativeMarginalContributionToRisk': 'MCTR (USD)',
    'exposure': 'Exposure (USD)',
    'avgProportionOfRisk': 'Avg Prop. of Risk'
  },
  inplace=True
)

display(factor_df)

# For an asset-level risk breakdown for a set of factors, leverage the get_table() function on the FactorRiskReport class. In this case, let's see the Z-Scores for the factors Beta, Dividend Yield, and Downside Risk:

# In[ ]:


zscore_table = risk_report.get_table(
    mode=FactorRiskTableMode.ZScore,
    factors=["Beta","Dividend Yield","Downside Risk"]    # Skip passing in a value here to get a table with all model factors
)

display(pd.DataFrame(zscore_table))

# ## Historical Portfolio Risk
# 
# First let's pull the daily annualized risk across the duration of your portfolio:

# In[ ]:


risk_data = risk_report.get_view(
  mode=FactorRiskViewsMode.Risk,
  start_date=risk_report.earliest_start_date,
  end_date=risk_report.latest_end_date,
  unit=FactorRiskUnit.Notional
)

historical_risk = pd.DataFrame(risk_data.get('overviewTimeSeries')).filter(items=['date', 'annualizedExAnteRiskPercent']).set_index('date')
historical_risk.rename(columns={'annualizedExAnteRiskPercent': 'Total Risk'}, inplace=True)

historical_risk.plot(title='Annualized Risk % (ex-ante)')

# For each day, you can see what percent of your risk is contributed to factor risk and what percent is idiosyncratic:

# In[ ]:


historical_risk = pd.DataFrame(risk_data.get('overviewTimeSeries')).filter(items=['date', 'factorProportionOfRisk', 'specificProportionOfRisk']).set_index('date')
historical_risk.rename(columns={'factorProportionOfRisk': 'Factor Risk', 'specificProportionOfRisk': 'Specific Risk'}, inplace=True)

historical_risk.plot(title='Factor and Specific Risk')

# It's even possible to break down that factor risk further by category...

# In[ ]:


prop_of_risk = risk_report.get_factor_proportion_of_risk(
  factor_names=['Market', 'Industry', 'Style'],
  start_date=risk_report.earliest_start_date,
  end_date=risk_report.latest_end_date
).set_index('date')

prop_of_risk.plot(title='Proportion of Risk By Factor Category')

# And by factors within a category. In this case, let's try the Style factors:

# In[ ]:


prop_of_risk = risk_report.get_factor_proportion_of_risk(
  factor_categories=['Style'],
  start_date=risk_report.earliest_start_date,
  end_date=risk_report.latest_end_date
).set_index('date')

prop_of_risk.plot(title='Proportion of Risk of Style Factors')

# *Other questions? Reach out to the [Portfolio Analytics team](mailto:gs-marquee-analytics-support@gs.com)!*
