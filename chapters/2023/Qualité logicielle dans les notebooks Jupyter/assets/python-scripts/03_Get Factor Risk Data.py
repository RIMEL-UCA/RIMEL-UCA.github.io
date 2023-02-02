#!/usr/bin/env python
# coding: utf-8

# ## Pull Portfolio Factor Risk Data with GS Quant
# 
# **First get your portfolio's factor risk and performance reports:**

# In[ ]:


import pandas as pd
from IPython.display import display
from dateutil.relativedelta import relativedelta
import warnings

from gs_quant.markets.portfolio_manager import PortfolioManager
from gs_quant.markets.report import PerformanceReport, FactorRiskViewsMode, FactorRiskTableMode, FactorRiskUnit
from gs_quant.session import GsSession, Environment

GsSession.use(Environment.PROD)
warnings.filterwarnings("ignore", category=RuntimeWarning)


portfolio_id = 'MPWQQ8B05FKPCCH6'
risk_model_id = 'BARRA_USFAST'


pm = PortfolioManager(portfolio_id)
risk_report = pm.get_factor_risk_report(risk_model_id)
performance_report = pm.get_performance_report()

# Uncomment this section to get active risk report instead
#benchmark = SecurityMaster.get_asset(id_value='SPX', id_type=AssetIdentifier.BLOOMBERG_ID)
#risk_report = PortfolioManager('ENTER PORTFOLIO ID').get_factor_risk_report(risk_model_id='AXWW4M', benchmark_id=benchmark.get_marquee_id())

# **Now let's plot the portfolio's historical annualized risk:**

# In[ ]:


risk_data = risk_report.get_view(
    mode=FactorRiskViewsMode.Risk,
    start_date=risk_report.latest_end_date - relativedelta(years=1),
    end_date=risk_report.latest_end_date)

historical_risk = pd.DataFrame(risk_data.get('overviewTimeSeries')).filter(items=['date', 'annualizedExAnteRiskPercent']).set_index('date')
historical_risk.rename(columns={'annualizedExAnteRiskPercent': 'Total Risk'}, inplace=True)
historical_risk.plot(title='Annualized Risk % (ex-ante)')

# **Similarly, you can also pull historical proportion of risk in terms of factor and idiosyncratic risk:**

# In[ ]:


historical_risk = pd.DataFrame(risk_data.get('overviewTimeSeries')).filter(items=['date', 'factorProportionOfRisk', 'specificProportionOfRisk']).set_index('date')
historical_risk.rename(columns={'factorProportionOfRisk': 'Factor Risk', 'specificProportionOfRisk': 'Specific Risk'}, inplace=True)
historical_risk.plot(title='Factor and Specific Risk')

# **Then pull the portfolio's risk data by factor category:**

# In[ ]:


category_table = risk_data.get('factorCategoriesTable')
display(pd.DataFrame(category_table).filter(items=['name', 'proportionOfRisk', 'marginalContributionToRiskPercent',
                                                   'relativeMarginalContributionToRisk', 'exposure', 'avgProportionOfRisk']))

# **Now generate the factor risk by asset z-score data:**

# In[ ]:


# Get ZScore Factor Risk by Asset Table
zscore_table = risk_report.get_table(mode=FactorRiskTableMode.ZScore)
display(pd.DataFrame(zscore_table))

# **Similarly, create the factor exposure by asset table:**

# In[ ]:


# Get Exposure Factor Risk by Asset Table
exposure_table = risk_report.get_table(mode=FactorRiskTableMode.Exposure)
display(pd.DataFrame(exposure_table))

# **Just like the previous two steps, generate the factor risk by asset MCTR data. This time, let's query the percentages instead of the notional values by utilizing the unit parameter:**

# In[ ]:


# Get MCTR Factor Risk by Asset Table
mctr_table = risk_report.get_table(mode=FactorRiskTableMode.Mctr, unit=FactorRiskUnit.Percent)
display(pd.DataFrame(mctr_table))

# **Then plot historical proportion of risk across all factor categories:**

# In[ ]:


# Parse Proportion of Risk Timeseries
prop_of_risk = risk_report.get_factor_proportion_of_risk(
    factor_names=['Market', 'Industry', 'Style'],
    start_date=risk_report.latest_end_date - relativedelta(years=1),
    end_date=risk_report.latest_end_date).set_index('date')

prop_of_risk.plot(title='Proportion of Risk By Factor Category')

