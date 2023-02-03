#!/usr/bin/env python
# coding: utf-8

# ### Pull Portfolio Factor Attribution Data with GS Quant
# 
# First get your portfolio's factor risk report:

# In[ ]:


import pandas as pd
from IPython.display import display
import warnings
from dateutil.relativedelta import relativedelta

from gs_quant.markets.portfolio_manager import PortfolioManager
from gs_quant.markets.report import FactorRiskViewsMode, FactorRiskTableMode
from gs_quant.session import GsSession, Environment

GsSession.use(Environment.PROD)
warnings.filterwarnings("ignore", category=RuntimeWarning)

portfolio_id = 'MPWQQ8B05FKPCCH6'
risk_model_id = 'BARRA_USFAST'


pm = PortfolioManager(portfolio_id)
risk_report = pm.get_factor_risk_report(risk_model_id)

# Uncomment the two lines below to get active risk data instead
#benchmark = SecurityMaster.get_asset(id_value='SPX', id_type=AssetIdentifier.BLOOMBERG_ID)
#risk_report = PortfolioManager('ENTER PORTFOLIO ID').get_factor_risk_report(risk_model_id=risk_model_id, benchmark_id=benchmark.get_marquee_id())


# Then get historical overview of your factor and specific PnL over time:

# In[ ]:


# Get Historical PnL and PnL by Type
pnl = risk_report.get_factor_pnl(
    factor_names=['Factor', 'Specific', 'Total','Market', 'Country', 'Industry', 'Style'],
    start_date=risk_report.latest_end_date - relativedelta(years=1),
    end_date=risk_report.latest_end_date)

pnl_overview = pnl.filter(items=['date', 'Factor', 'Specific', 'Total']).set_index('date')
pnl_overview.cumsum().plot(title='PnL Overview')

# You can also break down your factor PnL further by factor category:

# In[ ]:


pnl_by_type = pnl.filter(items=['date', 'Market', 'Country', 'Industry', 'Style', 'Specific']).set_index('date')
pnl_by_type.cumsum().plot(title='PnL by Factor Category')

# Here is a summary of your most updated factor attribution by factor category:

# In[ ]:


# Get Pnl By Type Table
table_data = risk_report.get_view(
    mode=FactorRiskViewsMode.Attribution,
    start_date=risk_report.latest_end_date - relativedelta(years=1),
    end_date=risk_report.latest_end_date).get('factorCategoriesTable')

display(pd.DataFrame(table_data).filter(items=['name', 'pnl', 'minExposure', 'maxExposure', 'avgExposure']))


# Now get a table with your most updated asset PnL by factor:

# In[ ]:


# Get Factor PnL by Asset Table
pnl_table = risk_report.get_table(
    mode=FactorRiskTableMode.Pnl,
    start_date=risk_report.earliest_start_date,
    end_date=risk_report.latest_end_date
)
display(pd.DataFrame(pnl_table))
