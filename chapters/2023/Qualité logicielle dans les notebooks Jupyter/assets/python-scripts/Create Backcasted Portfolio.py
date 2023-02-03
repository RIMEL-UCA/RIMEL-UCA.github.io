#!/usr/bin/env python
# coding: utf-8

# ## Pull Backcasted Factor Risk
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

# First you will import the necessary modules and add your client id and client secret.

# In[ ]:


import datetime as dt
import warnings

from gs_quant.datetime import business_day_offset
from gs_quant.markets.portfolio import Portfolio
from gs_quant.markets.portfolio_manager import PortfolioManager
from gs_quant.markets.position_set import PositionSet
from gs_quant.markets.report import FactorRiskReport
from gs_quant.models.risk_model import FactorRiskModel
from gs_quant.session import GsSession, Environment

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
warnings.filterwarnings("ignore", category=RuntimeWarning)

print('GS Session initialized.')

# **Next define your positions and risk model:**
# 
# *We will resolve all identifiers (bloomberg IDs, SEDOLs, RICs, etc) historically on our end as of the position date*

# In[ ]:


positions = [
    {
        'identifier': 'AAPL UW',
        'quantity': 25
    }, {
        'identifier': 'GS UN',
        'quantity': 50
    }, {
        'identifier': 'FB UW',
        'quantity': 25
    }, {
        'identifier': 'AMZN UN',
        'quantity': 50
    }, {
        'identifier': 'MSFT UW',
        'quantity': 25
    }, {
        'identifier': 'AZN UW',
        'quantity': 50
    }
]
risk_model_id = 'AXUS4M'

print('Positions and risk model ID saved.')

# ### Quick Tip!
# 
# *Premium clients get access to many more risk models (including premium vendors like MSCI Barra),
# while non-premium clients get access to a limited suite of models. To see which models you have access to,
# simply run the following:*

# In[ ]:


risk_models = FactorRiskModel.get_many(limit=100)
for risk_model in risk_models:
    print(f'{risk_model.name}: {risk_model.id}\n')

# **Create a portfolio with those positions held on the last previous business day:**

# In[ ]:


portfolio = Portfolio(name='My Backcasted Portfolio')
portfolio.save()

pm = PortfolioManager(portfolio.id)
pm.update_positions([PositionSet.from_dicts(date=business_day_offset(dt.date.today(), -1, roll='forward'),
                                            positions=positions)])

# **Now we can create a factor risk report for the portfolio...**

# In[ ]:


pm.schedule_reports(backcast=True)
risk_report = FactorRiskReport(risk_model_id='AXUS4M', fx_hedged=True)
risk_report.set_position_source(portfolio.id)
risk_report.save()

print(f'Portfolio created with ID "{portfolio.id}".')

# **And run risk calculations backcasted one year:**

# In[ ]:


risk_results = risk_report.run(backcast=True, is_async=False)

# **Once that's done, you can pull the results directly from the `risk_results` object:**

# In[ ]:


risk_results = risk_results[risk_results['factorCategory'] == 'Aggregations']

factor_exposures = risk_results.filter(items=['date', 'factor', 'exposure']).pivot(index='date', columns='factor', values='exposure')
factor_pnl= risk_results.filter(items=['date', 'factor', 'pnl']).pivot(index='date', columns='factor', values='pnl')

factor_exposures.plot(title='Factor Exposures')
factor_pnl.cumsum().plot(title='Factor PnL')


print(f'Compare to your portfolio UI here: https://marquee.gs.com/s/portfolios/{portfolio.id}/attribution')

# In[ ]:


risk_results = risk_results[risk_results['factorCategory'] == 'Aggregations']

factor_exposures = risk_results.filter(items=['date', 'factor', 'exposure']).pivot(index='date', columns='factor', values='exposure')
factor_pnl= risk_results.filter(items=['date', 'factor', 'pnl']).pivot(index='date', columns='factor', values='pnl')

factor_exposures.plot(title='Factor Exposures')
factor_pnl.cumsum().plot(title='Factor PnL')


print(f'Compare to your portfolio UI here: https://marquee.gs.com/s/portfolios/{portfolio.id}/attribution')

# **And run risk calculations backcasted one year:**

# In[ ]:


risk_results = risk_report.run(backcast=True, is_async=False)

# **Once that's done, you can pull the results directly from the `risk_results` object:**

# In[ ]:


risk_results = risk_results[risk_results['factorCategory'] == 'Aggregations']

factor_exposures = risk_results.filter(items=['date', 'factor', 'exposure']).pivot(index='date', columns='factor', values='exposure')
factor_pnl= risk_results.filter(items=['date', 'factor', 'pnl']).pivot(index='date', columns='factor', values='pnl')

factor_exposures.plot(title='Factor Exposures')
factor_pnl.cumsum().plot(title='Factor PnL')


print(f'Compare to your portfolio UI here: https://marquee.gs.com/s/portfolios/{portfolio.id}/attribution')
