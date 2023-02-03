#!/usr/bin/env python
# coding: utf-8

# ## Create a Portfolio with GS Quant
# 
# **First define your positions and risk model:**
# 
# *We will resolve all identifiers (Bloomberg IDs, SEDOLs, RICs, etc) historically on our end as of the position date*

# In[ ]:


import datetime as dt
import warnings

from gs_quant.markets.portfolio import Portfolio
from gs_quant.markets.portfolio_manager import PortfolioManager, CustomAUMDataPoint
from gs_quant.markets.position_set import Position, PositionSet
from gs_quant.markets.report import FactorRiskReport
from gs_quant.session import GsSession, Environment
from gs_quant.target.portfolios import RiskAumSource

GsSession.use(Environment.PROD)
warnings.filterwarnings("ignore", category=RuntimeWarning)

portfolio_position_sets = [
    PositionSet(
        date=dt.date(day=3, month=5, year=2022),
        positions=[
            Position(identifier='AAPL UW', quantity=25, tags={'Analyst': 'Marcus Goldman'}),
            Position(identifier='GS UN', quantity=50, tags={'Analyst': 'Samuel Sachs'})
        ]
    ),
    PositionSet(
        date=dt.date(day=1, month=7, year=2022),
        positions=[
            Position(identifier='AAPL UW', quantity=26, tags={'Analyst': 'Marcus Goldman'}),
            Position(identifier='GS UN', quantity=51, tags={'Analyst': 'Samuel Sachs'})
        ]
    )
]
risk_model_id = 'BARRA_USFAST'

print('Positions and risk model ID saved.')

# **Now, we'll create a new empty portfolio...**

# In[ ]:


portfolio = Portfolio(name='My New Portfolio')
portfolio.save(overwrite=True)

# **And update your portfolio with the positions you specified earlier:**

# In[ ]:


pm = PortfolioManager(portfolio.id)
pm.update_positions(portfolio_position_sets)

# **Now you can create a factor risk report for your portfolio...**

# In[ ]:


risk_report = FactorRiskReport(risk_model_id=risk_model_id, fx_hedged=True)
risk_report.set_position_source(portfolio.id)
risk_report.save()

# **And schedule all portfolio reports to begin calculating analytics:**

# In[ ]:


pm.schedule_reports()

print(f'Check out your new portfolio in Marquee! View it here: https://marquee.gs.com/s/portfolios/{portfolio.id}/summary')
