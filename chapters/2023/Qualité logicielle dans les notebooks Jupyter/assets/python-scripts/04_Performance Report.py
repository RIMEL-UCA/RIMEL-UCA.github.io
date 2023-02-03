#!/usr/bin/env python
# coding: utf-8

# ### Pull Performance Report Results with GS Quant
# 
# First authenticate your session and then get your portfolio's performance report:

# In[ ]:


import datetime as dt
import pandas as pd
import warnings

from IPython.display import display

from gs_quant.markets.portfolio_manager import PortfolioManager
from gs_quant.markets.securities import Asset, AssetIdentifier
from gs_quant.session import GsSession, Environment

GsSession.use(Environment.PROD)
warnings.filterwarnings("ignore", category=RuntimeWarning)

pm = PortfolioManager('MPWQQ8B05FKPCCH6')
performance_report = pm.get_performance_report()
start_date = performance_report.earliest_start_date
end_date = performance_report.latest_end_date

# Now that we have our performance report, we can leverage the unique functionalities of the PerformanceReport class to pull daily exposure and PnL data:

# In[ ]:


all_exposures = performance_report.get_many_measures(
    start_date=start_date,
    end_date=end_date,
    measures=['pnl', 'grossExposure', 'netExposure']
)

print(all_exposures)
all_exposures.plot(title='Performance Breakdown')

# Now let's pull Brinson Attribution data to analyze the PnL of your portfolio compared to a benchmark, which can be any equity ETF, Index, or Basket in Marquee:

# In[ ]:


asset = Asset.get(id_value='MXWO', id_type=AssetIdentifier.TICKER)

brinson_attribution_results = performance_report.get_brinson_attribution(
    benchmark=asset.get_marquee_id(),
    include_interaction=True,
    start_date=start_date,
    end_date=end_date
)

display(pd.DataFrame(brinson_attribution_results))

# 
