#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.session import Environment, GsSession

# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None)

# # Factor Models
# 
# The GS Quant `FactorRiskModel` class allows users to access vendor factor models such as Barra. The `FactorRiskModel` interface supports date-based querying of the factor risk model outputs such as factor returns, covariance matrix and specific risk for assets.
# 
# In this tutorial, we’ll look at querying available risk models, their coverage universe, and how to access the returns and volatility of factors in the model. We also show how to query factor exposures (z-scores), specific risk and total risk for a given set of assets in the model's universe.
# 
# The factor returns represent the regression outputs of the model for each day. The definitions of each factor vary depending on the model. More details can be found in the [Marquee Data Catalog](https://marquee.gs.com/s/discover/data-services/catalog?query=factor+risk+model).

# ### Risk Model Coverage
# 
# Popular third party risk models that have been onboarded onto Marquee for programmatic access are below.
# 
# | Risk Model Name                     | Risk Model Id     | Description|
# |-------------------------------------|-------------------|-------------
# | Barra USMED Short (BARRA_USMEDS)    | BARRA_USMEDS      |Barra (MSCI) US Total Market Equity Model for medium-term investors. Includes all styles from the long-term model plus additional factors for investment horizons between 1 month and 1 year (responsive variant).|
# | Barra USSLOW Long (BARRA_USSLOWL)   | BARRA_USSLOWL     |Barra (MSCI) US Total Market Equity Model for long-term investors. Designed with a focus on portfolio construction and reporting for long investment horizons (stable variant).|
# | Barra GEMLT Long (BARRA_GEMLTL)     | BARRA_GEMLTL      |Barra (MSCI) Global Total Market Equity Model for long-term investors. Designed with a focus on portfolio construction and reporting for global equity investors (stable variant).|
# | Barra US Fast (BARRA_USFAST)        | BARRA_USFAST      |Barra (MSCI) US Equity Trading Model for short-term investors. Includes all styles from the medium-term model plus additional factors for shorter investment horizons.|
# | Wolfe Developed Markets All-Cap v1  | WOLFE_QES_DM_AC_1 | Wolfe's Developed Markets All-Cap model is intended for global portfolios with an emphasis on Developed Markets. The model combines next generation factors like short interest and interest rate sensitivity with conventional factors like value and growth.|
# | Wolfe US TMT v2                     | WOLFE_QES_US_TMT_2| Wolfe's US TMT model is intended for sector portfolios. The model uses sector-specific factors in a TMT estimation universe to explain more systematic risk and return relative to a broad market model.|
# | Wolfe Europe All-Cap v2             | WOLFE_QES_EU_AC_2 | Wolfe's Europe All-Cap model is intended for European portfolios with a focus on the developed markets. The model combines next generation factors like short interest and interest rate sensitivity with conventional factors like value and growth.|
# | Wolfe US Healthcare v2              | WOLFE_QES_US_HC_2 | Wolfe's US Healthcare model is intended for sector portfolios. The model uses sector-specific factors in a Healthcare estimation universe to explain more systematic risk and return relative to a broad market model.|
# 
# After selecting a risk model, we can create an instance of the risk model to pull information on the model coverage such as the available dates, asset coverage universe, available factors and model description. The `RiskModelCoverage` enum of the model indicates whether the scope of the universe is Global, Region or Country and the `Term` enum refers to the horizon of the model.

# In[ ]:


from gs_quant.models.risk_model import FactorRiskModel

model_id = 'BARRA_USMEDS'
factor_model = FactorRiskModel.get(model_id)

# check available history for a factor model to decide start and end dates
available_days = factor_model.get_dates()

print(f'Data available for {model_id} from {available_days[0]} to {available_days[-1]}')
print(f'{model_id}:\n - Name: {factor_model.name}\n - Description: {factor_model.description}\n - Coverage: {factor_model.coverage.value}\n - Horizon: {factor_model.term.value}')
print(f'For all info https://marquee.gs.com/v1/risk/models/{model_id}')

# ### Query Factor Data
# 
# The following parameters are required for querying factor data:
# 
# * `start_date` - date or datetime that is a business day
# * `end_date` - date or datetime that is a business day. If an end date is not specified, it will default to the last available date
# * `limit_factor` - A boolean to limit output to only exposed factors. Set to False when not querying data for a particular asset.

# ##### Get Available Factors
# 
# For each model, we can retrieve a list of factors available. Each factor has a `name`, `id`, `type` and `factorCategory`.
# 
# A factor's `factorCategory` can be one of the following:
# * Style - balance sheet and market metrics
# * Industry - an asset's line of business (i.e. Barra uses GICS classification)
# * Country - reference an asset’s exchange country location
# 

# In[ ]:


import datetime as dt
from gs_quant.models.risk_model import FactorRiskModel

model_id = 'BARRA_USMEDS'
factor_model = FactorRiskModel.get(model_id)

available_factors = factor_model.get_factor_data(dt.date(2020, 1, 4)).set_index('identifier')
available_factors.sort_values(by=['factorCategory']).tail()

# ##### Get All Factor Returns
# 
# To query factor returns, we can either use `get_factor_returns_by_name` to retrieve the returns with names or `get_factor_returns_by_id` to get the returns with factor ids. We can leverage [the timeseries package](https://developer.gs.com/docs/gsquant/data/data-analytics/timeseries/) to transform and visualize the results.

# In[ ]:


import matplotlib.pyplot as plt
import datetime as dt
from gs_quant.timeseries import beta
from gs_quant.models.risk_model import FactorRiskModel

model_id = 'BARRA_USMEDS'
factor_model = FactorRiskModel.get(model_id)

factor_returns = factor_model.get_factor_returns_by_name(dt.date(2020, 1, 4))
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

factor_returns[['Growth', 'Momentum', 'Size']].cumsum().plot(title='Factor Performance over Time for Risk Model', ax=ax[0])
factor_beta = beta(factor_returns['Growth'], factor_returns['Momentum'], 63, prices = False)
factor_beta.plot(title='3m Rolling Beta of Growth to Momentum', ax=ax[1])
fig.autofmt_xdate()
plt.show()

# ##### Covariance Matrix
# 
# The covariance matrix represents an N-factor by N-factor matrix with the diagonal representing the variance of each factor for each day. The covariance matrix is in daily variance units.

# In[ ]:


import datetime as dt
import pandas as pd
from gs_quant.models.risk_model import FactorRiskModel

model_id = 'BARRA_USMEDS'
factor_model = FactorRiskModel.get(model_id)

cov_matrix = factor_model.get_covariance_matrix(dt.date(2021, 1, 4), dt.date(2021, 2, 26)) * 100

# set display options below--set max_rows and max_columns to None to return full dataframe
max_rows = 10
max_columns = 7
pd.set_option('display.max_rows', max_rows)
pd.set_option('display.max_columns', max_columns)

# get the last available matrix
round(cov_matrix.loc['2021-02-26'], 6)

# ##### Factor Correlation and Volatility
# 
# The `Factor` Class allows for quick analytics for a specified factor to easily support comparing one factor across different models or to another factor.
# 
# The factor volatility and correlation functions use the covariance matrix for calculations:
# * Volatility is the square root of the diagonal
# * Correlation is derived from the covariance matrix by dividing the cov(x,y) by the vol(x) * vol(y)

# In[ ]:


import datetime as dt
import matplotlib.ticker as mtick
import numpy as np
import matplotlib.pyplot as plt
from gs_quant.models.risk_model import FactorRiskModel

model_id = 'BARRA_USMEDS'
factor_model = FactorRiskModel.get(model_id)

momentum = factor_model.get_factor('Momentum')
growth = factor_model.get_factor('Growth')

start_date = dt.date(2020, 1, 6)
end_date = dt.date(2021, 2, 26)

vol = momentum.volatility(start_date, end_date)
corr = momentum.correlation(growth, start_date, end_date)

# plot
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(vol.index, corr*100, 'g-', label='Momentum vs Growth Correlation (LHS)')
ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.plot(vol.index, vol*1e4, 'b-', label='Momentum Volatility (RHS)')
plt.xticks(vol.index.values[::30])
fig.legend(loc="lower right", bbox_to_anchor=(.75, -0.10))
fig.autofmt_xdate()
plt.title('Momentum vs Growth Historical Factor Analysis')
plt.show()

# ### Query Asset Data
# 
# The factor risk represents the beta coefficient that can be attributed to the model, whereas the specific (residual) risk refers to the error term that is not explained by the model.
# 
# | Measure                    | Definition    |
# |----------------------------|---------------|
# | `Specific Risk`            | Annualized idiosyncratic risk or error term which is not attributable to factors in percent units |
# | `Total Risk`               | Annualized risk which is the sum of specific and factor risk in percent units |
# | `Historical Beta`          | The covariance of the residual returns relative to the model's estimation universe or benchmark (i.e results of a one factor model)  |
# | `Residual Variance`        | Daily error variance that is not explained by the model which is equal to $$\frac{({\frac{\text{Specific Risk}}{100}})^2}{252}$$ |
# | `Universe Factor Exposure` | Z-score for each factor relative to the model's estimation universe |
# 
# We can retrieve an asset universe on a given date by passing in an empty list and a `RiskModelUniverseIdentifierRequest` to the `DataAssetsRequest`.

# ##### Get Risk Model Universe Coverage

# In[ ]:


import datetime as dt
import pandas as pd
from gs_quant.models.risk_model import RiskModelUniverseIdentifierRequest as Identifier, DataAssetsRequest
from gs_quant.models.risk_model import FactorRiskModel


model_id = 'BARRA_USMEDS'
factor_model = FactorRiskModel.get(model_id)

asset_universe_for_request = DataAssetsRequest(Identifier.gsid, []) # entire universe
universe_on_date = factor_model.get_asset_universe(dt.date(2021, 1, 4), assets=asset_universe_for_request)

# set display options below--set max_rows to None to return full list of identifiers
max_rows = 10
pd.set_option('display.max_rows', max_rows)
universe_on_date

# ##### Query Aggregated Risk
# 
# For asset data, we can query for a specific measure or pull data for a list of measures over a range of dates.

# In[ ]:


import datetime as dt
import matplotlib.pyplot as plt
from gs_quant.models.risk_model import FactorRiskModel, DataAssetsRequest, RiskModelUniverseIdentifierRequest as Identifier

model_id = 'BARRA_USMEDS'
factor_model = FactorRiskModel.get(model_id)

asset_bbid = 'AAPL UW'

# get risk
universe_for_request = DataAssetsRequest(Identifier.bbid, [asset_bbid])
specific_risk = factor_model.get_specific_risk(dt.date(2020, 1, 4), dt.date(2021, 2, 24), universe_for_request)
total_risk = factor_model.get_total_risk(dt.date(2020, 1, 4), dt.date(2021, 2, 24), universe_for_request)
factor_risk = total_risk - specific_risk

plt.stackplot(total_risk.index, specific_risk[asset_bbid], factor_risk[asset_bbid], labels=['Specific Risk','Factor Risk'])
plt.title(f'{asset_bbid} Risk')
plt.xticks(total_risk.index.values[::50])
plt.legend(loc='upper right')
plt.gcf().autofmt_xdate()
plt.show()

# ##### Query Factor Exposures (z-scores)
# 
# When querying the asset factor exposures, set the `limit_factor` to True to receive only non zero exposures.

# In[ ]:


import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt

from gs_quant.models.risk_model import FactorRiskModel, DataAssetsRequest, RiskModelUniverseIdentifierRequest as Identifier

model_id = 'BARRA_USMEDS'
factor_model = FactorRiskModel.get(model_id)

asset_bbid = 'AAPL UW'
universe_for_request = DataAssetsRequest(Identifier.bbid, [asset_bbid])

factor_exposures = factor_model.get_universe_factor_exposure(dt.date(2020, 1,4), dt.date(2021, 2, 24), universe_for_request)

available_factors = factor_model.get_factor_data(dt.date(2020, 1, 4)).set_index('identifier')
available_factors.sort_values(by=['factorCategory']).tail()

factor_exposures.columns = [available_factors.loc[x]['name'] for x in factor_exposures.columns]

sns.boxplot(data=factor_exposures[['Beta', 'Momentum', 'Growth', 'Profitability']])
plt.title(f'Distribution of {asset_bbid} Factor Exposures since 1/4/20')
plt.show()

# ##### Query Multiple Asset Measures

# In[ ]:


import datetime as dt
from gs_quant.models.risk_model import FactorRiskModel, Measure, DataAssetsRequest, RiskModelUniverseIdentifierRequest as Identifier

model_id = 'BARRA_USMEDS'
factor_model = FactorRiskModel.get(model_id)

# get multiple measures across a date range for a universe specified
start_date = dt.date(2021, 1, 4)
end_date = dt.date(2021, 2, 24)

asset_bbid = 'AAPL UW'
universe_for_request = DataAssetsRequest(Identifier.bbid, [asset_bbid])

data_measures = [Measure.Universe_Factor_Exposure, Measure.Asset_Universe, Measure.Historical_Beta, Measure.Specific_Risk]
asset_risk_data = factor_model.get_data(data_measures, start_date, end_date, universe_for_request, limit_factors=True)

for i in range(len(asset_risk_data.get('results'))):
    date =  asset_risk_data.get('results')[i].get('date')
    universe = asset_risk_data.get('results')[i].get('assetData').get('universe')
    factor_exposure = asset_risk_data.get('results')[i].get('assetData').get('factorExposure')
    historical_beta = asset_risk_data.get('results')[i].get('assetData').get('historicalBeta')
    specific_risk = asset_risk_data.get('results')[i].get('assetData').get('specificRisk')
    print(f'date: {date}')
    print(f'universe: {universe}')
    print(f'factor id to factor exposure: {factor_exposure}')
    print(f'historical beta: {historical_beta}')
    print(f'specific risk: {specific_risk}')
    print('\n')


