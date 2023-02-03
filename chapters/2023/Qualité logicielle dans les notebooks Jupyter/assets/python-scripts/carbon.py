#!/usr/bin/env python
# coding: utf-8

# ## Pull Marquee Carbon Analytics Data with GS Quant
# 
# ### Permission Prerequisites
# 
# To execute all the code in this tutorial, you will need the following application scopes:
# - **read_financial_data**
# 
# ## Authenticate and Initialize Your Session
# 
# First you will import the necessary modules and add your client id and client secret.

# In[ ]:


from IPython.display import display
from gs_quant.api.gs.assets import GsAssetApi
from gs_quant.api.gs.carbon import CarbonCoverageCategory, CarbonEmissionsIntensityType

from gs_quant.markets.portfolio_manager import PortfolioManager
from gs_quant.session import GsSession, Environment

client = None
secret = None
scopes = None

## External users must fill in their client ID and secret below and comment out the line below

#client = 'ENTER CLIENT ID'
#secret = 'ENTER CLIENT SECRET'
#scopes = ('read_financial_data',)

GsSession.use(
    Environment.PROD,
    client_id=client,
    client_secret=secret,
    scopes=scopes
)

print('GS Session initialized.')

# ## Define Your Entity
# 
# Carbon Analytics can be pulled from any object that inherits from the `PositionedEntity` class, such as `PortfolioManager`, `Index` or `Basket`. In this example, we will get analytics for a Marquee portfolio.

# In[ ]:


pm = PortfolioManager('ENTER PORTFOLIO ID')

# ## Pull Carbon Analytics
# 
# There are various parameters each of these methods take in.
#  - reporting_year - takes 'Latest' and last 4 complete years i.e., from T-2 to T-5. defaults to Latest
#  - currency - uses 'Currency' enum, defaults to the entity currency.
#  - include_estimates - Parameter to choose if estimated emissions are to be included or not, defaults to false.
#  - use_historical_data - Parameter to choose historical positions or backcast latest composition, defaults to false.
#  - normalize_emissions - Parameter to normalize entity notional to 1,000,000 in denominated currency passed.
#  - analytics_view - Parameter to view analytics using long component or short component of the portfolio
#  - scope - uses 'CarbonScope' enum with totalGHG, scope1, scope2 values, defaults to totalGHG.
#  - coverage_category - category for data coverage, uses 'CarbonCoverageCategory' enum with weights and numberOfCompanies, defaults to weights.
#  - target_coverage_category - category for SBTI and Net Zero Targets, uses 'CarbonTargetCoverageCategory' enum with portfolioEmissions and capitalAllocated, defaults to portfolioEmissions.
#  - classification - classification to group financed emissions, uses CarbonEmissionsAllocationCategory with sector, industry and region, defaults to sector
#  - intensity_metric - intensity metric to query attribution for, uses CarbonEmissionsIntensityType with enterprise value, marketcap and revenue. defaults to enterprise value.
#  - benchmark_id - Marquee identifier for the benchmark to do attribution analysis with.

# ### Data Coverage
# 
# Pull the data coverage for a reporting year based on weights or number of companies in the entity. You can choose to include estimated emissions, choose between using historical compositions or backcasting latest composition

# In[ ]:


print('Data Coverage:')
display(pm.get_carbon_coverage(include_estimates=True, coverage_category=CarbonCoverageCategory.NUMBER_OF_COMPANIES))

# ### SBTI and Net Zero Emissions Target Coverage
# 
# Pull Science Based Target Coverage and Net Zero Emissions Target Coverage for a reporting year based on capital allocated or portfolio emissions.

# In[ ]:


print('Science Based Target and Net Zero Emissions Target Coverage:')
display(pm.get_carbon_sbti_netzero_coverage(include_estimates=True))

# ### Financed Emissions and Emissions Intensity profile
# 
# Pull Financed Emissions profile and intensity metrics for a reporting year in respective denomination. Other parameters are to include estimates, use historical data, normalize emissions.

# In[ ]:


print('Financed Emissions and Intensity Profile:')
display(pm.get_carbon_emissions(include_estimates=True))

# ### Financed Emissions by sector, industry and region
# 
# Aggregate financed emissions and capital for each of the categories for a reporting year in respective denomination. We can pass a scope to look at specific scope.

# In[ ]:


print('Financed Emissions by Sector:')
display(pm.get_carbon_emissions_allocation(include_estimates=True))

# ### Attribution analysis with benchmark
# 
# Pull up brinson attribution analysis for sector allocation and security selection with respect to benchmark. Pass in the benchmark id, intensity type.
# 
# Benchmark can be either an asset id or a portfolio id. For an asset, get the asset id by resolving the identifier and, for a portfolio use the portfolio id.
# 
# Resolve asset id from identifier:

# In[ ]:


identifier = 'SPX'
mqids = GsAssetApi.resolve_assets(identifier=[identifier], fields=['id'], limit=1)
try:
    benchmark_id = mqids[identifier][0]['id']
except:
    raise ValueError('Error in resolving the following identifier: ' + identifier)

print('Attribution Analysis:')
display(pm.get_carbon_attribution_table(benchmark_id=benchmark_id, include_estimates=True, intensity_metric=CarbonEmissionsIntensityType.EI_REVENUE))

# You can also pull up the whole analytics using 'get_carbon_analytics()'.
# 
# *Other questions? Reach out to the [Portfolio Analytics team](mailto:gs-marquee-analytics-support@gs.com)!*
