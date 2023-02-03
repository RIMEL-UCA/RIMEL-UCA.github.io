#!/usr/bin/env python
# coding: utf-8

# # Portfolio Optimizer
# 
# The portfolio optimizer brings together the power of the Axioma Portfolio Optimizer with Marquee's risk analytics infrastructure
# to make minimizing your portfolio's factor risk possible within the same ecosystem.
# 
# To use the optimizer, you must have a license to the Axioma Portfolio Optimizer. Please reach out to the
# [Marquee sales team](mailto:gs-marquee-sales@ny.email.gs.com?Subject=Portfolio Optimizer Trial Request)
# to learn more about how to get a license or how to bring an existing license to Marquee.
# 
# ## Step 1: Authenticate and Initialize Your Session
# 
# First you will import the necessary modules and add your client id and client secret.

# In[ ]:


import datetime as dt

import pandas as pd
from IPython.display import display

from gs_quant.markets.optimizer import OptimizerUniverse, FactorConstraint, MaxFactorProportionOfRiskConstraint, \
    AssetConstraint, \
    SectorConstraint, OptimizerSettings, OptimizerStrategy, OptimizerConstraints, OptimizerObjective, OptimizerType
from gs_quant.markets.position_set import Position, PositionSet
from gs_quant.markets.securities import Asset, AssetIdentifier
from gs_quant.models.risk_model import FactorRiskModel
from gs_quant.session import GsSession, Environment
from gs_quant.target.hedge import CorporateActionsTypes

# Enter client id and secret
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('read_product_data',))

print('GS Session initialized.')

# ## Step 2: Define Your Initial Position Set
# 
# Use the `PositionSet` class in GS Quant to define the initial holdings to optimize.
# You can define your positions as a list of identifiers with quantities or, alternatively, as a
# list of identifiers with weights, along with a reference notional value.
# 
# *GS Quant will resolve all identifiers (Bloomberg IDs, SEDOLs, RICs, etc) historically as of the optimization date.*

# In[ ]:


position_set = PositionSet(
    date=dt.date(day=23, month=9, year=2022),
    positions=[
        Position(identifier='AAPL UW', quantity=26),
        Position(identifier='GS UN', quantity=51)
    ]
)

# ## Step 3: Define Your Optimizer Universe
# 
# An optimizer universe corresponds to the assets that can be used when constructing an optimization, which can be created
# using the `OptimizerUniverse` class:
# 
# | Parameter | Description | Type| Default Value|
# |-----------------|---------------|-------------|-------------
# | `assets`      | Assets to include in the universe. |`List[Asset]`| N/A |
# | `explode_composites`     | Explode indices, ETFs, and baskets and include their underliers in the universe. |`boolean`| `True` |
# | `exclude_initial_position_set_assets`       | Exclude assets in the initial holdings from the optimization. | `boolean` | `False` |
# | `exclude_corporate_actions_types`     | Set of of corporate actions to be excluded in the universe. |`List[CorporateActionsTypes]`| `[]` |
# | `exclude_hard_to_borrow_assets`       | Exclude hard to borrow assets from the universe. | `boolean` | `False` |
# | `exclude_restricted_assets`       | Exclude assets on restricted trading lists from the universe. | `float` | `False` |
# | `min_market_cap`       | Lowest market cap allowed for any universe constituent. | `float` | `None` |
# | `max_market_cap`       | Highest market cap allowed for any universe constituent. | `float` | `None` |

# In[ ]:


universe = OptimizerUniverse(
    assets=[Asset.get('SPX', AssetIdentifier.BLOOMBERG_ID)],
    explode_composites=True,
    exclude_corporate_actions_types=[CorporateActionsTypes.Mergers]
)

# ## Step 4: Define Your Risk Model and Factor Risk Constraints
# 
# You can run the optimizer using a factor risk model of your choice, so long as you have a license for it, by leveraging
# the `FactorRiskModel` class. For any factor in the risk model, you can set more granular constraints on the optimized
# portfolio's exposure to the factor.
# 
# If you'd like to limit your optimized portfolio's factor proportion of risk,
# you can do so with the `MaxFactorProportionOfRiskConstraint` class.
# 
# In this example, let's use the Axioma AXUS4S model and limit the final exposure to Volatility be $10,000 and
# the final exposure of Market Sensitivity to be 5,000. Let's also set a constraint so that the final positions don't have
# more than 30% of its total risk to be factor risk.

# In[ ]:


risk_model = FactorRiskModel.get('AXIOMA_AXUS4S')

factor_constraints = [
    FactorConstraint(risk_model.get_factor('Volatility'), 10000),
    FactorConstraint(risk_model.get_factor('Market Sensitivity'), 5000)
]

prop_of_risk_constraint = MaxFactorProportionOfRiskConstraint(30)

# ## Step 5: Define Other Optimization Constraints
# 
# Outside factor-specific constraints, it's also possible to limit the holding value of individual assets, assets
# belonging to a particular GICS sector, and/or assets in a particular country of domicile in the optimization.
# 
# In this example, let's constrain the optimization to have 0-5% Microsoft and Twitter and limit the optimization's notional
# coming from Energy and Health Care assets to each be 0-30%.

# In[ ]:


asset_constraints = [
    AssetConstraint(Asset.get('MSFT UW', AssetIdentifier.BLOOMBERG_ID), 0, 5),
    AssetConstraint(Asset.get('TWTR UN', AssetIdentifier.BLOOMBERG_ID), 0, 5)
]

sector_constraints = [
    SectorConstraint('Energy', 0, 30),
    SectorConstraint('Health Care', 0, 30)
]

# ## Step 6: Configure Your Optimization Settings
# 
# All other settings for the optimization can be set via the `OptimizerSettings` class:
# 
# | Parameter          | Description | Type| Default Value|
# |--------------------|---------------|-------------|-------------
# | `notional`         | Max gross notional value of the optimization |`float`| `10000000` |
# | `allow_long_short` | Allow a long/short optimization |`boolean`| `False` |
# | `min_names`        | Minimum number of assets in the optimization |`float`| `0` |
# | `max_names`        | Maximum number of assets in the optimization |`float`| `100` |
# | `min_weight_per_constituent`       | Minimum weight of each constituent in the optimization |`float`| None |
# | `max_weight_per_constituent`       | Maximum weight of each constituent in the optimization |`float`| None |
# | `max_adv`        | Maximum average daily volume of each constituent in the optimization (in percent) |`float`| `15` |
# 

# In[ ]:


settings = OptimizerSettings(allow_long_short=False)

# ## Step 7: Create and Run a Strategy
# 
# It's finally time to take all these parameters and construct an optimizer strategy using the `OptimizerStrategy` class:
# 

# In[ ]:


constraints = OptimizerConstraints(
    asset_constraints=asset_constraints,
    sector_constraints=sector_constraints,
    factor_constraints=factor_constraints,
    max_factor_proportion_of_risk=prop_of_risk_constraint
)

strategy = OptimizerStrategy(
    initial_position_set=position_set,
    constraints=constraints,
    settings=settings,
    universe=universe,
    risk_model=risk_model,
    objective=OptimizerObjective.MINIMIZE_FACTOR_RISK
)

strategy.run(optimizer_type=OptimizerType.AXIOMA_PORTFOLIO_OPTIMIZER)

optimization = strategy.get_optimization() # Returns just the optimization results as a PositionSet object
optimized_position_set = strategy.get_optimized_position_set()

print('OPTIIMZATION RESULTS')
result = [{'Asset': p.identifier, 'Quantity': p.quantity} for p in optimization.positions]
display(pd.DataFrame(result))

# ## Step 8: Create a Basket or Portfolio With Your Results
# 
# Now that you have a position set for your optimization and your optimized position set, you can upload either to a basket or
# portfolio by following the following tutorials:
# 
# - [Create a Basket](https://nbviewer.org/github/goldmansachs/gs-quant/blob/master/gs_quant/documentation/06_baskets/tutorials/Basket%20Create.ipynb)
# - [Create a Portfolio](https://nbviewer.org/github/goldmansachs/gs-quant/blob/master/gs_quant/documentation/10_one_delta/scripts/portfolios/Create%20New%20Portfolio.ipynb)
# 
# *Other questions? Reach out to the [Portfolio Analytics team](mailto:gs-marquee-analytics-support@gs.com)!*
# 
