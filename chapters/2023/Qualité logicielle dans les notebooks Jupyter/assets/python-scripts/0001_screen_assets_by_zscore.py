#!/usr/bin/env python
# coding: utf-8

# 
# ## Query Factor data and screen for different maximum and minimum thresholds

# In[ ]:


# import statements
from typing import Dict, List, Union
import pandas as pd
from gs_quant.markets.factor import Factor
import logging

from gs_quant.session import GsSession, Environment

from gs_quant.models.risk_model import FactorRiskModel, DataAssetsRequest, RiskModelUniverseIdentifierRequest as Identifier, \
    ReturnFormat

# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('read_product_data run_analytics',))


# define a FactorThreshold class that takes in a minimum and maximum
class FactorThreshold:
    def __init__(self, factor: Factor, min_threshold: float = None, max_threshold: float = None):
        self.factor = factor
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

# define a helper function to do the heavy lifting
def get_screened_universe(assets_zscore: pd.DataFrame,
                          factors: List[FactorThreshold],
                          use_absolute_value: bool = True,
                          format: ReturnFormat = ReturnFormat.DATA_FRAME) -> Union[List[Dict], pd.DataFrame]:
    """ Get universe zscores with exposures based on min and max threshold of exposures to factor list

    :param assets_zscore: a pandas df of assets zscore to factors
    :param factors: list of factors for screening
    :param use_absolute_value: whether to threshold based on absolute value of factor exposure
    :param format: which format to return the results in

    :return: factor exposure for assets requested based on factor list input and max and min threshold values
    """

    factors_in_zscore = assets_zscore.columns.tolist()
    wrong_factors = [f.factor.name for f in factors if f.factor.name not in factors_in_zscore]
    if wrong_factors:
        logging.warning(f'Factors: {wrong_factors} not in the asset zscore input, will be filtered out..')

    if assets_zscore.empty:
        assets_zscore_return = {} if format == ReturnFormat.JSON else assets_zscore
        return assets_zscore_return

    factors_to_filter = [f for f in factors if f.factor.name not in wrong_factors]
    for f in factors_to_filter:
        if f.min_threshold is not None:
            assets_zscore = assets_zscore[abs(assets_zscore[f.factor.name]) >= f.min_threshold] if use_absolute_value else \
                assets_zscore[assets_zscore[f.factor.name] >= f.min_threshold]
        if f.max_threshold is not None:
            assets_zscore = assets_zscore[abs(assets_zscore[f.factor.name]) <= f.max_threshold] if use_absolute_value else \
                assets_zscore[assets_zscore[f.factor.name] <= f.max_threshold]
    if format == ReturnFormat.JSON:
        assets_zscore = assets_zscore.to_dict()
    return assets_zscore


# get the model and dates available
model_id = 'BARRA_USMEDS'
factor_model = FactorRiskModel.get(model_id)

# get multiple measures across a date range for a universe specified
dates = factor_model.get_dates()
latest_date = dates.pop()

total_factors = factor_model.get_many_factors(start_date=latest_date, end_date=latest_date)
style_factors = [f for f in total_factors if f.type != 'Category' and f.category == "Style"]
print([f.name for f in style_factors])

# set maximum and minimum zscore threshold for the factors
growth = factor_model.get_factor('Growth')
beta = factor_model.get_factor('Beta')

style_factor_thresholds = [
    FactorThreshold(factor=growth, min_threshold=0, max_threshold=1),
    FactorThreshold(factor=beta, min_threshold=-1, max_threshold=0)
]


assets = DataAssetsRequest(Identifier.sedol, [])
assets_to_zscore = factor_model.get_universe_factor_exposure(
    start_date=latest_date,
    end_date=latest_date,
    assets=assets,
    get_factors_by_name=True
).reset_index(level=1, drop=True)

print(assets_to_zscore.head())


screened = get_screened_universe(assets_to_zscore, style_factor_thresholds)
print(screened.head())
