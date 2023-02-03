#!/usr/bin/env python
# coding: utf-8

# # GS Quant Meets Markets x MSCI

# #### Step 1: Import Modules

# In[ ]:


# Import modules
from typing import List

from gs_quant.api.utils import ThreadPoolManager
from gs_quant.data import Dataset
from gs_quant.api.gs.assets import GsAssetApi
from gs_quant.models.risk_model import FactorRiskModel, DataAssetsRequest
from functools import partial

from gs_quant.markets.baskets import Basket
from gs_quant.markets.indices_utils import ReturnType
from gs_quant.markets.position_set import PositionSet
from gs_quant.session import Environment, GsSession
import matplotlib.pyplot as plt
import datetime as dt
from dateutil.relativedelta import relativedelta
import pandas as pd

# #### Step 2: Authenticate

# In[ ]:


# Initialize session -- for external users, input client id and secret below
client = None
secret = None
GsSession.use(Environment.PROD, client_id=client, client_secret=secret, scopes='')

# #### Step 3: Implement basic functions to fetch coverage universe, ratings, factor & liquidity data

# In[ ]:


# Initialize functions

def batch_liquidity(dataset_id: str, asset_ids: list, day: dt.date, size: int=200) -> pd.DataFrame:
    data = Dataset(dataset_id)
    tasks = [partial(data.get_data, day, day, assetId=asset_ids[i:i+size]) for i in range(0, len(asset_ids), size)]
    results = ThreadPoolManager.run_async(tasks)
    return pd.concat(results)


def batch_ratings(dataset_id: str, gsid_ids: list, day: dt.date, fallback_month, filter_value: str= None, size: int=5000) -> pd.DataFrame:
    data = Dataset(dataset_id)
    start_date = day - relativedelta(month=fallback_month)
    tasks = [partial(data.get_data, start_date=start_date, gsid=gsid_ids[i:i+size], rating=filter_value) for i in range(0, len(gsid_ids), size)] if filter_value else \
            [partial(data.get_data, start_date=start_date, gsid=gsid_ids[i:i + size]) for i in range(0, len(gsid_ids), size)]
    results = ThreadPoolManager.run_async(tasks)
    return pd.concat(results)


def batch_asset_request(day: dt.date, gsids_list: list, limit: int=1000) -> list:
    date_time = dt.datetime.combine(day, dt.datetime.min.time())
    fields = ['gsid', 'bbid', 'id', 'delisted', 'assetClassificationsIsPrimary']
    tasks = [partial(GsAssetApi.get_many_assets_data, gsid=gsids_list[i:i + limit], as_of=date_time, limit=limit*10, fields=fields) for i in range(0, len(gsids_list), limit)]
    results = ThreadPoolManager.run_async(tasks)
    return [item for sublist in results for item in sublist]


def get_universe_with_xrefs(day: dt.date, model: FactorRiskModel) -> pd.DataFrame:
    print(f'---------Getting risk {model.id} coverage universe on {day}------------')
    # get coverage universe on date
    universe = model.get_asset_universe(day, day).iloc[:, 0].tolist()
    print(f'{len(universe)} assets in {model.id} on {day} that map to gsids')
    # need to map from id -> asset_identifier on date
    asset_identifiers = pd.DataFrame(batch_asset_request(day, universe))
    print(f'{len(asset_identifiers)} assets found')

    asset_identifiers = asset_identifiers[asset_identifiers['assetClassificationsIsPrimary'] != 'false']
    print(f'{len(asset_identifiers)} asset xrefs after is not primary dropped')
    asset_identifiers = asset_identifiers[asset_identifiers['delisted'] != 'yes']
    print(f'{len(asset_identifiers)} asset xrefs after delisted assets are dropped')

    asset_identifiers = asset_identifiers[['gsid', 'bbid', 'id']].set_index('gsid')
    asset_identifiers = asset_identifiers[~asset_identifiers.index.duplicated(keep='first')]  # remove duplicate gsids
    asset_identifiers.reset_index(inplace=True)
    print(f'{len(asset_identifiers)} positions after duplicate gsids removed')

    return pd.DataFrame(asset_identifiers).set_index('id')


def get_and_filter_ratings(day: dt.date, gsid_list: List[str], filter_value: str = None) -> list:
    # get ratings of assets from the ratings dataset and only keep 'Buy' ratings
    print(f'---------Filtering coverage universe by rating: {filter_value}------------')
    fallback_month = 3
    ratings_df = batch_ratings('RATINGS_CL', gsid_list, day, fallback_month, filter_value)
    df_by_asset = [ratings_df[ratings_df['gsid'] == asset] for asset in set(ratings_df['gsid'].tolist())]
    most_recent_rating = pd.concat([df.iloc[-1:] for df in df_by_asset])
    print(f'{len(most_recent_rating)} unique assets with ratings after filtering applied')
    return list(most_recent_rating['gsid'].unique())


def get_and_filter_factor_exposures(day: dt.date, identifier_list: List[str], factor_model: FactorRiskModel, factors: List[str]= [] , filter_floor: int = 0.5) -> pd.DataFrame:
    # get factor info and filter by factors
    print(f'---------Filtering coverage universe by factors: {factors}------------')
    available_factors = factor_model.get_factor_data(day).set_index('identifier')
    req = DataAssetsRequest('gsid', identifier_list)
    factor_exposures = factor_model.get_universe_factor_exposure(day, day, assets=req).fillna(0)

    factor_exposures.columns = [available_factors.loc[x]['name'] for x in factor_exposures.columns]
    factor_exposures = factor_exposures.droplevel(1)
    print(f'{len(factor_exposures)} factor exposures available')
    for factor in factors:
        factor_exposures = factor_exposures[factor_exposures[factor] >= filter_floor]
        print(f'{len(factor_exposures)} factor exposures returned after filtering by {factor} with floor exposure {filter_floor}')
    return factor_exposures


def get_and_filter_liquidity(day: dt.date, asset_ids: List[str], filter_floor: int = 0) -> pd.DataFrame:
    # get mdv22Day liquidity info and take assets above average adv
    print(f'---------Filtering coverage universe by liquidity value: {filter_floor}------------')
    liquidity = batch_liquidity('GSEOD', asset_ids, day).set_index("assetId")
    print(f'{len(liquidity)} liquidity data available for requested universe')
    if filter_floor:
        liquidity = liquidity[liquidity['mdv22Day'] >= filter_floor]
        print(f'{len(liquidity)} unique assets with liquidity data returned after filtering')
    return liquidity


def backtest_strategy(day: dt.date, position_set: List[dict], risk_model_id: str):
    # make a request to pretrade liquidity to get backtest timeseries
    print(f'---------Backtesting strategy------------')
    query = {"currency":"USD",
             "notional": 1000000,
             "date": day.strftime("%Y-%m-%d"),
             "positions":position_set,
             "participationRate":0.1,
             "riskModel":risk_model_id,
             "timeSeriesBenchmarkIds":[],
             "measures":["Time Series Data"]}
    result = GsSession.current._post('/risk/liquidity', query)
    result = result.get("timeseriesData")
    return result


def graph_df_list(df_list, title):
    for df in df_list:
        plt.plot(df[0], label=df[1])
    plt.legend(title='Measures')
    plt.xlabel('Date')
    plt.title(title)
    plt.show()

# #### Step 4: Strategy Implementation
# 
# Proposed Methodology
# - Starting universe: Chosen risk model coverage universe
# - High Conviction names: Retain GS "Buy" ratings only
# - High ESG names: Retain high ESG scores only, using BARRA GEMLTL ESG model
# - High Profitability names: Retain high Profitability scores only, using BARRA GEMLTL ESG model
# - Liquidity adjustment: Removing the tail of illiquid names
# - Weighting: MDV-based weighting

# In[ ]:


# Get risk model and available style factors
start = dt.datetime.now()

# Get risk model
model_id = "BARRA_GEMLTL_ESG"
factor_model = FactorRiskModel.get(model_id)

# Get last date of risk model data
date = factor_model.get_most_recent_date_from_calendar() - dt.timedelta(1)
print(f"-----Available style factors for model {model_id}-----")
factor_data = factor_model.get_factor_data(date, date)
factor_data = factor_data[factor_data['factorCategoryId'] == 'RI']
print(factor_data['name'])

# In[ ]:


# Get universe
mqid_to_id = get_universe_with_xrefs(date, factor_model)

# In[ ]:


# Get available ratings for past 3 months and return most recent ratings data per asset
ratings_filter = 'Buy'
ratings_universe = get_and_filter_ratings(date, mqid_to_id['gsid'].tolist(), filter_value=ratings_filter)

# In[ ]:


# Pass in factors to filter by
factors = ['ESG', 'Profitability']
filter_floor = 0.5

exposures = get_and_filter_factor_exposures(date, ratings_universe, factor_model, factors=factors, filter_floor=filter_floor)
ids = mqid_to_id.reset_index().set_index("gsid")
exposures = exposures.join(ids, how='inner')

# In[ ]:


# Filter by liquidity, which takes in the MQ Id
asset_ids = exposures['id'].tolist()
liquidity_floor = 1000000
liquidity = get_and_filter_liquidity(date, asset_ids, filter_floor=liquidity_floor)
liquidity = liquidity.join(mqid_to_id, how='inner')

# In[ ]:


# Get weights as ADV / total ADV
total_adv = sum(list(liquidity['mdv22Day']))
liquidity['weights'] = liquidity['mdv22Day'] / total_adv

# #### Step 5: Backtest Strategy

# In[ ]:


# Backtest composition
backtest_set = [{'assetId': index, "weight": row['weights']} for index, row in liquidity.iterrows()]
position_set = [{'bbid': row['bbid'], "weight": row['weights']} for index, row in liquidity.iterrows()]
print("Position set for basket create: ")
print(pd.DataFrame(position_set))
print(f'Total time to build position set with requested parameters {dt.datetime.now() - start}')


backtest = backtest_strategy(date, backtest_set, model_id)
print("Available measures to plot for backtested strategy: ")
measures = list(backtest[0].keys())
measures.remove("name")
print(measures)

# In[ ]:


# Graph Normalized Performance
np = ['normalizedPerformance']
series_to_plot = []
for measure in np:
    timeseries = backtest[0].get(measure)
    timeseries = {dt.datetime.strptime(data[0], "%Y-%m-%d"): data[1] for data in timeseries}
    timeseries = (pd.Series(timeseries), measure)
    series_to_plot.append(timeseries)

graph_df_list(series_to_plot, "Normalized Performance")

# In[ ]:


# Plot many measures
measures.remove("netExposure")
measures.remove("cumulativePnl")
measures.remove("maxDrawdown")

series_to_plot = []
for measure in measures:
    timeseries = backtest[0].get(measure)
    timeseries = {dt.datetime.strptime(data[0], "%Y-%m-%d"): data[1] for data in timeseries}
    timeseries = (pd.Series(timeseries), measure)
    series_to_plot.append(timeseries)

graph_df_list(series_to_plot, "Backtested Strategy Measures")

# #### Step 6: Basket Creation

# In[ ]:


# Create basket with positions
my_basket = Basket()
my_basket.name = 'Basket Name'
my_basket.ticker = 'Basket Ticker'
my_basket.currency = 'USD'
my_basket.return_type = ReturnType.PRICE_RETURN
my_basket.publish_to_bloomberg = True
my_basket.publish_to_reuters = True
my_basket.publish_to_factset = False
data=[]
for row in position_set:
    data.append([row['bbid'], row['weight']])

positions_df = pd.DataFrame(data, columns=['identifier', 'weight'])
position_set = PositionSet.from_frame(positions_df)
my_basket.position_set = position_set

my_basket.get_details() # we highly recommend verifying the basket state looks correct before calling create!

# In[ ]:


# Publish basket
my_basket.create()
my_basket.poll_status(timeout=10000, step=20) # optional: constantly checks create status until report succeeds, fails, or the poll times out (this example checks every 20 seconds for 2 minutes)
my_basket.get_url() # will return a url to your Marquee basket page ex. https://marquee.gs.com/s/products/MA9B9TEMQ2RW16K9/summary






