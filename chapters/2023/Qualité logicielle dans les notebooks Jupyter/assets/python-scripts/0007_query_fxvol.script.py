#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime
from functools import lru_cache
import pandas as pd
from gs_quant.api.gs.assets import GsAssetApi
from gs_quant.data import Dataset
from gs_quant.session import GsSession, Environment

# In[ ]:


# Internals
@lru_cache(maxsize=128)
def _get_coverage(ds):
    print("Fetching coverage", end=" ")
    cov = ds.get_coverage()["assetId"].to_list()
    print("[DONE]")
    return cov

# In[ ]:


def _nest_list(flat_list, nested_size):
    return [flat_list[i:i + nested_size] for i in range(0, len(flat_list), nested_size)]

# In[ ]:


def _get_data_by_asset_parameters(ds, start_date, end_date, pricing_location="NYC", chunk_size=100, **kwargs):
    coverage = _get_coverage(ds)
    assets = []
    asset_chunks = _nest_list(coverage, 5000)
    print("Reading Assets:", end=" ")
    for ch in asset_chunks:
        print("#", end="")
        kwargs["id"] = ch
        assets += GsAssetApi.get_many_assets(["id"], limit=10000, **kwargs)
    print(" [DONE]")

    ids = [a.id for a in assets]
    print("Assets to fetch " + str(len(ids)))
    chunks = _nest_list(ids, chunk_size)
    frames = [pd.DataFrame()]
    print("Reading Data:", end=" ")
    for ch in chunks:
        print("#", end="")
        frames.append(ds.get_data(start_date, end_date, assetId=ch, pricingLocation=pricing_location))
    print(" [DONE]")
    return pd.concat(frames)

# In[ ]:


def get_fxivol(ds, bbid: str = None, tenor=None, strike_reference=None, relative_strike=None, pricing_location="NYC",
               start_date=datetime.date.today(), end_date=datetime.date.today()):
    """
    :param ds: Dataset object
    :param bbid: selected identifier i.e. EURUSD
    :param tenor: tenor of the option i.e. 1y, 5y
    :param strike_reference: one of "delta", "forward", "spot", one supported at at time
    :param relative_strike: moneyness of the option i.e. -5, 0, 5 ... etc Mixing negative and positive are not supported
    :param pricing_location:
    :param start_date: start date of the query
    :param end_date:  end date of the query
    :return: DataFrame with data
    """
    query = dict()
    if bbid is not None:
        query["asset_parameters_call_currency"] = [bbid[0:3], bbid[3:]]
        query["asset_parameters_put_currency"] = [bbid[0:3], bbid[3:]]
    if strike_reference is not None:
        relative_strike_l = relative_strike
        if strike_reference == "delta" and relative_strike is not None:
            if not isinstance(relative_strike, list):
                relative_strike_l = [relative_strike]
            if min(relative_strike_l) < 0 < max(relative_strike_l):
                print("ERROR: Relative strikes should be same sign in the query to work properly")
                return None
            query["asset_parameters_option_type"] = "Put" if relative_strike_l[0] < 0 else "Call"
            query["asset_parameters_strike_price_relative"] = [str(abs(s)) + "D" if s != 0 else "DN" for s in
                                                               relative_strike_l]

    if strike_reference == "forward":
        query["asset_parameters_strike_price_relative"] = "ATMF"
    if strike_reference == "spot":
        query["asset_parameters_strike_price_relative"] = ["ATMS", "Spot", "ATM"]

    if tenor is not None:
        query["asset_parameters_expiration_date"] = tenor

    print(query)
    return _get_data_by_asset_parameters(ds, start_date, end_date, pricing_location=pricing_location, **query)

# In[ ]:


def get_fx_fwd(ds, bbid: str = None, tenor=None, pricing_location="NYC",
               start_date=datetime.date.today(), end_date=datetime.date.today()):
    """
    :param ds: Dataset object
    :param bbid: selected identifier i.e. EURUSD
    :param tenor: tenor of the option i.e. 1y, 5y
    :param pricing_location:
    :param start_date: start date of the query
    :param end_date:  end date of the query
    :return: DataFrame with data
    """
    args = dict()
    if bbid is not None:
        args["asset_parameters_pair"] = bbid
    if tenor is not None:
        args["asset_parameters_expiration_date"] = tenor
    return _get_data_by_asset_parameters(ds, start_date, end_date, pricing_location=pricing_location, **args)

# In[ ]:


# USER DETAILS
CLIENT_ID = None
SECRET = None
GsSession.use(Environment.PROD, client_id=CLIENT_ID, client_secret=SECRET)

# In[ ]:


#SELECT DATASET
FX_VOL_DS = Dataset("FXIVOL_V2_PREMIUM")
start_date = datetime.date(2021, 12, 17)
end_date = datetime.date(2022, 12, 17)


# In[ ]:


vol = get_fxivol(FX_VOL_DS, start_date=start_date, end_date=end_date, bbid="EURUSD", relative_strike=[-10],tenor="5y",
                     strike_reference="delta")
print(vol)

# In[ ]:


vol = get_fxivol(FX_VOL_DS, start_date=start_date, end_date=end_date, bbid="EURUSD", relative_strike=[10],tenor="5y",
                     strike_reference="delta")
print(vol)

# In[ ]:


vol = get_fxivol(FX_VOL_DS, start_date=start_date, end_date=end_date, bbid="USDPLN",
                     strike_reference="delta",
                     tenor="3m")
print(vol)

# In[ ]:


vol = get_fxivol(FX_VOL_DS, start_date=start_date, end_date=end_date, bbid="EURUSD", relative_strike=[0, 10, 15],
                     strike_reference="delta",
                     tenor="1y")
print(vol)

# In[ ]:


vol = get_fxivol(FX_VOL_DS, start_date=start_date, end_date=end_date, bbid="USDPLN",
                     strike_reference="forward",
                     tenor="5y")
print(vol)

# In[ ]:


vol = get_fxivol(FX_VOL_DS, start_date=start_date, end_date=end_date, bbid="USDPLN",
                     strike_reference="spot",
                     tenor="1y")
print(vol)

# In[ ]:



