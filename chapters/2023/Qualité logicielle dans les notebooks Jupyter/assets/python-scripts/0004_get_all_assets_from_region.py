#!/usr/bin/env python
# coding: utf-8

# Examples require an initialized GsSession and relevant entitlements. External clients need to substitute thier own client id and client secret below. Please refer to [Authentication](https://developer.gs.com/p/docs/institutional/platform/authentication/) for details.

# In[1]:


from gs_quant.session import GsSession
GsSession.use(client_id=None, client_secret=None, scopes=('read_product_data',))

# In[2]:


from gs_quant.api.gs.assets import GsAssetApi
from gs_quant.data import Dataset
import pandas as pd

# In[3]:


ds = Dataset('EDRVOL_PERCENT_STANDARD')  # https://marquee.gs.com/s/developer/datasets/EDRVOL_PERCENT_STANDARD
ids = ds.get_coverage().get('assetId')

# ### Filtering coverage by Asset properties
# A lot of datasets use Marquee Asset Id as symbol dimension. Marquee Asset Service to get more properties for these assets and/or filter by them.
# Below example showcases how to filter assets by Region

# In[4]:


# find all assets from given region (One of "Asia", "Europe", "Americas")
def get_assets_in_region(region, asset_ids):
    step = 500
    size = asset_ids.shape[0]
    assets = []
    # go over all assets in the dataset in batches of 500 and query for region
    for i in range(0, size, step):
        end = min(i + step, size)
        # prepare the query
        query = {'id': asset_ids.values[i: end].tolist(), 'region': region}
        # run the query
        results = GsAssetApi.get_many_assets_data(limit=step, **query)
        assets.extend(results)
    return assets

# In[5]:


filtered_assets = get_assets_in_region('Europe', ids)

# In[6]:


pd.DataFrame(filtered_assets, columns=['id', 'bbid', 'region', 'currency', 'exchange'])
