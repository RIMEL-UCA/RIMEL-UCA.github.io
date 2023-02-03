#!/usr/bin/env python
# coding: utf-8

# Examples require an initialized GsSession and relevant entitlements. External clients need to substitute thier own client id and client secret below. Please refer to [Authentication](https://developer.gs.com/p/docs/institutional/platform/authentication/) for details.

# In[1]:


from gs_quant.session import GsSession
GsSession.use(client_id=None, client_secret=None, scopes=('read_product_data',))

# ## How to query data 
# The Data APIs support many ways to query datasets to intuitively fetch only the data users need.
# More details on [Querying Data](https://developer.gs.com/p/docs/services/data/data-access/query-data/) can be found in the documentation

# In[2]:


from datetime import date, timedelta, datetime
from gs_quant.data import Dataset
import pydash

# Data in Marquee is available in the form of Datasets (collections of homogenous data). Each Dataset has a set of entitlements, a fixed schema, and assets in coverage.

# In[3]:


dataset_id = 'FXIVOL_STANDARD' # https://marquee.gs.com/s/developer/datasets/FXIVOL_STANDARD
ds = Dataset(dataset_id)

# Data for limited number of assets or spanning a small time frame can be queried in one go by specifying the assets to query and date/time range.

# In[4]:


start_date = date(2019, 1, 15)
end_date = date(2019, 1, 18)

data = ds.get_data(start_date, end_date, bbid=['EURCAD'])
data.head()

# Instead of a range, one can also specify a set of date/times to get data for just those specific date/times

# In[5]:


data = ds.get_data(dates=[date(2019, 1, 15), date(2019, 1, 18)],
                   bbid=['EURCAD'])
data.head()

# For larger number of assets or for longer time ranges, 
# we recommend iterating over assets and time to avoid hitting API query limits. 

# In[6]:


# loop over assets
def iterate_over_assets(dataset, coverage, start, end, batch_size=5, query_dimension='assetId', delta=timedelta(days=6)):
    for ids in pydash.chunk(coverage[query_dimension].tolist(), size=batch_size):
        print('iterate over assets', ids)
        iterate_over_time(start, end, ids, dataset, delta=delta, query_dimension=query_dimension)

# loop over time
def iterate_over_time(start, end, ids, dataset, delta=timedelta(days=6), query_dimension='assetId'):
    iter_start = start
    while iter_start < end:
        iter_end = min(iter_start + delta, end)
        print('time iteration since', iter_start, 'until', iter_end)
        data = dataset.get_data(iter_start, iter_end, **{query_dimension: ids})
        # Add your code here to make use of fetched data
        
        iter_start = iter_end

# In[7]:


dataset_id = 'EDRVOL_PERCENT_V1_STANDARD'  # https://marquee.gs.com/s/developer/datasets/EDRVOL_PERCENT_V1_STANDARD 
ds = Dataset(dataset_id)

coverage = ds.get_coverage()

iterate_over_assets(ds, coverage, date(2021, 5, 1), date(2021, 5, 31), batch_size=5)

# Similar approach can be used to download all data of a dataset

# In[ ]:


coverage = ds.get_coverage(include_history=True)
coverage = coverage.sort_values(by='historyStartDate', axis=0)
start_date = datetime.strptime(coverage['historyStartDate'].values[0], '%Y-%m-%d').date()

# warning: long running operation
iterate_over_assets(ds, coverage, start_date, date.today())

