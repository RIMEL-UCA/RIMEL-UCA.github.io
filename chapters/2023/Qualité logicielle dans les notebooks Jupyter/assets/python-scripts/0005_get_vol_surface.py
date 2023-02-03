#!/usr/bin/env python
# coding: utf-8

# Examples require an initialized GsSession and relevant entitlements. External clients need to substitute thier own client id and client secret below. Please refer to [Authentication](https://developer.gs.com/p/docs/institutional/platform/authentication/) for details.

# In[1]:


from gs_quant.session import GsSession
GsSession.use(client_id=None, client_secret=None, scopes=('read_product_data',))

# In[2]:


from gs_quant.data import Dataset
import datetime as dt

# The entire vol surface is published for each snap of the data, so we don’t have to pull diffs and reconstruct the surface

# In[3]:


dataset_id = 'EDRVOL_PERCENT_V1_STANDARD' # https://marquee.gs.com/s/developer/datasets/EDRVOL_PERCENT_STANDARD
ds = Dataset(dataset_id)

# Get the latest available datapoint.

# In[4]:


last = ds.get_data_last(as_of=dt.date.today(), bbid=['SPX'])
last

# We can use this to identify the timestamp of the latest surface and then query for the full surface matching this timestamp

# In[5]:


snap_time = last.index[0].date()  # the dataframe is indexed on date
df = ds.get_data(bbid=['SPX'], start=snap_time, end=snap_time, strikeReference='forward')

df.head()

# The dataframe can be consumed directly or can be serialized for consumption by a different process

# In[ ]:


with open(f'SPX_{snap_time}_curve.csv', 'w') as f:
    df.to_csv(f)

# Above process can be abstracted to a function

# In[6]:


def get_latest_vol_surface(dataset, bbid, strike_reference='delta', intraday=False):
    # Get the date/time of the most recent snap
    as_of = dt.datetime.now() if intraday else dt.date.today()
    last_data = dataset.get_data_last(as_of=as_of, bbid=bbid)

    # Pull the surface with the date/time of the most recent snap
    last_time = last_data.index[0] if intraday else last_data.index[0].date()
    df = ds.get_data(bbid=bbid, start=last_time, end=last_time, strikeReference=strike_reference)

    # Write latest vol surface to CSV
    with open(f'{bbid}_{last_time}.csv', 'w') as f:
        print(f'Writing latest vol surface for {bbid} to {bbid}_{last_time}.csv')
        df.to_csv(f)

    return df

# In[7]:


get_latest_vol_surface(ds, 'SPX', strike_reference='forward').head()
