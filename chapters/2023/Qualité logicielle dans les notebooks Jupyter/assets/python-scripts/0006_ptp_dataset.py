#!/usr/bin/env python
# coding: utf-8

# Examples require an initialized GsSession and relevant entitlements. External clients need to substitute thier own client id and client secret below. Please refer to [Authentication](https://developer.gs.com/p/docs/institutional/platform/authentication/) for details.

# In[1]:


from gs_quant.session import GsSession, Environment
from datetime import date
from gs_quant.data import PTPDataset
import pydash
import pandas as pd

GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('read_product_data',))

# # PTP Datasets
# 
# PlotTool Pro provides a way to easily upload and visualize your data stored with Marquee Data Services, and you can use `gs_quant`'s `PTPDataset` class to upload, save, and view your `pandas` data structures. 
# 
# For now, PTP datasets are only available for internal Marquee users.
# 
# Data uploaded to a PTP dataset must be
# 1. Numeric
# 1. Indexed by a `pd.DatetimeIndex`
# 1. EOD (as opposed to real-time)
# 
# For further information, see the `PTPDataset` class. 

# ## Workflow

# #### Create a PTP Dataset
# 
# When creating a new PTP dataset, you first initialize an object, passing in a `pandas` Series or DataFrame, and an optional name. If you don't provide a name, the dataset's name will default to "GSQ Default." 
# 
# If you pass a DataFrame, the field names in your dataset will be derived from the DataFrame's column names; if you provide a Series, the field name will be taken from the series name (if it has one), or otherwise "values" by de
# 
# **Note**: your data is not *saved* until you call `.sync()` on your PTP dataset object. 

# In[7]:


df = pd.DataFrame({'col': range(50), 'fieldb': range(100, 150)}, index=pd.date_range(start=date(2021, 1, 1), periods=50, freq='D'))
dataset = PTPDataset(df, 'Test Dataset')

# #### Sync
# 
# Sync your dataset to save your data. 

# In[8]:


dataset.sync()

# #### Plot
# 
# Generate a transient plot expression, which will bring you to a plot displaying your data. 
# 
# Make sure to hit "Copy to My Plots" to save the plot (although you can always re-generate another transient plot expression). 
# 
# If you're running `gs_quant` on a device with a default browser, `.plot()` will automatically open the plot for you. 

# In[9]:


dataset.plot(open_in_browser=False)

# #### Other dataset functions
# 
# Since `PTPDataset` inherits from the `Dataset` class, it has all the functionality of a normal `Dataset` object in addition to the methods above. 
# 
# For example, to delete a dataset: 

# In[10]:


dataset.delete()
