#!/usr/bin/env python
# coding: utf-8

# ## Comparing, Reconciling, and Combining COVID-19 Data Sources

# ### Summary
# 
# In this note we use `gs-quant` to compare COVID-19 data sources. To do this, we first retrieve COVID-19-related time 
# series data and preprocess it, joining different sources together to analyze confirmed cases. 
# 
# The contents of this notebook are as follows:
# 
# - [1 - Getting Started](#1---Getting-Started)
# - [2 - COVID-19 Data](#2---COVID-19-Data)
# - [3 - Comparing Global Sources](#3---Comparing-Global-Sources)
# - [4 - Comparing US Sources](#4---Comparing-US-Sources)
# - [5 - Comparing subregions, combining with mobility data](#5---Comparing-subregions,-combining-with-mobility-data)

# ### 1 - Getting Started

# Start every session with authenticating with your unique client id and secret. For information on
# how to get setup on GS Quant, see [Getting Started](/covid/guides/getting-started). Below produced
# using gs-quant version 0.8.126

# In[ ]:


from gs_quant.session import GsSession, Environment

GsSession.use(client_id=None, client_secret=None, scopes=('read_product_data',))

# ### 2 - COVID-19 Data

# We'll start by defining a general function to load various datasets, which includes regional data, for the past week:

# In[ ]:


from gs_quant.data import Dataset
import datetime

# Note: There is data going back to 2019-12-31, you will need to write your own code to batch data fetching
def get_datasets(datasets):
    ds_dict = {}
    end = datetime.date(2020, 7, 9)
    start = end - datetime.timedelta(weeks=1)
    
    for dataset in datasets:
        try:
            df = Dataset(dataset).get_data(start, end)

            keys = [x for x in ['countryId', 'subdivisionId'] if x in df.columns] + ['date']
            val_map = {'newConfirmed': 'totalConfirmed', 'newFatalities': 'totalFatalities'}
            vals = [x for x in list(val_map.keys()) if x in df.columns]

            df_t = df.groupby(keys).sum().groupby(level=0).cumsum().reset_index()[keys + vals].rename(columns=val_map)
            ds_dict[dataset] = df.reset_index().merge(df_t, on=keys, suffixes=('', '_y')).set_index('date')

        except Exception as err:
            print(f'Failed to obtain {dataset} with {getattr(err, "message", repr(err))}')
    return ds_dict

# We create a list of some of the available datasets, and fetch all of them, so that we can compare them.

# In[ ]:


country_datasets = [
    'COVID19_COUNTRY_DAILY_ECDC', 
    'COVID19_COUNTRY_DAILY_WHO',
    'COVID19_COUNTRY_DAILY_WIKI', 
    'COVID19_US_DAILY_CDC'
]
df = get_datasets(country_datasets)

# Next we look at the date ranges of each dataset to determine how much history they have, and ensure they are 
# up-to-date:

# In[ ]:


for name, ds in df.items():
    print('{:<30}  {}  {}'.format(name, ds.index[0].date(), ds.index[-1].date())) 

# ### 3 - Comparing Global Sources
# 
# Below is a general function to compare the time series of certain columns across datasets:

# In[ ]:


import re
from typing import Union

def compare_time_series(df, datasets, columns: Union[str, list], grouping: str = 'countryId', suffix_identifier: float = 1):
    columns = [columns] if isinstance(columns, str) else columns
    suffixes = list(map(lambda ds_name: '_' + re.findall('\_([A-Z]+)', ds_name)[-suffix_identifier], datasets))
    df_combo = None

    for ds_name in datasets:
        ds = df[ds_name]
        df_combo = ds if df_combo is None else df_combo
        df_suffixes = ('', '_' + re.findall('\_([A-Z]+)', ds_name)[-suffix_identifier])
        df_combo = df_combo.merge(ds, on=['date', grouping], suffixes=df_suffixes)

    return df_combo[[grouping] + [column + suffix for suffix in suffixes for column in columns]]

# For example, if we want to compare the time series for total confirmed cases across the WHO, ECDC, and Wikipedia 
# datasets globally, we can do the following:

# In[ ]:


datasets = ['COVID19_COUNTRY_DAILY_ECDC', 'COVID19_COUNTRY_DAILY_WHO', 'COVID19_COUNTRY_DAILY_WIKI'] 
df_to_compare = compare_time_series(df, datasets, columns='totalConfirmed')

df_to_compare.describe().style.background_gradient(cmap='Blues', axis=1).format('{:,.2f}')

# This shows statistical properties for each dataset for all common countries and dates. As we can see, there's some 
# variation in the data sources. Let's dig in a little further and plot the relationship between the WHO and ECDC for 
# a number of countries: 

# In[ ]:


import seaborn as sns

select_countries = ['GB', 'DE', 'IT', 'ES', 'FR', 'RU']
to_plot = df_to_compare[df_to_compare.countryId.isin(select_countries)]

sns.lmplot(x="totalConfirmed_ECDC", y="totalConfirmed_WHO", col="countryId", data=to_plot, col_wrap=3, height=3, fit_reg=False);

# As we can see, there is some dispersion between sources for certain countries. For information on the various ISO
# country codes, see [this guide](https://developer.gs.com/docs/covid/guides/standards/iso-countries/).
# 
# ### 4 - Comparing US Sources
# 
# Now let's take a closer look at the US data, adding in the CDC dataset:

# In[ ]:


datasets = ['COVID19_US_DAILY_CDC', 'COVID19_COUNTRY_DAILY_ECDC', 'COVID19_COUNTRY_DAILY_WHO', 'COVID19_COUNTRY_DAILY_WIKI'] 
df_to_compare = compare_time_series(df, datasets, columns='totalConfirmed')

df_to_compare.describe().style.background_gradient(cmap='Blues',axis=1).format('{:,.2f}')

# As of 21 of May 2020, CDC had the most confirmed cases, followed by Wikipedia, and then ECDC and WHO. This is not 
# overly surprising given the information collection and validation flows. Now let's examine the last few points:

# Now let's compare all the series side by side:

# In[ ]:


df_to_compare.plot(figsize=(10, 6), title='US')

# In[ ]:


import matplotlib.pyplot as plt

(df_to_compare['totalConfirmed_WHO']-df_to_compare['totalConfirmed_ECDC']).plot(figsize=(10, 6), title='Differences vs WHO', 
                                                                                label='ECDC')
(df_to_compare['totalConfirmed_WHO']-df_to_compare['totalConfirmed_CDC']).plot(label='CDC')
(df_to_compare['totalConfirmed_WHO']-df_to_compare['totalConfirmed_WIKI']).plot(label='WIKI')

plt.legend()

# This chart illustrates how the ECDC and CDC map cases versus the WHO. At the start of the epidemic these sources were
# much closer, and diverged over time, with CDC leading in reporting for the US versus the ECDC and WHO. 

# ### 5 - Comparing subregions, combining with mobility data

# Finally, we illustrate how to compare datasets for specific countries (in this case, Italy) at different level of granularity (region, province, etc.) and how to ccombine epidemic data with mobility data from Google.
# 
# As before, we fetch data for Italy, at three levels of granularity.

# In[ ]:


datasets = ['COVID19_ITALY_DAILY_DPC', 'COVID19_REGION_DAILY_DPC', 'COVID19_PROVINCE_DAILY_DPC'] 
df = get_datasets(datasets)

# In[ ]:


df_to_compare = compare_time_series(df, datasets, columns='totalConfirmed', suffix_identifier=3)

df_to_compare.describe().style.background_gradient(cmap='Blues',axis=1).format('{:,.2f}')

# We write a function to compare the data across different geographic subdivisions.

# In[ ]:


from functools import reduce
import pandas as pd

def compare_totals_across_breakdowns(df, data1, data2, column_to_check):
    
    # pick the common indices between the data being compared
    common_idx_province = reduce(lambda x, y:  x & y, 
                            df[data1[0]].groupby(data1[1]).apply(lambda x: x.index).tolist())
    common_idx_region = reduce(lambda x, y:  x & y, 
                            df[data2[0]].groupby(data2[1]).apply(lambda x: x.index).tolist())
    idx = common_idx_province & common_idx_region
    
    # calculate the difference, and rename column
    diff = df[data1[0]].groupby(data1[1]).apply(lambda x : x.loc[idx][column_to_check]).T.apply(sum,axis=1) -\
           df[data2[0]].groupby(data2[1]).apply(lambda x : x.loc[idx][column_to_check]).T.apply(sum,axis=1)
    diff = pd.DataFrame(diff).rename(columns={0: f'{data1[0]}-{data2[0]}'})
    return diff

diff1 = compare_totals_across_breakdowns(df, ('COVID19_ITALY_DAILY_DPC','countryId'),
                                         ('COVID19_REGION_DAILY_DPC','subdivisionId'),'totalConfirmed')
diff2 = compare_totals_across_breakdowns(df, ('COVID19_REGION_DAILY_DPC','subdivisionId'),
                                        ('COVID19_PROVINCE_DAILY_DPC','administrativeRegion'),'totalConfirmed')

# We plot the discrepancies below...

# In[ ]:


to_plot = diff1.join(diff2)

sns.lmplot(x="COVID19_ITALY_DAILY_DPC-COVID19_REGION_DAILY_DPC", y="COVID19_REGION_DAILY_DPC-COVID19_PROVINCE_DAILY_DPC",
          data=to_plot)

# ... and interestingly, this indicates there is no discrepancy at all when we compare country-level aggregate data with region-level aggregate data, but we do see discrepancies when we compate province-level with region-level data.

# Finally, we illustrate how to join region-level data from Italy with mobility data from Google, which allows us to check, for example, how the increase in cases of COVID-19 affected mobility patterns in the population. 

# In[ ]:


from datetime import datetime

df_mob = Dataset('COVID19_SUBDIVISION_DAILY_GOOGLE').get_data(start_date=datetime(2020,2,1).date(), countryId='IT')

# In[ ]:


df_mob.head(2)

# We now join mobility data with region-level data in Italy, for the subdivision of Liguria.

# In[ ]:


def join_dfs(subdivision, mobility, column_to_compare):
    df_red = df['COVID19_REGION_DAILY_DPC'][df['COVID19_REGION_DAILY_DPC']['subdivisionId'] == subdivision]
    subdivision_name = df_red['subdivisionName'].unique()[0]
    df1 = df_red[[column_to_compare]]
    df2 = df_mob[df_mob.subdivisionId.isin([subdivision]) & df_mob.group.isin([mobility])]
    df_joint = df1.merge(df2, on='date')
    return df_joint, mobility, subdivision_name

df_joint, mobility, subdivision_name = join_dfs('IT-42', 'retailRecreation', 'newConfirmed')

# Finally, we plot a chart comparing mobility data with data on the growth of the epidemic.

# In[ ]:


df_joint['Change in Mobility'] = df_joint['value'].diff()
df_joint[['newConfirmed','value']].rename(columns={'newConfirmed':'New Confirmed Cases',
                                                   'value':f'Change in {mobility} mobility'}).plot(figsize=(10, 7,), grid=True,
            title=f'Comparison of new confirmed cases and mobility in {subdivision_name}')

# We can observe a dramatic drop in moblity as the rate of new cases began to increase, a pattern that persisted during the peak of the epidemic in the region of Liguria.
# 
# Please reach out to `covid-data@gs.com` with any questions.

# ### Disclaimer
# This website may contain links to websites and the content of third parties ("Third Party Content"). We do not monitor, 
# review or update, and do not have any control over, any Third Party Content or third party websites. We make no 
# representation, warranty or guarantee as to the accuracy, completeness, timeliness or reliability of any 
# Third Party Content and are not responsible for any loss or damage of any sort resulting from the use of, or for any 
# failure of, products or services provided at or from a third party resource. If you use these links and the 
# Third Party Content, you acknowledge that you are doing so entirely at your own risk.
