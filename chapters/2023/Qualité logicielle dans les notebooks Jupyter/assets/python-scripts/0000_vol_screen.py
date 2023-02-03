#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.data import Dataset
from gs_quant.timeseries import percentiles, volatility, last_value, Returns
from gs_quant.datetime import business_day_offset
import seaborn as sns
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format 
import matplotlib.pyplot as plt
from scipy import stats
import warnings
from datetime import date
warnings.filterwarnings('ignore')
sns.set(style="darkgrid", color_codes=True)

# In[ ]:


from gs_quant.session import GsSession
# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics',)) 

# ## Screen Functions

# In[ ]:


def format_df(data_dict):
    df = pd.concat(data_dict, axis=1)
    df.columns = data_dict.keys()
    return df.fillna(method='ffill').dropna()

# In[ ]:


def volatility_screen(crosses, start_date, end_date, tenor='3m', history='2y', plot=True):
    #replace with premium dataset for more history
    fxspot_dataset, fxvol_dataset = Dataset('FXSPOT'), Dataset('FXIMPLIEDVOL')
    spot_data, impvol_data, spot_fx, data = {}, {}, {}, {}
    for cross in crosses:
        spot_fx[cross] = fxspot_dataset.get_data(start_date, end_date, bbid=cross)[['spot']].drop_duplicates(keep='last')['spot']
        spot_data[cross] = volatility(spot_fx[cross], tenor)  # realized vol 
        impvol_data[cross] = fxvol_dataset.get_data(start_date, end_date, bbid=cross, tenor=tenor, deltaStrike='DN', location='NYC')[['impliedVolatility']]* 100

    spdata, ivdata = format_df(spot_data), format_df(impvol_data)
    diff = ivdata.subtract(spdata).dropna()
    for cross in crosses:
        data[cross] = {'Spot': last_value(spot_fx[cross]),
                       f'{tenor} Implied': last_value(ivdata[cross]),
                       f'{tenor} Realized': last_value(spdata[cross]),
                       'Diff': last_value(diff[cross]),
                       f'{history} Implied Low': min(ivdata[cross]),
                       f'{history} Implied High': max(ivdata[cross]),
                       '%-ile': last_value(percentiles(ivdata[cross]))
                      }
    df = pd.DataFrame(data)
    vol_screen = df.transpose()
    
    if plot:
        for fx in vol_screen.index:
            plt.scatter(vol_screen.loc[fx]['%-ile'], vol_screen.loc[fx]['Diff'])
            plt.legend(vol_screen.index,loc='best', bbox_to_anchor=(0.9, -0.13), ncol=3)
    
        plt.xlabel('Percentile of Current Implied Vol')
        plt.ylabel('Implied vs Realized Vol')
        plt.title('Entry Point vs Richness')
        plt.show()
    return vol_screen.sort_values(by=['Diff']).style.background_gradient(subset=['Diff']).format("{:.1f}")

# ### FX Implied Volatility Screen
# Let's pull [GS FX Spot](https://marquee.gs.com/s/developer/datasets/FXSPOT) and [GS FX Implied Volatility](https://marquee.gs.com/s/developer/datasets/FXIMPLIEDVOL) and look at implied vs realized vol as well as current implied level as percentile relative to the last 2 years. Note, FX Spot uses GS NYC closes.
# 
# 
# If you are looking for additional history or coverage, please see our premium version [link](https://marquee.gs.com/s/developer/datasets/FXIMPLIEDVOL_PREMIUM).
# 
# The FX Structuring team uses this analysis to screen for the most attractive vols to buy in the 3m tenor by looking at where implied trades in its own history and where realized trades in relationship with implieds.
# 

# ### Entry Point vs Richness
# 
# Note: Lower left corner shows currencies with low and cheap vol. Upper right corner
# shows currencies with high and rich vol.

# In[ ]:


from gs_quant.datetime import business_day_offset
from dateutil.relativedelta import relativedelta
g10 = ['USDJPY', 'EURUSD', 'AUDUSD', 'GBPUSD', 'USDCAD', 'USDNOK', 'NZDUSD', 'USDSEK', 'USDCHF']

end = business_day_offset(date.today(), -1, roll='forward')
start = business_day_offset(end - relativedelta(years=2), -1, roll='forward')

tenor = '3m'
history = '2y'

screen = volatility_screen(g10, start, end, tenor, history, plot=True)
screen
