#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.session import GsSession
# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics',)) 

# ### Forward Volatility Screen Functions

# In[ ]:


from gs_quant.data import Dataset
from gs_quant.timeseries import last_value, forward_vol,  VolReference
from gs_quant.markets.securities import AssetIdentifier, SecurityMaster
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format 
import warnings
from datetime import date
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

def forward_volatility_screen(crosses, tenor, forward_tenor, start_date, end_date, plot=True):
    data = {}
    #replace with premium dataset for more history
    impvol = Dataset('FXIMPLIEDVOL')
    for bbid in crosses:
        asset = SecurityMaster.get_asset(bbid, AssetIdentifier.BLOOMBERG_ID)
        fwd_vol = forward_vol(asset, tenor=tenor, forward_start_date=forward_tenor, strike_reference=VolReference.DELTA_NEUTRAL, relative_strike=0)
        change = fwd_vol.diff(5)
        implied = impvol.get_data(start_date , end_date, bbid=bbid, tenor=tenor, deltaStrike='DN', location='NYC')[['impliedVolatility']]
        data[bbid] = {f'{tenor}{forward_tenor} Fwd Vol': last_value(fwd_vol),
                       '1w Change': last_value(change),
                       f'{tenor} Implied Vol': last_value(implied['impliedVolatility'])*100,
                      'Roll-down': last_value(fwd_vol) - last_value(implied['impliedVolatility'])*100,
                      'Strike to Lows': min(implied['impliedVolatility'])*100 - last_value(fwd_vol),
                      'Strike to Highs': max(implied['impliedVolatility'])*100 - last_value(fwd_vol),
                      'Diff' : (max(implied['impliedVolatility'])*100 - last_value(fwd_vol)) + (min(implied['impliedVolatility'])*100 - last_value(fwd_vol))
                      }
    df = pd.DataFrame(data)
    fva_screen = df.transpose()
    fva_screen = fva_screen.sort_values(by=['Roll-down'])
    
    if plot:
        for fx in fva_screen.index:
            plt.scatter(fva_screen.loc[fx]['Roll-down'], fva_screen.loc[fx]['Diff'])
            plt.legend(fva_screen.index,loc='best', bbox_to_anchor=(0.9, -0.13), ncol=3)
        plt.xlabel('Carry (vols)')
        plt.ylabel('Upside vs Downside')
        plt.title('Carry vs Upside/Downside')
        plt.show()
    return fva_screen.style.background_gradient(subset=['Roll-down']).format("{:.2f}")


# ### Forward Volatility
# The FX Structuring desk uses this analysis to screen for the most attractive 6m forward 6m vols in FVA format
# 
# We are pulling [GS FX Implied Volatility](https://marquee.gs.com/s/developer/datasets/FXIMPLIEDVOL) by default. If you are looking for additional history or coverage, please see our premium version [link](https://marquee.gs.com/s/developer/datasets/FXIMPLIEDVOL_PREMIUM).

# ### Carry vs Upside/Downside
# 
# Note: Upper left corner shows attractive long vol crosses with low rolldown and high
# upside vs. downside difference. Lower right corner shows attractive short vol crosses
# with high carry and low upside vs. downside difference.
# 

# In[ ]:


from gs_quant.datetime import business_day_offset
from dateutil.relativedelta import relativedelta

g10 = ['USDJPY', 'EURUSD', 'AUDUSD', 'GBPUSD', 'USDCAD', 'USDNOK', 'NZDUSD', 'USDSEK', 'USDCHF']
EM = ['USDBRL', 'USDZAR',  'USDMXN', 'USDCLP','USDPLN', 'USDCZK', 'USDHUF', 'USDSGD', 'USDINR','USDKRW','USDSGD', 'USDINR',
            'USDTWD', 'USDTRY', 'USDPHD']


forward_tenor = '6m'
tenor = '6m'

end = business_day_offset(date.today(), -1, roll='forward')
start = business_day_offset(end - relativedelta(years=2), -1, roll='forward')

screen = forward_volatility_screen(g10, tenor, forward_tenor, start, end, plot=True)
screen

# In[ ]:



