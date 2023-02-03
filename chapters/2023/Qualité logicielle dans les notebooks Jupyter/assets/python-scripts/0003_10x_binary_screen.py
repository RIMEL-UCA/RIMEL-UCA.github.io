#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.session import GsSession
# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics',)) 

# ### 10x Binaries
# 
# This screen shows how far a OTMS a 3m binary option must be struck to have a payout ratio of approximately 10:1. 
# Normalized = % OTMS / 3m realized vol

# In[ ]:


from gs_quant.timeseries import last_value, realized_volatility
from gs_quant.markets.securities import AssetIdentifier, SecurityMaster
from gs_quant.markets import PricingContext
from gs_quant.instrument import FXBinary
from gs_quant.target.common import OptionType
from gs_quant.markets.portfolio import Portfolio
from gs_quant.risk import FXSpot, Price
from gs_quant.data import DataContext
from gs_quant.api.gs.data import * 
import pandas as pd
import warnings
from datetime import date
import matplotlib.pyplot as plt
from gs_quant.datetime import business_day_offset
from dateutil.relativedelta import relativedelta
warnings.filterwarnings('ignore')
pd.set_option('display.precision', 3)

# ### Functions
# 
# 1) Calculating payout for binaries using defaults of 3m and 10:1
# 
# 
# 2) Plot historical spot with call and put strikes

# In[ ]:


def payout_struck_binaries(crosses, payout_ratio='10%', tenor='3m', normalize=True, start_date=business_day_offset(business_day_offset(date.today(), -1, roll='forward') - relativedelta(years=1), -1, roll='forward')):
    """Prices FX Binaries for inputted crosses strike price using inputted payout ratio
    
    Screens for % OTMS for a given payout ratio
    
    : param crosses: array of string FX pairs e.g. ['USDJPY', 'USDNOK']
    : param payout_ratio: string with % used to solve for strike price e.g 10% solves for a 10:1 payout ratio
    : param tenor: FX Binary expiration date
    : param normalize: divides % OTMS by realized volatility 
    : param start_date: start date for volatility history, if normalizing % OTMS, defaults to 1yr
    
    """
    binaries = Portfolio([FXBinary(pair=cross, expiration_date=tenor, option_type=direction, 
                                          strike_price=f'p={payout_ratio}', premium=0) 
                                 for direction in ('Call', 'Put') for cross in crosses])
    with PricingContext():   
        binaries.resolve()
        spot = binaries.calc(FXSpot)            
    bf = binaries.to_frame()
    bf.index = bf.index.droplevel(0)
    bf = bf[['pair', 'option_type', 'strike_price']]
    bf['spot'] = list(spot)
    bf['% otms'] = abs(bf.strike_price/bf.spot-1)*100
    
    #normalizing otms
    if normalize:
        df_vol = pd.DataFrame(columns=['pair','implied_vol'])
        for cross in crosses:
            asset = SecurityMaster.get_asset(cross, AssetIdentifier.BLOOMBERG_ID)
            with DataContext(start=start_date, end=business_day_offset(date.today(), -1, roll='forward')): 
                vol = last_value(realized_volatility(asset, w=tenor))
            df_vol = df_vol.append({'pair': f'{cross[:3]} {cross[3:]}', 'implied_vol': vol}, ignore_index=True)
        bf = bf.merge(df_vol, left_on='pair', right_on='pair')
        bf['normalized'] = bf['% otms'] / bf['implied_vol']
    return bf.set_index(['option_type', 'pair']).sort_values(by=['option_type', '% otms'])

# In[ ]:


def plot_strikes(crosses, binaries, start_date=business_day_offset(business_day_offset(date.today(), -1, roll='forward') - relativedelta(years=2), -1, roll='forward')):
    """Plots historical spot for each FX pair and strike prices from FX Binary 
    
    : param crosses: array of string FX pairs e.g. ['USDJPY', 'USDNOK']
    : param binaries: Dataframe with crosses as index with strikes for both call and put FX Binary
    : param start: start date for spot history, defaults to 2y
    
    """
    for cross in binaries.index.get_level_values('pair'):
        with DataContext(start=start_date, end=business_day_offset(date.today(), -1, roll='forward')): 
            asset = SecurityMaster.get_asset(cross.replace(" ", ""), AssetIdentifier.BLOOMBERG_ID)
            q = GsDataApi.build_market_data_query(
                [asset.get_marquee_id()],
                QueryType.SPOT,
                source=None,
                real_time=False
            )
        spot = GsDataApi.get_market_data(q)
        spot.plot()
        plt.axhline(y=binaries.loc[(OptionType.Call, f'{cross}')]['strike_price'], color='g', label='Call Strike')
        plt.axhline(y=binaries.loc[(OptionType.Put, f'{cross}')]['strike_price'], color='r', label='Put Strike')
        plt.title(f'{cross} Binary')
        plt.legend()
        plt.show()

# ### 10X Binary Screen

# In[ ]:


g10 = ['USDJPY', 'EURUSD', 'AUDUSD', 'GBPUSD', 'USDCAD', 'USDNOK', 'NZDUSD', 'USDSEK', 'USDCHF']

results = payout_struck_binaries(crosses=g10)
results.style.background_gradient(subset=['% otms'])

# In[ ]:


plot_strikes(g10, results, start_date=start_date)
