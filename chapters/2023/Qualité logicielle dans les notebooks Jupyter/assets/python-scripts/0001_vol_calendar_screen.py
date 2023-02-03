#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.session import GsSession
# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics',))

# ## Vol Calendar Functions

# In[ ]:


from gs_quant.data import Dataset
from gs_quant.timeseries import last_value, forward_vol, percentiles
from gs_quant.markets.securities import AssetIdentifier, SecurityMaster
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format 
import warnings
from datetime import date
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt


def vol_calendar_screen(crosses, long_tenor, short_tenor, start_date, end_date, history='2y',boxplot=True, deltaStrike='DN', location='NYC'):
    impvol = Dataset('FXIMPLIEDVOL')
    data = {}
    boxplt = pd.DataFrame()
    for bbid in crosses:
        asset = SecurityMaster.get_asset(bbid, AssetIdentifier.BLOOMBERG_ID)
        long_implied = impvol.get_data(start_date , end_date, bbid=bbid, tenor=long_tenor, deltaStrike=deltaStrike, location=location)[['impliedVolatility']]
        short_implied = impvol.get_data(start_date , end_date, bbid=bbid, tenor=short_tenor, deltaStrike=deltaStrike, location=location)[['impliedVolatility']]
        spread = (long_implied - short_implied)*100
        spread = spread.dropna()
       
        boxplt[bbid] = spread['impliedVolatility']
        data[bbid] = {'Current': last_value(spread['impliedVolatility']),
                       f'{history} Min': min(spread['impliedVolatility']),
                       f'{history} Max': max(spread['impliedVolatility']),
                      '%-ile': last_value(percentiles(spread['impliedVolatility'])),
                      }
    df = pd.DataFrame(data)
    vol_cal_screen = df.transpose()
    vol_cal_screen = vol_cal_screen.sort_values(by=['Current'], ascending=False).style.background_gradient(subset=['Current']).format("{:.2f}")
    if boxplot:
        f, ax = plt.subplots(figsize=(7, 6))
        ax = sns.boxplot(data=boxplt, orient="h", palette="mako").set_title('Spread Ranges')
    return vol_cal_screen

# In[ ]:


#assumes tenors are ordered from shortest to longest
def steepest_flattest(crosses, tenors, top=10):
    impvol = Dataset('FXIMPLIEDVOL')
    vols = pd.DataFrame()
    screen = pd.DataFrame(columns=['Cross', 'Tenors', 'Vol Far', 'Vol Near', 'Abs Spread', '%-ile'])
    for bbid in crosses:
        asset = SecurityMaster.get_asset(bbid, AssetIdentifier.BLOOMBERG_ID)
        for tenor in tenors:
            imp_vol = impvol.get_data(start , end, bbid=bbid, tenor=tenor, deltaStrike='DN', location='NYC')['impliedVolatility']
            vols[bbid + f'_{tenor}'] = imp_vol
        for x in tenors[:-1]:
            for y in tenors[1:]:
                if(x != y):
                    vol_far = vols[bbid + f'_{y}']
                    vol_near = vols[bbid + f'_{x}']
                    spread = vol_far - vol_near
                    row = {'Cross' : bbid, 'Tenors': f'{y}' +f'- {x}', 'Vol Far': last_value(vol_far)*100, 'Vol Near': last_value(vol_near)*100, 'Abs Spread' : last_value(spread), 
                                   '%-ile': last_value(percentiles(spread))}
                    screen = screen.append(row, ignore_index=True) 
    display(screen.sort_values(by=['%-ile'], ascending=False).head(top).style.set_caption(f'{top} Steepest Calendars').background_gradient(subset=['%-ile']))
    display(screen.sort_values(by=['%-ile'], ascending=True).head(top).style.set_caption(f'{top} Flattest Calendars').background_gradient(subset=['%-ile']))
    return screen

# ## FX Volatility Calendar Spread
# 
# The FX Structuring team uses this analysis to screen for the steepest and flattest spreads between 1y implied volatility - 3m implied volatility.
# 
# We are pulling [GS FX Implied Volatility](https://marquee.gs.com/s/developer/datasets/FXIMPLIEDVOL) by default. If you are looking for additional history or coverage, please see our premium version [link](https://marquee.gs.com/s/developer/datasets/FXIMPLIEDVOL_PREMIUM).

# In[ ]:


from gs_quant.datetime import business_day_offset
from dateutil.relativedelta import relativedelta
# Inputs to modify

g10 = ['USDJPY', 'EURUSD', 'AUDUSD', 'GBPUSD', 'USDCAD', 'USDNOK', 'NZDUSD', 'USDSEK', 'USDCHF']

end = business_day_offset(date.today(), -1, roll='forward')
start = business_day_offset(end - relativedelta(years=2), -1, roll='forward')


long_tenor = '1y'
short_tenor = '3m'
history = '2y'



screen = vol_calendar_screen(crosses=g10, long_tenor='1y', short_tenor='3m', start_date=start, end_date=end, boxplot=True)
screen

# ### Steepest  & Flattest Calendars
# 
# The FX Structuring team uses this analysis to screen for the steepest calendars, both in absolute terms and in percentile terms across multiple tenors. The steepest and flattest calendars are sorted by the percentile of calendar vs. its own history.

# In[ ]:


tenors = ['3m', '6m', '1y', '2y']
result = steepest_flattest(crosses=g10, tenors=tenors, top=10)

# In[ ]:



