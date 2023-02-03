#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gs_quant.session import GsSession
# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics')) 

# ### Volatility Swap Spread Screen

# #### Functions

# In[2]:


from gs_quant.datetime import business_day_offset
from gs_quant.datetime.relative_date import RelativeDate
from gs_quant.timeseries import implied_volatility, VolReference, last_value
from datetime import date
from gs_quant.data import Dataset, DataContext 
def get_data(bbids=['USDJPY', 'AUDUSD'], long_tenor='6m', end=business_day_offset(date.today(), -1, roll='forward'), start=RelativeDate('-5y').apply_rule()):
    """Pulls historical implied volatility and realized volatility data 
     
    : param bbids: array of string FX pairs e.g. ['USDJPY', 'USDNOK']
    : param long_tenor: implied vol tenor
    : param end: end date for historical data
    : param start: start date for historical data
    
    """
    #implied vol
    vol_data = Dataset('FXIMPLIEDVOL').get_data(start, end, bbid=bbids, tenor=long_tenor, deltaStrike='DN', location='NYC')
    vol_df = pd.pivot_table(vol_data, values='impliedVolatility', index=['date'], columns=['bbid'])
    vol_df = vol_df*100
    
    shift_start = RelativeDate(f'-{long_tenor[0]}v', base_date=start).apply_rule()
    #internal users replace with 'WMFXSPOT'
    spot_data = Dataset('FXSPOT').get_data(shift_start, end, bbid=bbids)
    #replace with 'midPrice' if using WMFXSPOT
    spot_df = pd.pivot_table(spot_data, values='spot', index=['date'], columns=['bbid'])
    return vol_df, spot_df

# In[6]:


from gs_quant.timeseries import returns, percentile, filter_, FilterOperator, sum_, sqrt, union
def robust_volatility(series, tenor='6m', cutoff=5, pct=[90, 10]):
    ret = returns(1/series, 1)
    std_rob = last_value((percentile(ret, pct[0], tenor) - percentile(ret, pct[1], tenor))/2)
    zap = filter_(abs(ret), FilterOperator.GREATER, 5*std_rob)
    combined = union([zap, (0*ret)/ret])
    zap_count = sum_(zap, tenor)
    days = int(tenor[0])*21
    var = sqrt((252/(days - zap_count))*sum_(combined**2, tenor))*100
    return var

# In[7]:


from gs_quant.timeseries import volatility, returns, Returns, exponential_std, annualize, std
def calculate_realized_vol(spot_df, tenor='6m'):
    """Calculates realized vol using historical spot data
    
    : param tenor: realized vol tenor
    """
    weighting = (22-1)/(22+1)
    short_vol, long_vol, robust_vol = {}, {}, {}
    
    for ccy, row in spot_df.iteritems():
        long_vol[ccy] = volatility(row, tenor, returns_type=Returns.LOGARITHMIC)
        short_vol[ccy] = annualize(returns(row,1, Returns.LOGARITHMIC).ewm(alpha=1-weighting, adjust=True).std())*100
        robust_vol[ccy] = robust_volatility(row, tenor)
        
    return pd.DataFrame.from_dict(long_vol), pd.DataFrame.from_dict(short_vol), pd.DataFrame.from_dict(robust_vol)

# In[13]:


from gs_quant.timeseries import LinearRegression
import pandas as pd
pd.set_option('display.precision', 2)
import warnings
import itertools
def calculate_vol_swap_screen(bbids=['USDJPY', 'AUDUSD'], long_tenor='6m', 
                              end=business_day_offset(date.today(), -1, roll='forward'), 
                              start=RelativeDate('-5y').apply_rule()):
    """Calculates volatility swap spread screen
    
    : param bbids: array of string FX pairs e.g. ['USDJPY', 'USDNOK']
    : param long_tenor: implied vol and realized vol tenor
    : param short_tenor: shorter realized vol tenor
    : param end: end date for historical data
    : param start: start date for historical data
    """
    
    vol_df, spot_df = get_data(bbids, long_tenor, end, start)
    long_rvol, short_rvol, robust_vol = calculate_realized_vol(spot_df, long_tenor)
    results = pd.DataFrame(columns=['crosses', '6m Implied spread', 'Beta', f'Entry vs {long_tenor} Z-score', 
                                    'Avg Carry Z-score', 'Score', f'{long_tenor} Realized Vol', f'{long_tenor} Carry', f'{long_tenor} 5y Avg', f'{long_tenor} 10th', 
                                    f'{long_tenor} 90th', '1m Realized Vol', '1m Carry'])
    pairs = itertools.combinations(crosses, 2)
    for pair in pairs:
        short, long = pair[0], pair[1]
        beta = LinearRegression(vol_df[short], vol_df[long], fit_intercept=False).coefficient(1)
        iv_spread =  vol_df.iloc[-1][long] - beta*vol_df.iloc[-1][short]
        rv_long_spread =  long_rvol[long] - beta*long_rvol[short] 
        rv_short_spread = short_rvol[long] - beta*short_rvol[short]
        robust_spread = robust_vol[long] - beta*robust_vol[short]
        z_score = (robust_spread.mean() - iv_spread)/robust_spread.std()
        carry_long = rv_long_spread[-1] - iv_spread
        carry_short = rv_short_spread[-1] - iv_spread
        carry_avg = (carry_long + carry_short)/2
        carry_zscore = carry_avg / robust_spread.std()
        results = results.append({'crosses': f'{long} vs. {short}', '6m Implied spread': iv_spread, 'Beta': beta, f'Entry vs {long_tenor} Z-score': z_score,
                                 'Avg Carry Z-score': carry_zscore, 'Score': z_score + carry_zscore, f'{long_tenor} Realized Vol': rv_long_spread[-1], 
                                  f'{long_tenor} Carry': carry_long,f'{long_tenor} 5y Avg': robust_spread.mean(), 
                                  f'{long_tenor} 10th': rv_long_spread.quantile(0.1), f'{long_tenor} 90th': rv_long_spread.quantile(0.9), 
                                  '1m Realized Vol': rv_short_spread[-1], '1m Carry': carry_short}, ignore_index=True)
    return results.set_index('crosses').sort_values('6m Implied spread')

# ### Vol Swap Spread Screen
# 
# This screen compares the historical realized volatility spread between two crosses with the implied volatility spread. The spread is beta adjusted on the short leg. The score is the combination of the entry vs 6m z-score and the average carry z-score (long carry + 1m carry). The volatility spreads with highest score are more attractive to buy.

# In[14]:


crosses = ['EURUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'AUDUSD', 'GBPUSD', 'USDJPY' , 'USDNOK', 'USDMXN', 'USDSEK']
screen = calculate_vol_swap_screen(crosses)
screen.sort_values('Score', ascending=False).style.background_gradient(subset=['Score'])
