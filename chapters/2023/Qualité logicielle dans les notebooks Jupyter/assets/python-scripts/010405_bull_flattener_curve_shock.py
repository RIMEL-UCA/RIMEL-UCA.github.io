#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.session import Environment, GsSession
from gs_quant.common import PayReceive, Currency
from gs_quant.instrument import IRSwaption
from gs_quant.risk import CurveScenario, MarketDataPattern
import matplotlib.pyplot as plt
import pandas as pd

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


swaption = IRSwaption(PayReceive.Receive, '5y', Currency.USD, expiration_date='13m', strike='atm')
swaption.resolve()

# In[ ]:


original_price = swaption.price()
# retrieve the market data our instrument is sensitive to.
market_data = swaption.market().market_data_dict
print('Base price:     {:,.2f}'.format(original_price))

# In[ ]:


# Pivot point is the tenor at which curve shift =0 and influences the type and shape of curve shift
# Price the swaption under a bull flattener scenario of -10bp
bull_flattener_scenario = CurveScenario(market_data_pattern=MarketDataPattern('IR', 'USD'),
                                        curve_shift=-10, tenor_start=15, tenor_end=25, pivot_point=15)

with bull_flattener_scenario:
    swaption_bull_flattener = swaption.price()
    market_data_bull_flattener = swaption.market().market_data_dict

print('Price under bull flattener curve shift: {:,.2f}'.format(swaption_bull_flattener))

# In[ ]:


# Compare swap rate market data coordinates before and after curve scenario shock
market_data_df = pd.DataFrame([{mkt_data: value * 1e4 for mkt_data, value in market_data.items() if (mkt_data.mkt_type=="IR" and mkt_data.mkt_asset=="USD" and mkt_data.mkt_class=="SWAP OIS")},
                               {mkt_data: value * 1e4 for mkt_data, value in market_data_bull_flattener.items() if (mkt_data.mkt_type=="IR" and mkt_data.mkt_asset=="USD" and mkt_data.mkt_class=="SWAP OIS")}],
                              index=['Values', 'Shocked values']).transpose()
market_data_df

# In[ ]:


# Plotting swap rate market data before and after curve scenario shock 
swap_curve = pd.DataFrame.from_dict({int(''.join(list(filter(str.isdigit, str(v))))): market_data_df.loc[v]
                                    for v in market_data_df.index}, orient='index')

swap_curve['Shock'] = swap_curve['Shocked values'] - swap_curve['Values']
swap_curve.plot(figsize=(12, 8), title='USD Swap Curve Before and After {}bp bull flattening Shock'.format(bull_flattener_scenario.curve_shift))
plt.xlabel('Tenor (years)')
plt.ylabel('bp')
