#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gs_quant.session import GsSession
# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics',)) 

# #### Costless 1x2 Put Spreads 
# 
# This screen looks for max payouts and breakeven levels for 1x2 zero cost put spreads for a given tenor. Top strike is fixed at the specified delta and the bottom strike is solved to make the structure zero-cost.

# In[2]:


from gs_quant.markets.portfolio import Portfolio
from gs_quant.markets import PricingContext
from gs_quant.instrument import FXOption
from gs_quant.risk import FXSpot
from gs_quant.common import BuySell
import pandas as pd; pd.set_option('display.precision', 2)

def get_screen(df_portfolio, strike_label='40d'):
    screen = pd.pivot_table(df_portfolio, values='strike_price', index=['pair', 'Spot'], columns=['buy_sell'])
    screen = screen.reset_index('Spot')
    upper_label, lower_label = f'Upper Strike ({strike_label})', 'Lower Strike (Mid)*'
    screen = screen.rename(columns={BuySell.Buy: upper_label, BuySell.Sell: lower_label})
    screen['Max Payout'] = screen[upper_label] / screen[lower_label] - 1
    screen['Lower Breakeven (Mid)'] = 2 * screen[lower_label] - screen[upper_label]
    screen['Lower Breakeven OTMS'] = abs(screen['Lower Breakeven (Mid)'] / screen['Spot'] - 1)
    return screen

def calculate_bear_put_spreads(upper_strike='40d', crosses=['USDNOK'], tenor='3m'):
    portfolio = Portfolio()
    for cross in crosses:
        option_type = 'Put' if cross[:3] == 'USD' else 'Call'
        upper_leg = FXOption(pair=f'{cross}', strike_price=upper_strike, notional_amount=10e6, 
                             option_type=option_type, buy_sell=BuySell.Buy, expiration_date=tenor, premium_currency=cross[:3])
        lower_leg = FXOption(pair=f'{cross}', strike_price=f'P={abs(upper_leg.premium)}', notional_amount=20e6,
                             option_type=option_type, buy_sell=BuySell.Sell, expiration_date=tenor)
        portfolio.append((upper_leg, lower_leg))
    with PricingContext():
        portfolio.resolve()
        spot = portfolio.calc(FXSpot)
    summary = portfolio.to_frame()
    summary['Spot'] = list(spot.result())
    return get_screen(summary, upper_strike)

# In[3]:


g10 = ['USDJPY', 'EURUSD', 'AUDUSD', 'GBPUSD', 'USDCAD', 'USDNOK', 'NZDUSD', 'USDSEK', 'USDCHF']

result = calculate_bear_put_spreads(upper_strike='40d', crosses=g10, tenor='3m')
result.sort_values(by='Max Payout', ascending=False).style.format({
    'Max Payout': '{:,.2%}'.format, 'Lower Breakeven OTMS': '{:,.2%}'.format}).background_gradient(
    subset=['Max Payout', 'Lower Breakeven OTMS'])
