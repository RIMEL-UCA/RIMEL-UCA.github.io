#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from gs_quant.datetime import business_day_offset
from gs_quant.markets import PricingContext, BackToTheFuturePricingContext
from gs_quant.risk import RollFwd, MarketDataPattern, MarketDataShock, MarketDataShockBasedScenario, MarketDataShockType
from gs_quant.instrument import FXOption, IRSwaption
from gs_quant.timeseries import *
from gs_quant.timeseries import percentiles
warnings.filterwarnings('ignore')
sns.set(style="darkgrid", color_codes=True)

# In[ ]:


from gs_quant.session import GsSession
# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics',)) 

# In this notebook, we'll look at entry points for G10 vol, look for crosses with the largest downside sensivity to SPX, indicatively price several structures and analyze their carry profile.
# 
# * [1: FX entry point vs richness](#1:-FX-entry-point-vs-richness)
# * [2: Downside sensitivity to SPX](#2:-Downside-sensitivity-to-SPX)
# * [3: AUDJPY conditional relationship with SPX](#3:-AUDJPY-conditional-relationship-with-SPX)
# * [4: Price structures](#4:-Price-structures)
# * [5: Analyse rates package](#5:-Analyse-rates-package)

# ### 1: FX entry point vs richness
# Let's pull [GS FX Spot](https://marquee.gs.com/s/developer/datasets/FXSPOT_PREMIUM) and [GS FX Implied Volatility](https://marquee.gs.com/s/developer/datasets/FXIMPLIEDVOL_PREMIUM) and look at implied vs realized vol as well as current implied level as percentile relative to the last 2 years.

# In[ ]:


def format_df(data_dict):
    df = pd.concat(data_dict, axis=1)
    df.columns = data_dict.keys()
    return df.fillna(method='ffill').dropna()

# In[ ]:


g10 = ['USDJPY', 'EURUSD', 'AUDUSD', 'GBPUSD', 'USDCAD', 'USDNOK', 'NZDUSD', 'USDSEK', 'USDCHF', 'AUDJPY']
start_date =  date(2005, 8, 26)
end_date = business_day_offset(date.today(), -1, roll='preceding')
fxspot_dataset, fxvol_dataset = Dataset('FXSPOT_PREMIUM'), Dataset('FXIMPLIEDVOL_PREMIUM')

spot_data, impvol_data, spot_fx = {}, {}, {}
for cross in g10:
    spot = fxspot_dataset.get_data(start_date, end_date, bbid=cross)[['spot']].drop_duplicates(keep='last')
    spot_fx[cross] = spot['spot']
    spot_data[cross] = volatility(spot['spot'], 63)  # realized vol 
    vol = fxvol_dataset.get_data(start_date, end_date, bbid=cross, tenor='3m', deltaStrike='DN', location='NYC')[['impliedVolatility']]
    impvol_data[cross] = vol.drop_duplicates(keep='last') * 100

spdata, ivdata = format_df(spot_data), format_df(impvol_data)
diff = ivdata.subtract(spdata).dropna()

# In[ ]:


_slice = ivdata['2018-09-01': '2020-09-08']
pct_rank = {}
for x in _slice.columns:
    pct = percentiles(_slice[x])
    pct_rank[x] = pct.iloc[-1]

for fx in pct_rank:
    plt.scatter(pct_rank[fx], diff[fx]['2020-09-08'])
    plt.legend(pct_rank.keys(),loc='best', bbox_to_anchor=(0.9, -0.13), ncol=3)
    
plt.xlabel('Percentile of Current Implied Vol')
plt.ylabel('Implied vs Realized Vol')
plt.title('Entry Point vs Richness')
plt.show()

# ### 2: Downside sensitivity to SPX
# 
# Let's now look at beta and correlation with SPX across G10.

# In[ ]:


spx_spot = Dataset('TREOD').get_data(start_date, end_date, bbid='SPX')[['closePrice']]
spx_spot = spx_spot.fillna(method='ffill').dropna()
df = pd.DataFrame(spx_spot)

#FX Spot data
fx_spots = format_df(spot_fx)
data = pd.concat([spx_spot, fx_spots], axis=1).dropna()
data.columns = ['SPX'] + g10

# In[ ]:


beta_spx, corr_spx = {}, {}

#calculate rolling 84d or 4m beta to S&P
for cross in g10:
    beta_spx[cross] = beta(data[cross],data['SPX'], 84)
    corr_spx[cross] = correlation(data['SPX'], data[cross], 84)

fig, axs = plt.subplots(5, 2, figsize=(18, 20))
for j in range(2):
    for i in range(5):
        color='tab:blue'
        axs[i,j].plot(beta_spx[g10[i + j*5]], color=color)
        axs[i,j].set_title(g10[i + j*5])
        color='tab:blue'
        axs[i,j].set_ylabel('Beta', color=color)
        axs[i,j].plot(beta_spx[g10[i + j*5]], color=color)
        ax2 = axs[i,j].twinx()
        color = 'tab:orange'        
        ax2.plot(corr_spx[g10[i + j*5]], color=color)
        ax2.set_ylabel('Correlation', color=color)
plt.show()

# ### Part 3: AUDJPY conditional relationship with SPX
# 
# Let's focus on AUDJPY and look at its relationship with SPX when SPX is significantly up and down.

# In[ ]:


# resample data to weekly from daily & get weekly returns
wk_data = data.resample('W-FRI').last()
rets = returns(wk_data, 1)
sns.set(style='white', color_codes=True)
spx_returns = [-.1, -.05, .05, .1]

betas = pd.DataFrame(index=spx_returns, columns=g10)
for ret in spx_returns:
    dns = rets[rets.SPX <= ret].dropna() if ret < 0 else rets[rets.SPX >= ret].dropna() 
    j = sns.jointplot(x='SPX', y='AUDJPY', data=dns, kind='reg')
    j.set_axis_labels('SPX with {}% Returns'.format(ret*100), 'AUDJPY')
    j.fig.subplots_adjust(wspace=.02)
    plt.show()

# Let's use the beta for all S&P returns to price a structure

# In[ ]:


sns.jointplot(x='SPX', y='AUDJPY', data=rets, kind='reg')

# ### 4: Price structures 
# 
# ##### Let's now look at a few AUDJPY structures as potential hedges
# 
# *  Buy 4m AUDJPY put using spx beta to size. Max loss limited to premium paid.
# *  Buy 4m AUDJPY put spread (4.2%/10.6% OTMS). Max loss limited to premium paid.
# 
# For more info on this trade, check out our market strats piece [here](https://marquee.gs.com/content/#/article/2020/08/28/gs-marketstrats-audjpy-as-us-election-hedge)

# In[ ]:


#buy 4m AUDJPY put
audjpy_put = FXOption(option_type='Put', pair='AUDJPY', strike_price= 's-4.2%', expiration_date='4m', buy_sell='Buy')    
print('cost in bps: {:,.2f}'.format(audjpy_put.premium / audjpy_put.notional_amount * 1e4))

# In[ ]:


#buy 4m AUDJPY put spread (5.3%/10.6% OTMS)
from gs_quant.markets.portfolio import Portfolio
put1 = FXOption(option_type='Put', pair='AUDJPY', strike_price= 's-4.2%', expiration_date='4m', buy_sell='Buy')
put2 = FXOption(option_type='Put', pair='AUDJPY', strike_price= 's-10.6%', expiration_date='4m', buy_sell='Sell')

fx_package = Portfolio((put1, put2))
cost = put2.premium/put2.notional_amount - put1.premium/put1.notional_amount 
print('cost in bps: {:,.2f}'.format(cost * 1e4))

# ##### ...And some rates ideas
# 
# * Sell straddle. Max loss unlimited.
# * Sell 3m30y straddle, buy 2y30y straddle in a 0 pv package. Max loss unlimited.

# In[ ]:


leg = IRSwaption('Straddle', '30y', notional_currency='USD', expiration_date='3m', buy_sell='Sell')
print('PV in USD: {:,.2f}'.format(leg.dollar_price()))

# In[ ]:


rates_package = Portfolio((IRSwaption('Straddle', '30y', notional_currency='USD', expiration_date='3m', buy_sell='Sell',
                                      name='3m30y ATM Straddle'),
                           IRSwaption('Straddle', '30y', notional_currency='USD', expiration_date='2y', buy_sell='Buy', 
                                      notional_amount='=solvefor([3m30y ATM Straddle].risk.Price,pv)', 
                                      name = '2y30y ATM Straddle')))
rates_package.resolve()

print('Package cost in USD: {:,.2f}'.format(rates_package.price().aggregate()))
print('PV Flat notionals ($$m):', round(leg1.notional_amount/1e6, 1),' by ',round(leg2.notional_amount/1e6, 1))

# ### 5: Analyse rates package

# In[ ]:


dates = pd.bdate_range(date(2020, 6, 8), leg1.expiration_date, freq='5B').date.tolist()

with BackToTheFuturePricingContext(dates=dates, roll_to_fwds=True):
    future = rates_package.price()
rates_future = future.result().aggregate()

rates_future.plot(figsize=(10, 6), title='Historical PV and carry for rates package')

print('PV breakdown between legs:')
results = future.result().to_frame()
results /= 1e6
results.loc['Total'] = results.sum()
results.round(1)

# Let's focus on the next 3m and how the calendar carries in different rates shocks.

# In[ ]:


dates = pd.bdate_range(dt.date.today(), leg1.expiration_date, freq='5B').date.tolist()
shocked_pv = pd.DataFrame(columns=['Base', '5bp per week', '50bp instantaneous'], index=dates)

p1, p2, p3 = [], [], []
with PricingContext(is_batch=True):
    for t, d in enumerate(dates):
        with RollFwd(date=d, realise_fwd=True):
            p1.append(rates_package.price())
            with MarketDataShockBasedScenario({MarketDataPattern('IR', 'USD'): MarketDataShock(MarketDataShockType.Absolute, t*0.0005)}):
                p2.append(rates_package.price())
            with MarketDataShockBasedScenario({MarketDataPattern('IR', 'USD'): MarketDataShock(MarketDataShockType.Absolute, 0.005)}):
                p3.append(rates_package.price())

shocked_pv.Base = [p.result().aggregate() for p in p1]
shocked_pv['5bp per week'] = [p.result().aggregate() for p in p2]
shocked_pv['50bp instantaneous'] = [p.result().aggregate() for p in p3]

shocked_pv/=1e6
shocked_pv.round(1)
shocked_pv.plot(figsize=(10, 6), title='Carry + scenario analysis')

# ### Disclaimers
# 
# Scenarios/predictions: Simulated results are for illustrative purposes only. GS provides no assurance or guarantee that the strategy will operate or would have operated in the past in a manner consistent with the above analysis. Past performance figures are not a reliable indicator of future results.
# 
# Indicative Terms/Pricing Levels: This material may contain indicative terms only, including but not limited to pricing levels. There is no representation that any transaction can or could have been effected at such terms or prices. Proposed terms and conditions are for discussion purposes only. Finalized terms and conditions are subject to further discussion and negotiation.
# www.goldmansachs.com/disclaimer/sales-and-trading-invest-rec-disclosures.html If you are not accessing this material via Marquee ContentStream, a list of the author's investment recommendations disseminated during the preceding 12 months and the proportion of the author's recommendations that are 'buy', 'hold', 'sell' or other over the previous 12 months is available by logging into Marquee ContentStream using the link below. Alternatively, if you do not have access to Marquee ContentStream, please contact your usual GS representative who will be able to provide this information to you.
# 
# Backtesting, Simulated Results, Sensitivity/Scenario Analysis or Spreadsheet Calculator or Model: There may be data presented herein that is solely for illustrative purposes and which may include among other things back testing, simulated results and scenario analyses. The information is based upon certain factors, assumptions and historical information that Goldman Sachs may in its discretion have considered appropriate, however, Goldman Sachs provides no assurance or guarantee that this product will operate or would have operated in the past in a manner consistent with these assumptions. In the event any of the assumptions used do not prove to be true, results are likely to vary materially from the examples shown herein. Additionally, the results may not reflect material economic and market factors, such as liquidity, transaction costs and other expenses which could reduce potential return.
# 
# OTC Derivatives Risk Disclosures: 
# Terms of the Transaction: To understand clearly the terms and conditions of any OTC derivative transaction you may enter into, you should carefully review the Master Agreement, including any related schedules, credit support documents, addenda and exhibits. You should not enter into OTC derivative transactions unless you understand the terms of the transaction you are entering into as well as the nature and extent of your risk exposure. You should also be satisfied that the OTC derivative transaction is appropriate for you in light of your circumstances and financial condition. You may be requested to post margin or collateral to support written OTC derivatives at levels consistent with the internal policies of Goldman Sachs. 
#  
# Liquidity Risk: There is no public market for OTC derivative transactions and, therefore, it may be difficult or impossible to liquidate an existing position on favorable terms. Transfer Restrictions: OTC derivative transactions entered into with one or more affiliates of The Goldman Sachs Group, Inc. (Goldman Sachs) cannot be assigned or otherwise transferred without its prior written consent and, therefore, it may be impossible for you to transfer any OTC derivative transaction to a third party. 
#  
# Conflict of Interests: Goldman Sachs may from time to time be an active participant on both sides of the market for the underlying securities, commodities, futures, options or any other derivative or instrument identical or related to those mentioned herein (together, "the Product"). Goldman Sachs at any time may have long or short positions in, or buy and sell Products (on a principal basis or otherwise) identical or related to those mentioned herein. Goldman Sachs hedging and trading activities may affect the value of the Products. 
#  
# Counterparty Credit Risk: Because Goldman Sachs, may be obligated to make substantial payments to you as a condition of an OTC derivative transaction, you must evaluate the credit risk of doing business with Goldman Sachs or its affiliates. 
#  
# Pricing and Valuation: The price of each OTC derivative transaction is individually negotiated between Goldman Sachs and each counterparty and Goldman Sachs does not represent or warrant that the prices for which it offers OTC derivative transactions are the best prices available, possibly making it difficult for you to establish what is a fair price for a particular OTC derivative transaction; The value or quoted price of the Product at any time, however, will reflect many factors and cannot be predicted. If Goldman Sachs makes a market in the offered Product, the price quoted by Goldman Sachs would reflect any changes in market conditions and other relevant factors, and the quoted price (and the value of the Product that Goldman Sachs will use for account statements or otherwise) could be higher or lower than the original price, and may be higher or lower than the value of the Product as determined by reference to pricing models used by Goldman Sachs. If at any time a third party dealer quotes a price to purchase the Product or otherwise values the Product, that price may be significantly different (higher or lower) than any price quoted by Goldman Sachs. Furthermore, if you sell the Product, you will likely be charged a commission for secondary market transactions, or the price will likely reflect a dealer discount. Goldman Sachs may conduct market making activities in the Product. To the extent Goldman Sachs makes a market, any price quoted for the OTC derivative transactions, Goldman Sachs may differ significantly from (i) their value determined by reference to Goldman Sachs pricing models and (ii) any price quoted by a third party. The market price of the OTC derivative transaction may be influenced by many unpredictable factors, including economic conditions, the creditworthiness of Goldman Sachs, the value of any underlyers, and certain actions taken by Goldman Sachs. 
#  
# Market Making, Investing and Lending: Goldman Sachs engages in market making, investing and lending businesses for its own account and the accounts of its affiliates in the same or similar instruments underlying OTC derivative transactions (including such trading as Goldman Sachs deems appropriate in its sole discretion to hedge its market risk in any OTC derivative transaction whether between Goldman Sachs and you or with third parties) and such trading may affect the value of an OTC derivative transaction. 
#  
# Early Termination Payments: The provisions of an OTC Derivative Transaction may allow for early termination and, in such cases, either you or Goldman Sachs may be required to make a potentially significant termination payment depending upon whether the OTC Derivative Transaction is in-the-money to Goldman Sachs or you at the time of termination. Indexes: Goldman Sachs does not warrant, and takes no responsibility for, the structure, method of computation or publication of any currency exchange rates, interest rates, indexes of such rates, or credit, equity or other indexes, unless Goldman Sachs specifically advises you otherwise.
# Risk Disclosure Regarding futures, options, equity swaps, and other derivatives as well as non-investment-grade securities and ADRs: Please ensure that you have read and understood the current options, futures and security futures disclosure document before entering into any such transactions. Current United States listed options, futures and security futures disclosure documents are available from our sales representatives or at http://www.theocc.com/components/docs/riskstoc.pdf,  http://www.goldmansachs.com/disclosures/risk-disclosure-for-futures.pdf and https://www.nfa.futures.org/investors/investor-resources/files/security-futures-disclosure.pdf, respectively. Certain transactions - including those involving futures, options, equity swaps, and other derivatives as well as non-investment-grade securities - give rise to substantial risk and are not available to nor suitable for all investors. If you have any questions about whether you are eligible to enter into these transactions with Goldman Sachs, please contact your sales representative. Foreign-currency-denominated securities are subject to fluctuations in exchange rates that could have an adverse effect on the value or price of, or income derived from, the investment. In addition, investors in securities such as ADRs, the values of which are influenced by foreign currencies, effectively assume currency risk.
# Options Risk Disclosures: Options may trade at a value other than that which may be inferred from the current levels of interest rates, dividends (if applicable) and the underlier due to other factors including, but not limited to, expectations of future levels of interest rates, future levels of dividends and the volatility of the underlier at any time prior to maturity. Note: Options involve risk and are not suitable for all investors. Please ensure that you have read and understood the current options disclosure document before entering into any standardized options transactions. United States listed options disclosure documents are available from our sales representatives or at http://theocc.com/publications/risks/riskstoc.pdf. A secondary market may not be available for all options. Transaction costs may be a significant factor in option strategies calling for multiple purchases and sales of options, such as spreads. When purchasing long options an investor may lose their entire investment and when selling uncovered options the risk is potentially unlimited. Supporting documentation for any comparisons, recommendations, statistics, technical data, or other similar information will be supplied upon request.
# This material is for the private information of the recipient only. This material is not sponsored, endorsed, sold or promoted by any sponsor or provider of an index referred herein (each, an "Index Provider"). GS does not have any affiliation with or control over the Index Providers or any control over the computation, composition or dissemination of the indices. While GS will obtain information from publicly available sources it believes reliable, it will not independently verify this information. Accordingly, GS shall have no liability, contingent or otherwise, to the user or to third parties, for the quality, accuracy, timeliness, continued availability or completeness of the data nor for any special, indirect, incidental or consequential damages which may be incurred or experienced because of the use of the data made available herein, even if GS has been advised of the possibility of such damages.
# Standard & Poor's ® and S&P ® are registered trademarks of The McGraw-Hill Companies, Inc. and S&P GSCI™ is a trademark of The McGraw-Hill Companies, Inc. and have been licensed for use by the Issuer. This Product (the "Product") is not sponsored, endorsed, sold or promoted by S&P and S&P makes no representation, warranty or condition regarding the advisability of investing in the Product.
# Notice to Brazilian Investors
# Marquee is not meant for the general public in Brazil. The services or products provided by or through Marquee, at any time, may not be offered or sold to the general public in Brazil. You have received a password granting access to Marquee exclusively due to your existing relationship with a GS business located in Brazil. The selection and engagement with any of the offered services or products through Marquee, at any time, will be carried out directly by you. Before acting to implement any chosen service or products, provided by or through Marquee you should consider, at your sole discretion, whether it is suitable for your particular circumstances and, if necessary, seek professional advice. Any steps necessary in order to implement the chosen service or product, including but not limited to remittance of funds, shall be carried out at your discretion. Accordingly, such services and products have not been and will not be publicly issued, placed, distributed, offered or negotiated in the Brazilian capital markets and, as a result, they have not been and will not be registered with the Brazilian Securities and Exchange Commission (Comissão de Valores Mobiliários), nor have they been submitted to the foregoing agency for approval. Documents relating to such services or products, as well as the information contained therein, may not be supplied to the general public in Brazil, as the offering of such services or products is not a public offering in Brazil, nor used in connection with any offer for subscription or sale of securities to the general public in Brazil.
# The offer of any securities mentioned in this message may not be made to the general public in Brazil. Accordingly, any such securities have not been nor will they be registered with the Brazilian Securities and Exchange Commission (Comissão de Valores Mobiliários) nor has any offer been submitted to the foregoing agency for approval. Documents relating to the offer, as well as the information contained therein, may not be supplied to the public in Brazil, as the offer is not a public offering of securities in Brazil. These terms will apply on every access to Marquee.
# Ouvidoria Goldman Sachs Brasil: 0800 727 5764 e/ou ouvidoriagoldmansachs@gs.com
# Horário de funcionamento: segunda-feira à sexta-feira (exceto feriados), das 9hs às 18hs.
# Ombudsman Goldman Sachs Brazil: 0800 727 5764 and / or ouvidoriagoldmansachs@gs.com
# Available Weekdays (except holidays), from 9 am to 6 pm.
# 
# 
