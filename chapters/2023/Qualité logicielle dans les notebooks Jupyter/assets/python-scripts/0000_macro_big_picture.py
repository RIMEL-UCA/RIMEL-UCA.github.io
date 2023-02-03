#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gs_quant.data import Dataset
from gs_quant.markets.securities import Asset, AssetIdentifier, SecurityMaster
from gs_quant.datetime import *
from gs_quant.api.fred.data import FredDataApi
from gs_quant.timeseries import returns, diff
from sklearn.decomposition import PCA
from collections import defaultdict
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd

# In[2]:


from gs_quant.session import GsSession
# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics', 'read_product_data'))

# In this notebook, we'll take a look at the current macro landscape and understand how risk has evolved in the last 6mo through the lens of principal component analysis (PCA).
# 
# The content of this notebook is split into the following parts:
# * [1: First, the data](#1:-First,-the-data)
# * [2: PCA](#2:-PCA)
# * [3: Interpreting top 2 risk drivers](#3:-Interpreting-top-2-risk-drivers)
# * [4: Realized vs predicted](#4:-Realized-vs-predicted)

# ### 1: First, the data
# 
# Let’s start by pulling volatility data from [gs data catalog](https://marquee.gs.com/s/discover/data-services/catalog) to get a sense for the macro landscape.

# In[3]:


start_date = dt.date(2012, 1, 3)
end_date = business_day_offset(dt.date.today(), -1, roll='preceding')

# gs data sets
fxspot, irspot, fxvol, irvol = [Dataset(x) for x in ['FXSPOT_PREMIUM', 'IR_SWAP_RATES', 'FXIMPLIEDVOL_PREMIUM', 'IR_SWAPTION_VOLS']]

fxv = fxvol.get_data(start_date, end_date, bbid=['EURUSD', 'USDJPY', 'AUDJPY', 'CADUSD', 'AUDUSD'], tenor='3m', deltaStrike='DN', location='NYC')
pd.pivot_table(fxv, values='impliedVolatility', index=['date'], columns=['bbid'], aggfunc=np.sum).plot(figsize=(10, 6), title='3m implied fx vol')

# In[4]:


irv = irvol.get_data(start_date, end_date, payOrReceive='Pay', currency=['USD', 'EUR', 'JPY', 'AUD'], terminationTenor='10y', expirationTenor='3m', strikeRelative='ATM')
pd.pivot_table(irv, values='impliedNormalVolatility', index=['date'], columns=['currency'], aggfunc=np.sum).plot(figsize=(10, 6), title='3m10y implied rate vol')

# In[5]:


FRED_API_KEY = 'insert key'
fred_API = FredDataApi(api_key=FRED_API_KEY)
fred_pull = fred_API.build_query(start=start_date, end=end_date)
fred_API.query_data(fred_pull, dataset_id='VIXCLS').plot(figsize=(10, 6), title='VIX')

# ### 2: PCA

# Let's now look across a number of macro series and run PCA to understand what's driving risk - to start we'll grab more data from [gs data catalog](https://marquee.gs.com/s/discover/data-services/catalog) and FRED and normalize it.
# 
# * GS FX spot [here](https://marquee.gs.com/s/developer/datasets/FXSPOT_PREMIUM) and vol [here](https://marquee.gs.com/s/developer/datasets/FXIMPLIEDVOL_PREMIUM)
# * GS Cash and Swap Rates [here](https://marquee.gs.com/s/developer/datasets/IR_SWAP_RATES) and vol [here]()
# * Fred API [here](https://research.stlouisfed.org/useraccount/login/secure/) 

# In[6]:


instruments = {
    'Equities' : ['SPX', 'NDX', 'N225'],
    'Commodities': ['Gold', 'WTI'],
    'Rates': ['5y', '30y'],
    'FX': ['USDJPY', 'EURUSD', 'USDCAD', 'AUDJPY'],
    'Credit': ['HY'],
    'Fundamental': ['Breakevens', 'VIX']
}

# In[7]:


fred_symbols = {
    'SPX': 'SP500',
    'NDX': 'NASDAQ100',
    'N225': 'NIKKEI225',
    'Gold': 'GOLDAMGBD228NLBM',
    'WTI': 'DCOILWTICO',   
    'HY': 'BAMLH0A0HYM2', 
    'VIX': 'VIXCLS',
    'Breakevens': 'T10YIE'
}

color_map = {
    'Equities': '#20396D',
    'FX': '#68A2BF',
    'Rates': '#CD252B',
    'Commodities': '#E3E311',
    'Credit': '#E3E000',
    'Vol': '#67E311',
    'Fundamental': '#25cdae',
}

realVolWindow = 66 
asset_map = {}
df = pd.DataFrame()

for asset_type, asset in instruments.items():
    for x in asset:
        asset_map[x] = asset_type 
        if asset_type=='FX':
            asset_map['{}_3m_impvol'.format(x)]= asset_type
            asset_map['{}_3m_impreal'.format(x)]= asset_type
            
            df[x] = fxspot.get_data(start_date, end_date, bbid=x)[['spot']]
            realVol_f = df[x].copy()
            realVol_f = [returns(realVol_f)[r-realVolWindow:r].std()*15.875 for r in range(len(realVol_f))]
            
            df['{}_3m_impvol'.format(x)] = fxvol.get_data(start_date, end_date, bbid=x, tenor='3m', deltaStrike='DN', location='NYC')[['impliedVolatility']]
            df['{}_3m_impreal'.format(x)] = fxvol.get_data(start_date, end_date, bbid=x, tenor='3m', deltaStrike='DN', location='NYC')[['impliedVolatility']]
            df['{}_3m_impreal'.format(x)] /= realVol_f
        elif asset_type=='Rates':
            asset_map['1y{}_curve'.format(x)] = asset_type
            asset_map['{}_xmkt'.format(x)] = asset_type
            asset_map['3m{}_impvol'.format(x)]= asset_type
            asset_map['3m{}_impreal'.format(x)]= asset_type
            asset_map['3y_vs_3m_{}_calendar'.format(x)]= asset_type
            asset_map['3y{}_skew'.format(x)]= asset_type
            
            irspot_d = irspot.get_data(start_date, end_date, currency='USD',effectiveTenor = '0b', terminationTenor=x, floatingRateOption='USD-LIBOR-BBA', floatingRateDesignatedMaturity='3m', clearingHouse='LCH', pricingLocation='NYC')
            irfront_d = irspot.get_data(start_date, end_date, currency='USD',effectiveTenor = '0b', terminationTenor='1y', floatingRateOption='USD-LIBOR-BBA', floatingRateDesignatedMaturity='3m', clearingHouse='LCH', pricingLocation='NYC')
            irspot_eur_d = irspot.get_data(start_date, end_date, currency='EUR',effectiveTenor = '0b', terminationTenor=x, floatingRateOption='EUR-EURIBOR-TELERATE', floatingRateDesignatedMaturity='6m', clearingHouse='LCH', pricingLocation='LDN')
            gammaVolA = irvol.get_data(start_date, end_date, payOrReceive='Pay', currency='USD', terminationTenor=x, floatingRateOption='USD-LIBOR-BBA', expirationTenor='3m', clearingHouse='LCH', strikeRelative='ATM', pricingLocation='NYC')[['impliedNormalVolatility']]
            vegaVolA = irvol.get_data(start_date, end_date, payOrReceive='Pay', currency='USD', terminationTenor=x, floatingRateOption='USD-LIBOR-BBA', expirationTenor='3y', clearingHouse='LCH', strikeRelative='ATM', pricingLocation='NYC')[['impliedNormalVolatility']]
            vegaVolhigh = irvol.get_data(start_date, end_date, payOrReceive='Pay', currency='USD', terminationTenor=x, floatingRateOption='USD-LIBOR-BBA', expirationTenor='3y', clearingHouse='LCH', strikeRelative='ATM+50', pricingLocation='NYC')[['impliedNormalVolatility']]   
            df[x] = irspot_d.rate.copy()*1e4
            realVol_r = [df[x].diff()[r-realVolWindow:r].std() for r in range(len(df[x]))]
            
            df['1y{}_curve'.format(x)] = (irspot_d.rate - irfront_d.rate) * 1e4
            df['{}_xmkt'.format(x)] = (irspot_d.rate - irspot_eur_d.rate)*1e4 
            df['3m{}_impvol'.format(x)] = gammaVolA
            df['3m{}_impreal'.format(x)] = gammaVolA
            df['3m{}_impreal'.format(x)] /= realVol_r
            df['3y_vs_3m_{}_calendar'.format(x)] = vegaVolA/gammaVolA
            df['3y{}_skew'.format(x)] = vegaVolhigh/vegaVolA
        else:
            df[x] = fred_API.query_data(fred_pull, dataset_id=fred_symbols[x])
            
data = df
data = data.fillna(method='ffill').dropna()
data.tail().head()

# In[8]:


frame_t = data.copy().apply(lambda x: x.diff() if x.name in instruments['Rates'] else returns(x))
frame_t.dropna(inplace=True)

# Let's run a 3-factor PCA for a 126 day (6mo) rolling period and record how much variance is explained by each component over our time frame.

# In[9]:


period = 126
components = 3
f_loadings = defaultdict(list)

for start in range(len(frame_t) - period):
    d = frame_t.iloc[start:start + period]
    d = (d - d.mean()) / d.std()
    model = PCA(n_components=components)
    model.fit(d)
    for i, c in enumerate(model.explained_variance_ratio_):
        f_loadings[i].append(c)

# In[10]:


factors = pd.DataFrame(f_loadings)
factors.index = frame_t.index[period:]
plt.figure(figsize=(12, 8))
plt.stackplot(factors.index, factors.transpose())
plt.title('Contribution to variance from each component')

# ### 3: Interpreting top 2 risk drivers
# Let’s now look at the top 2 components explaining risk in 2020 as well as over the entire period. Note, components can rotate over time, so we look at the absolute ratios of PC1 vs PC2 and vise versa to understand the primary drivers.

# In[11]:


# model 1 trained on full dataset
full_model = PCA(n_components=components)
full_data = (frame_t - frame_t.mean()) / frame_t.std()
full_model.fit(full_data)
components_full = pd.DataFrame(full_model.components_, columns=frame_t.columns)

# model 2 trained on 2020
model_2020 = PCA(n_components=components)
data2 = frame_t[frame_t.index.year == 2020]
model_2020.fit((data2 - data2.mean()) / data2.std())
components_2020 = pd.DataFrame(model_2020.components_, columns=data.columns, index=factors.columns)

# In[12]:


fig = plt.figure(figsize=(16, 16))
comp1, comp2 = 0, 1
for i, row in components_full.iteritems():
    plt.scatter(row[comp1], row[comp2], color=color_map[asset_map[row.name]], label=row.name)
    plt.text(row[comp1]-0.02, row[comp2]-.01, row.name, fontsize=9)

plt.axhline(0, color='grey')
plt.axvline(0, color='grey')
plt.title('PC1 vs PC2 - Full History')
plt.xlabel('PC'+str(comp1+1))
plt.ylabel('PC'+str(comp2+1))

# In[13]:


fig = plt.figure(figsize=(20, 20))

for (_, col1), (_, col2) in zip(components_full.iteritems(), components_2020.iteritems()):
    plt.scatter(col2[comp1], col2[comp2], color='pink')
    plt.scatter(col1[comp1], col1[comp2], color=color_map[asset_map[col2.name]])
    plt.plot((col2[comp1], col1[comp1]), (col2[comp2], col1[comp2]), color='grey', alpha=0.3)
    plt.text(col2[comp1], col2[comp2], col1.name, fontsize=9)

plt.axhline(0, color='grey')
plt.axvline(0, color='grey')

plt.title('PC1 vs PC2 - Full History vs 2020 Data only')
plt.xlabel('PC'+str(comp1+1))
plt.ylabel('PC'+str(comp2+1))

# In[14]:


PC1_f, PC2_f = components_full.iloc[0].abs(), components_full.iloc[1].abs()
PC1_2020, PC2_2020 = components_2020.iloc[0].abs(), components_2020.iloc[1].abs()

plt.subplot(2, 2, 1)
(PC1_f/PC2_f).sort_values(ascending=False).head(15).plot(kind='barh', title='PC1/PC2 - full', figsize=(16, 8))
plt.subplot(2, 2, 2)
(PC2_f/PC1_f).sort_values(ascending=False).head(15).plot(kind='barh', title='PC2/PC1 - full', figsize=(16, 8))
plt.subplot(2, 2, 3)
(PC1_2020/PC2_2020).sort_values(ascending=False).head(15).plot(kind='barh', title='PC1/PC2 - 2020', figsize=(16, 8))
plt.subplot(2, 2, 4)
(PC2_2020/PC1_2020).sort_values(ascending=False).head(15).plot(kind='barh', title='PC2/PC1 - 2020', figsize=(16, 8))

# ### 4: Realized vs modeled
# 
# Let’s now look at where the 3 component PCA model would tell us the returns for several series of interest should be vs where they have realized. The series with the largest model deviations present potential dislocations.

# In[15]:


pca_returns = pd.DataFrame(full_model.transform(full_data), index=full_data.index)
res = pca_returns.dot(components_full) * frame_t.std() + frame_t.mean()

# In[16]:


def transform(tr_df, since):
    t_d = tr_df.copy()[since:]
    for key, v in t_d.items():
        t_d[key] = (1 + t_d[key] / 100).cumprod() if key in instruments['Rates'] else t_d[key].cumsum()
    return t_d

def compare_plot(asset, actual, predicted):
    plt.figure(figsize=(10, 4))
    plt.plot(actual[asset], label='actual')
    plt.plot(predicted[asset], label='predicted')
    plt.title(asset)
    plt.legend()
    plt.show()

# In[17]:


# compare each of the variables since beginning of the year
since = '2020-01-01'
predicted = transform(res, since)
actual = transform(frame_t, since)

fx_graphs = ['USDCAD', 'AUDJPY'] + [f+'_3m_impvol' for f in ['USDJPY', 'AUDJPY']]
for asset in fx_graphs:
    compare_plot(asset, actual, predicted)

# In[18]:


rates_graphs = ['5y', '30y']
rates_vols = ['3m'+r+'_impvol' for r in rates_graphs]  + ['3y_vs_3m_'+r+'_calendar' for r in rates_graphs] + ['3y5y_skew']

for asset in rates_vols:
    compare_plot(asset, actual, predicted)

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
