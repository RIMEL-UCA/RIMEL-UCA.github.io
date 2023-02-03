#!/usr/bin/env python
# coding: utf-8

# In[8]:


from gs_quant.data import Dataset
from gs_quant.markets.portfolio import Portfolio
from gs_quant.timeseries import returns
from gs_quant.risk import MarketDataPattern, MarketDataShock, MarketDataShockBasedScenario, MarketDataShockType
from gs_quant.instrument import IRSwap
from gs_quant.common import PayReceive
from gs_quant import risk
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gs_quant.datetime.date import business_day_offset

# In[9]:


from gs_quant.session import GsSession
# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics', 'read_product_data'))

# In this notebook, we'll look at how several rate and fx instruments moved in the 2012 and 2016 elections and use these moves as shocks for our portfolio.
# 
# The content of this notebook is split into the following parts:
# * [1: Visualize last 2 elections](#1:-Visualize-last-2-elections)
# * [2: Grab portfolio](#2:-Grab-portfolio)
# * [3: Use prior elections as shocks](#3:-Use-prior-elections-as-shocks)
# * [4: Custom shocks](#4:-Custom-shocks)

# ### 1: Visualize last 2 elections
# 
# Let's start by retrieving the data from the [Marquee data catalog](https://marquee.gs.com/s/discover/data-services/catalog) and visualizing it.

# In[10]:


elec_12 = datetime.date(2012, 11, 6)
elec_16 = datetime.date(2016, 11, 8)
elec_20 = datetime.date(2020, 11, 3)

t0 = datetime.date(2012, 1, 1)
tn = datetime.date.today()

fx = Dataset('FXSPOT_PREMIUM').get_data(t0, tn, bbid='AUDJPY', tenor='3m', fields=('spot',))[['spot']].spot
fx_vol = Dataset('FXIMPLIEDVOL_PREMIUM').get_data(t0, tn, bbid='AUDJPY', deltaStrike='25DC', tenor='3m')[['impliedVolatility']].impliedVolatility
rates = Dataset('IR_SWAP2').get_data(t0, assetId=['MAAXGV0GZTW4GFNC'], tenor='30y', fields=('rate',))[['rate']].rate

# In[11]:


def slice_data(df, y_0):
    df_period = pd.concat((df.loc[x:y].reset_index() for x, y in
                           [(datetime.date(t, 8, 1), datetime.date(t + 1, 1, 31)) for t in [2012, 2016]]), axis=1)
    df_period['month'] = pd.DatetimeIndex(df_period.date.iloc[:, 0]).month_name()
    df_period.set_index('month', drop=True, inplace=True)
    df_period.drop('date', axis=1, inplace=True)
    df_period.columns = ['Democrat win 2012', 'Republican win 2016']
    return df_period

def plot_data(title, data):
    ax = data.plot(title=title, figsize=(10, 4))
    plt.axvline(x=len(fx[datetime.date(2012, 8, 1):datetime.date(2012, 11, 5)]), linestyle='--', color='r')
    ax.set_xticks([21*k for k in range(6)])
    _ = ax.set_xticklabels(['August', 'September', 'October', 'November', 'December', 'January'], horizontalalignment='left')

# In[12]:


fx_sliced, fx_vol_sliced, rates_sliced = [slice_data(x, 2012) for x in [fx, fx_vol, rates]]

# In[13]:


plot_data('AUDJPY', fx_sliced)
plot_data('AUDJPY 3m vol', fx_vol_sliced)
plot_data('30y rates', rates_sliced)

# ### 2: Grab portfolio

# Let's use our FX book from the previous notebook and add a few rates trades. We'll shock this book in the next step.

# In[14]:


mappers = {
    'pair': lambda row: row['foreign ccy'] + 'USD',
    'notional_amount': 'notional',
    'expiration_date': 'expiry',
    'option_type': lambda row: 'Call' if row['C/P'] == 'C' else 'Put',
    'strike_price': 'strike',
    'premium': lambda row: 0
}

portfolio = Portfolio.from_csv(r'FXBook.csv', mappings=mappers)

swap_10y = IRSwap(PayReceive.Receive, '10y', 'USD', notional_amount= 250e6, fixed_rate='atm+25')  
swap_5y = IRSwap(PayReceive.Pay, '5y', 'USD', notional_amount= 500e6, fixed_rate='atm+20') 

portfolio.append((swap_10y, swap_5y))
portfolio.resolve()
frame = portfolio.to_frame()
frame.index = frame.index.droplevel(0)
frame.replace(np.nan,'',True) 
frame.tail(3)

# ### 3: Use prior market moves as shocks
# 
# Let's now use the moves we examined in the first section as shocks to the portfolio in the second section.

# In[15]:


time_window = 2 
effective12 = business_day_offset(elec_12, time_window) 
effective16 = business_day_offset(elec_16, time_window)  

fx_shocks = fx.pct_change(time_window+5).loc[[effective12, effective16]] 
fx_vol_shocks = fx_vol.pct_change(time_window+5).loc[[effective12, effective16]] 
ir_shocks = rates.diff(time_window+5).loc[[effective12, effective16]] 

shock_df = pd.concat([fx_shocks*1e2, fx_vol_shocks*1e2, ir_shocks*1e4], axis=1).round(1) 
shock_df.index = ['2012 Election', '2016 Election'] 

def get_market_scenario(s_date): 
    return MarketDataShockBasedScenario(shocks={ 
        MarketDataPattern('IR', 'USD'): MarketDataShock(MarketDataShockType.Absolute, ir_shocks.loc[s_date]), 
        MarketDataPattern('FX', 'USD/AUD'): MarketDataShock(MarketDataShockType.Proportional, fx_shocks.loc[s_date]), 
        MarketDataPattern('FX', 'JPY/USD'): MarketDataShock(MarketDataShockType.Proportional, 0), 
        MarketDataPattern('FX Vol', 'JPY/AUD', 'ATM Vol'): MarketDataShock(MarketDataShockType.Absolute, fx_vol_shocks.loc[s_date]),
        MarketDataPattern('FX Vol', 'USD/AUD', 'ATM Vol'): MarketDataShock(MarketDataShockType.Absolute, fx_vol_shocks.loc[s_date]),
        MarketDataPattern('FX Vol', 'JPY/USD', 'ATM Vol'): MarketDataShock(MarketDataShockType.Absolute, fx_vol_shocks.loc[s_date]),

    }) 

win_12 = get_market_scenario(effective12) 
win_16 = get_market_scenario(effective16) 
shock_df

# In[16]:


# scenario carrying swap to election day and then applying spot shock
results, results_agg = {}, {}

with win_12:
    results['2012'] = portfolio.dollar_price()

with win_16:
    results['2016'] = portfolio.dollar_price()

results['eod'] = portfolio.dollar_price()

for scenario in ['2012', '2016']:
    results_agg[scenario] = results[scenario].aggregate() - results['eod'].aggregate()

plt.bar(results_agg.keys(), results_agg.values())

# ### 4: Custom shocks
# 
# Finally let's look at how to run a completely custom shock.

# In[17]:


# get points available to shock by looking at irdelta 
threshold = 25000
delta = portfolio.calc(risk.IRDelta)
delta.aggregate()[(delta.aggregate().value>threshold) | (delta.aggregate().value<-threshold)].round()

# In[19]:


custom_shocks = pd.read_csv(r'curve_shocks.csv', na_values='')
custom_shocks

# In[20]:


# Fill in your shocks programatically or in excel
custom_shocks['shock'] = [50, 40, 30]
custom_shocks

# In[21]:


shocks = {
    MarketDataPattern(custom_shocks.at[i, 'mkt_type'], custom_shocks.at[i, 'mkt_asset'], custom_shocks.at[i, 'mkt_class']):
    MarketDataShock(MarketDataShockType.Absolute, custom_shocks.at[i, 'shock'] / 1e4) for i in range(len(custom_shocks)-2)
}

shocks.update({
    MarketDataPattern(custom_shocks.at[i, 'mkt_type'], custom_shocks.at[i, 'mkt_asset'], custom_shocks.at[i, 'mkt_class'],
                      (custom_shocks.at[i, 'mkt_point'],)): 
    MarketDataShock(MarketDataShockType.Absolute, custom_shocks.at[i, 'shock'] / 1e4) for i in [len(custom_shocks)-2, len(custom_shocks)-1]
})

with MarketDataShockBasedScenario(shocks):
    results_agg['custom shocks'] = portfolio.dollar_price().aggregate() - results['eod'].aggregate()

plt.bar(results_agg.keys(), results_agg.values())

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
