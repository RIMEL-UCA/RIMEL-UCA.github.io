#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gs_quant.markets.securities import AssetIdentifier, SecurityMaster
from gs_quant.timeseries.measures import forward_vol, VolReference, implied_volatility
from gs_quant.timeseries.algebra import *
from gs_quant.timeseries.analysis import *
from gs_quant.data import DataContext
from gs_quant.instrument import EqOption
from gs_quant.markets.portfolio import Portfolio
from gs_quant.risk import EqDelta, EqGamma, EqVega, DollarPrice, Price
from gs_quant.markets import PricingContext
from gs_quant.backtests.triggers import *
from gs_quant.backtests.actions import *
from gs_quant.backtests import FlowVolBacktestMeasure
from gs_quant.backtests.strategy import Strategy
from gs_quant.backtests.equity_vol_engine import EquityVolEngine
from gs_quant.session import GsSession, Environment

from dateutil.relativedelta import relativedelta
from datetime import date, datetime
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,.2f}'.format

# In[2]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In this notebook, we will take a look at historical and future implied trends in equity implied volatility, formulate a trade idea, analyze a trading strategy backtest and calculate spot price and greeks. This is an example of the GS Quant functionalities and it is not a recommendation of a trade in the instruments discussed, please follow up with your salesperson to discuss specific implementations.
# 
# * [1 - Analyze Vol Spread](#2---Analyze-Vol-Spread)
# * [2 - Trade Idea & Backtest](#3---Trade-Idea-&-Backtest)
# * [3 - Compare Performance](#4---Compare-Performance)
# * [4 - Price & Greeks](#5---Price-&-Greeks)

# ## 1 - Analyze Vol Spread
# 
# Let's start by looking at historical forward vol for SPX and SX5E.

# In[3]:


def forward_vol_spread(asset1, asset2, tenor, forward_date):
    '''
    Calculate the forward starting vol spread between two assets
    '''
    asset1_sec = SecurityMaster.get_asset(asset1, AssetIdentifier.BLOOMBERG_ID)
    asset2_sec = SecurityMaster.get_asset(asset2, AssetIdentifier.BLOOMBERG_ID)
    
    implied = lambda asset: implied_volatility(asset, tenor, VolReference.FORWARD, 100)
    forward = lambda asset: forward_vol(asset, tenor, forward_date, VolReference.FORWARD, 100)
    
    asset1_fvol = implied(asset1_sec) if forward_date=='0m' else forward(asset1_sec)
    asset2_fvol = implied(asset2_sec) if forward_date=='0m' else forward(asset2_sec)
                                                                        
    return subtract(asset1_fvol, asset2_fvol)

# In[4]:


start_date, end_date = date(2014, 1, 1), date(2020, 11, 2)
DataContext.current = DataContext(start=start_date, end=end_date)

sx5e_spx_fvs = forward_vol_spread('SX5E', 'SPX', '2y', '1y')

current_roll_data = {}
for rMonth in [0, 6, 9, 12]:
    d = end_date + relativedelta(months=rMonth)
    months_to_start = 12 - rMonth
    curve = forward_vol_spread('SX5E', 'SPX', '2y', f'{months_to_start}m')
    if curve.size:
        current_roll_data[d] = last_value(curve)
current_roll = pd.Series(current_roll_data)

# In[5]:


sx5e_spx_fvs.plot(figsize=(12, 6), legend=True, label='SX5E vs SPX', title='Vol Spread')
current_roll.plot(figsize=(12, 6), legend=True, label='Current Spread Roll-Up')
plt.show()

# ## 2 - Trade Idea & Backtest
# 
# The analysis shows a projected narrowing of vol spread between SX5E and SPX. Let's create an option trading strategy that takes advantage of this and backtest it to see how it could have performed.

# In[6]:


# Create instruments
sx5e_straddle = (EqOption('.STOXX50E', '3y', 'ATMF', 'Call'), EqOption('.STOXX50E', '3y', 'ATMF','Put'))
spx_straddle = (EqOption('.SPX', '3y', 'ATMF', 'Call', multiplier=1), EqOption('.SPX', '3y', 'ATMF', 'Put', multiplier=1))

# Define triggers for trade and hedge actions.

# Trade and roll 1 sx5e_straddle every 1m
trade_sx5e_action = EnterPositionQuantityScaledAction(priceables=Portfolio(sx5e_straddle), trade_duration='1m', trade_quantity=1, trade_quantity_type=BacktestTradingQuantityType.quantity)
trade_sx5e_trigger = PeriodicTrigger(trigger_requirements=PeriodicTriggerRequirements(start_date=start_date, end_date=end_date, frequency='1m'),
                                actions=trade_sx5e_action)

# Hedge sx5e_straddle delta every business day
hedge_sx5e_action = HedgeAction(EqDelta, priceables=Portfolio(sx5e_straddle), trade_duration='B')
hedge_sx5e_trigger = PeriodicTrigger(trigger_requirements=PeriodicTriggerRequirements(start_date=start_date, end_date=end_date, frequency='B'),
                                actions=hedge_sx5e_action)

# Trade and roll 1 spx_straddle every 1m
trade_spx_action = EnterPositionQuantityScaledAction(priceables=Portfolio(spx_straddle), trade_duration='1m', trade_quantity=1, trade_quantity_type=BacktestTradingQuantityType.quantity)
trade_spx_trigger = PeriodicTrigger(trigger_requirements=PeriodicTriggerRequirements(start_date=start_date, end_date=end_date, frequency='1m'),
                                actions=trade_spx_action)

# Hedge spx_straddle delta every business day
hedge_spx_action = HedgeAction(EqDelta, priceables=Portfolio(spx_straddle), trade_duration='B')
hedge_spx_trigger = PeriodicTrigger(trigger_requirements=PeriodicTriggerRequirements(start_date=start_date, end_date=end_date, frequency='B'),
                                actions=hedge_spx_action)

EqVolE = EquityVolEngine()

# Define SX5E backtest
triggers_sx5e = [trade_sx5e_trigger, hedge_sx5e_trigger]

strategy_sx5e_result = EqVolE.run_backtest(Strategy(None, triggers_sx5e), start=start_date, end=end_date)
perf_sx5e = strategy_sx5e_result.get_measure_series(FlowVolBacktestMeasure.PNL)

# Define SPX backtest
triggers_spx = [trade_spx_trigger, hedge_spx_trigger]

strategy_spx_result = EqVolE.run_backtest(Strategy(None, triggers_spx), start=start_date, end=end_date)
perf_spx = strategy_spx_result.get_measure_series(FlowVolBacktestMeasure.PNL)


# In[7]:


perf_sx5e_spx_spread = subtract(perf_sx5e, perf_spx)
perf_sx5e_spx_spread.index = pd.DatetimeIndex(perf_sx5e_spx_spread.index)
perf_sx5e_spx_spread.plot(figsize=(12, 6), title='Strategy SX5E vs SPX 3y Vol Spread')
plt.show()

# ## 3 - Compare Performance
# 
# Now let's plot the trading strategy with the actual SX5E vs SPX vol spread to see how the two compare.

# In[8]:


fig, ax1 = plt.subplots()

# left axis
ax1.set_xlabel('date')
ax1.set_ylabel('Vol Spread',color='tab:red')

#right axis
ax2 = ax1.twinx() 
ax2.set_ylabel('Strategy PNL', color='tab:blue')  

sx5e_spx_fvs.plot(figsize=(12, 6), ax=ax1, color='tab:red')
current_roll.plot(figsize=(12, 6), ax=ax1, color='tab:green', legend=True, label='Current Spread Roll-Up')
perf_sx5e_spx_spread.plot(figsize=(12, 6), ax=ax2, color='tab:blue')

fig.tight_layout() 
plt.xlim(start_date, end_date + relativedelta(years=1))
plt.show()

# ## 4 - Price & Greeks
# 
# Finallly, let's calculate current strategy price and greeks.

# In[9]:


sx5e_straddle_port, spx_straddle_port = Portfolio(sx5e_straddle), Portfolio(spx_straddle)

# Price and greeks in USD
sx5e_straddle_price = sx5e_straddle_port.calc(DollarPrice).aggregate()
spx_straddle_price = spx_straddle_port.calc(DollarPrice).aggregate()
print(f'Strategy Price$: {sx5e_straddle_price - spx_straddle_price}')

greeks = (EqDelta, EqGamma, EqVega)
sx5e_straddle_greeks = sx5e_straddle_port.calc(greeks).aggregate().to_frame()
spx_straddle_greeks = spx_straddle_port.calc(greeks).aggregate().to_frame()
pd.concat([spx_straddle_greeks,sx5e_straddle_greeks],keys=['SPX', 'SX5E']).transpose()

# ### Disclaimers
# 
# Indicative Terms/Pricing Levels: This material may contain indicative terms only, including but not limited to pricing levels. There is no representation that any transaction can or could have been effected at such terms or prices. Proposed terms and conditions are for discussion purposes only. Finalized terms and conditions are subject to further discussion and negotiation.
# www.goldmansachs.com/disclaimer/sales-and-trading-invest-rec-disclosures.html If you are not accessing this material via Marquee ContentStream, a list of the author's investment recommendations disseminated during the preceding 12 months and the proportion of the author's recommendations that are 'buy', 'hold', 'sell' or other over the previous 12 months is available by logging into Marquee ContentStream using the link below. Alternatively, if you do not have access to Marquee ContentStream, please contact your usual GS representative who will be able to provide this information to you.
# Backtesting, Simulated Results, Sensitivity/Scenario Analysis or Spreadsheet Calculator or Model: There may be data presented herein that is solely for illustrative purposes and which may include among other things back testing, simulated results and scenario analyses. The information is based upon certain factors, assumptions and historical information that Goldman Sachs may in its discretion have considered appropriate, however, Goldman Sachs provides no assurance or guarantee that this product will operate or would have operated in the past in a manner consistent with these assumptions. In the event any of the assumptions used do not prove to be true, results are likely to vary materially from the examples shown herein. Additionally, the results may not reflect material economic and market factors, such as liquidity, transaction costs and other expenses which could reduce potential return.
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
# iTraxx® is a registered trade mark of International Index Company Limited.
# iTraxx® is a trade mark of International Index Company Limited and has been licensed for the use by Goldman Sachs Japan Co., Ltd. International Index Company Limited does not approve, endorse or recommend Goldman Sachs Japan Co., Ltd. or iTraxx® derivatives products.
# iTraxx® derivatives products are derived from a source considered reliable, but neither International Index Company Limited nor any of its employees, suppliers, subcontractors and agents (together iTraxx Associates) guarantees the veracity, completeness or accuracy of iTraxx® derivatives products or other information furnished in connection with iTraxx® derivatives products. No representation, warranty or condition, express or implied, statutory or otherwise, as to condition, satisfactory quality, performance, or fitness for purpose are given or assumed by International Index Company Limited or any of the iTraxx Associates in respect of iTraxx® derivatives products or any data included in such iTraxx® derivatives products or the use by any person or entity of iTraxx® derivatives products or that data and all those representations, warranties and conditions are excluded save to the extent that such exclusion is prohibited by law.
# None of International Index Company Limited nor any of the iTraxx Associates shall have any liability or responsibility to any person or entity for any loss, damages, costs, charges, expenses or other liabilities whether caused by the negligence of International Index Company Limited or any of the iTraxx Associates or otherwise, arising in connection with the use of iTraxx® derivatives products or the iTraxx® indices.
# Standard & Poor's ® and S&P ® are registered trademarks of The McGraw-Hill Companies, Inc. and S&P GSCI™ is a trademark of The McGraw-Hill Companies, Inc. and have been licensed for use by the Issuer. This Product (the "Product") is not sponsored, endorsed, sold or promoted by S&P and S&P makes no representation, warranty or condition regarding the advisability of investing in the Product.
# 

# In[ ]:



