#!/usr/bin/env python
# coding: utf-8

# <!-- </style><figure align = "left" style="page-break-inside: avoid;"><figcaption style="font-weight: bold; font-size:16pt; font-family:inherit;" align="center"></figcaption><br> --> 
# <img src= "images/APEX.png">
# 

# ## Introduction: What's APEX?
# APEX is a portfolio trade scheduler that optimizes execution with the latest intraday risk and market impact models from Goldman Sachs’ Quantitative Execution Services (QES) team.
# 
# ## Modeling Pillars
# <img src= "images/three_pillars.png">
# 
# ## Constraints and Features
# <img src= "images/apex_constraints_and_features.png">
# 
# ## The APEX Trade Lifecycle
# <img src= "images/how_apex_works.png">

# ### First, let's load a sample portfolio:
# #### Import Libs and Utils:

# In[ ]:


from qes_utils import persistXls, plotCost, plotVar, plotBuySellNet, plotGrossRemaining, plotMultiStrategyPortfolioLevelAnalytics

from gs_quant.api.gs.assets import GsAssetApi
from gs_quant.session import GsSession, Environment
from gs_quant.common import Position
from gs_quant.target.risk import OptimizationRequest, OptimizationType
from gs_quant.api.gs.risk import GsRiskApi

import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np
import copy
import datetime
from matplotlib import cm

# #### Establish GS_Quant Connection: 

# - Fill in client_id and client_secret\
# - Set up Marquee API: https://marquee.gs.com/s/developer/docs/getting-started
# - Once you create the application, click on the Application page and scroll down to the ‘Scope’ section. Request the “read_product_date”& “run_analytics” scope for your application. 

# In[ ]:


print('INFO: Setting up Marquee Connection')
client_id =
client_secret =
GsSession.use(Environment.PROD, client_id=client_id, client_secret=client_secret, scopes=['read_product_data', 'run_analytics'])

# #### Set up the portfolio: 

# In[ ]:


print('INFO: Setting up portfolio to schedule using APEX...')
portfolio_input = pd.read_csv('trade_list_world.csv').rename(columns={'Symbol': 'sedol', 'Shares': 'qty'})
portfolio_input.dtypes

# #### Convert Identifier (SEDOL) to marqueeids:

# SEDOL access needs to be requested on Marquee with the following steps:
# - Go to https://marquee.gs.com/s/developer/datasets/SEDOL 
# - Select an application to request access for 
# - Request will be auto approved

# In[ ]:


assets = GsAssetApi.get_many_assets(sedol=list(portfolio_input['sedol']), fields=['sedol', 'rank'], listed=[True], type='Single Stock')
identifier_to_marqueeid_map = pd.DataFrame([{'sedol': list(filter(lambda x: x.type=='SED', i.identifiers))[0].value, 'ID': i.id, 'rank': i.rank} for i in assets])\
                               .sort_values(['sedol', 'rank'], ascending=False).groupby('sedol').head(1)[['sedol','ID']].rename(columns={'ID': 'marqueeid'})
print(f'found {len(identifier_to_marqueeid_map)} sedols to mrquee ids map...')

# #### Identify assets with missing marquee ids and drop them from the portfolio

# In[ ]:


portfolio_input = portfolio_input.merge(identifier_to_marqueeid_map, how='left', on='sedol')

missing_marqueeids = portfolio_input[portfolio_input['marqueeid'].isnull()]
if len(missing_marqueeids):
    print(f'WARNING: the following bbids are missing marqueeids:\n{missing_marqueeids}\ndropping from the optimization...')
else: 
    print('INFO: all the assets has been succesfuly converted to marquee id')
portfolio_input = portfolio_input.dropna()
portfolio_input.head()

# ###  At this point, we have a portfolio we can optimize using APEX.
# Our portfolio is now ready for optimization with APEX.
# ###  We'll run two variations:
# #####  1. single optimization analysis - optimize the basket using defined parameters and investigate the cost-risk trade-off.
# #####  2. trade scenario analysis - run multiple optimizations upon different risk aversion (urgency) parameters and compare the cost-risk trade-off among optimized execution strategies

# ### 1. APEX Optimization: run my trade list in the APEX optimizer and explore the various analytics:
# #### in this section, we'll explore how to set optimization parameters and how to display multiple optimal trajectory analytics to develop further intuition for the decisions made by APEX
# 
# we'll run an APEX-IS (Arrival) risk-cost minimization optimal trade allocation, in the following form:
# \begin{equation*}
# Min \displaystyle \Bigg( \lambda \sum_{t=1}^T (\mbox{Risk of Residual Holdings}) + (1-\lambda) \sum_{t=1}^T (\mbox{Market Impact of Trades}) \Bigg) 
# \end{equation*}
# \begin{equation*}s.t.\end{equation*}
# \begin{equation*}Ax <= b\end{equation*}
# 
# where:
# \begin{equation*}(\mbox{Risk of Residual Holdings})\end{equation*} 
# - Incorporates the intraday and overnight expected risk, utilizing our high frequency intraday QES covariances. in other words, "every $ I decided to trade later, is running at the Risk of missing the arrival price"
# 
# \begin{equation*}(\mbox{Market Impact of Trades})\end{equation*}
# - Denote the expected market impact per asset, as a function of the physical interaction with the order book. in other words, "every $ that I will trade now, will incur some expected market impact, based on the intraday predicted evolution of spread\volume\volatility\participation rate, and other intraday calibrated parameters"
# 
# \begin{equation*}\lambda\end{equation*}
# - Risk Aversion parameter
# 
# \begin{equation*}Ax <= b\end{equation*}
# - set of linear constraints (see features available at the top of the notebook)

# #### Set up the optimization constraints
# 
# | Optimisation Parameters | Description | Value Chosen |
# | :- | :- | -: |
# | Start Time \ End Time | APEX allowed "Day1" trade horizon, in GMT* | 11pm previous day to 11pm |
# | Urgency | APEX Urgency, from VERY_LOW to VERY_HIGH | Medium |
# | Target Benchmark | Currently supports 'IS', 'CLOSE' | IS |
# | Imbalance | (Optional) setting dollar imbalance for the trade duration; "the net residual must be within +-5% of the residual gross to trade, throughout the entire trade duration" | 0.05 (5%) |
# | Participation rate | Setting volume cap for trading | 0.075 (7.5%) |
# 
# - Note that APEX allowed start end times range from 23:00 previous day to 23:00 of the query day.
#   For example, if today is the 9th of October, APEX global optimization can run from start time of 23:00 on T-1 to 23:00 on T.
# - Please also note that APEX will automatically optimize up to 5 business days, providing an optimized intraday solution with granularity of 30\60 minutes.
# - For a full set of parameters, please refer to the constraints & features image at the top, review the APEX api guide or contact [gs-qes-quant@gs.com](mailto:gs-qes-quant@gs.com)
# 

# In[ ]:


## set optimization configuration
print('INFO: Constructing Optimization Request...')
date_today = datetime.datetime.now().strftime('%Y-%m-%d')
date_yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')

apex_optimization_config =  {
                 'executionStartTime': date_yesterday + 'T23:00:00.000Z',  #execution start time
                 'executionEndTime': date_today +'T21:15:00.000Z',  # execution end time (for day 1, can run multiday if not complete on day 1)
                 'waitForResults': False,
                 'parameters': {'urgency': 'MEDIUM',  #VERY_LOW, LOW, HIGH, VERY_HIGH...
                                'targetBenchmark': 'IS',  #CLOSE
                                'imbalance': 0.05,  #Optional --> setting $ imbalance for the trade duration to never exceed +-20% of residual gross to trade
                                'participationRate': 0.075 #setting volume cap of 10%
                                },
                 }

# #### Send Optimization + Analytics request to Marquee

# In[ ]:


def sendApexRequestAndGetAnalytics(portfolio_input, apex_optimization_config):
    positions = [Position(asset_id=row.marqueeid, quantity=row.qty) for _, row in portfolio_input.iterrows()]
    print('setting up the optimization request....')
    request = OptimizationRequest(positions=positions,
                                  execution_start_time=apex_optimization_config['executionStartTime'],
                                  execution_end_time=apex_optimization_config['executionEndTime'],
                                  parameters=apex_optimization_config['parameters'],
                                  **{'type': OptimizationType.APEX})
    print('Sending the request to the marquee service...')
    opt = GsRiskApi.create_pretrade_execution_optimization(request)
    analytics_results = GsRiskApi.get_pretrade_execution_optimization(opt.get('optimizationId'))
    print ('COMPLETE!')
    return analytics_results

results_dict = sendApexRequestAndGetAnalytics(portfolio_input, apex_optimization_config)

# In[ ]:


print('INFO: High Level Cost estimation and % expected Completion:')
pd.DataFrame(results_dict['analytics']['portfolioAnalyticsDaily']).set_index('tradeDayNumber')

# In[ ]:


print('missing assets:')
pd.DataFrame(results_dict['analytics']['assetsExcluded'])

# #### Actual Optimization Parameters Used in APEX
# - Although a set of optimization parameters was specified above, APEX might conclude that the parameters joined feasible space does not exist (infeasible set).
# - APEX can then choose to soften/drop/relax the constraints in a hierarchical fashion. 

# In[ ]:


constraints_hierarchy = pd.DataFrame(results_dict['analytics']['constraintsConsultations'])['constraints']
pd.concat([pd.DataFrame(constraints_hierarchy.values[i]).assign(iteration=i) for i in constraints_hierarchy.index]).set_index(['iteration', 'name'])['status'].unstack().T

# #### What kind of Analytics provided by APEX ?
# ##### APEX provide a vast set of numbers that helps understanding unravel the decision made by the optimizer:

# In[ ]:


results_dict['analytics'].keys()

# #### Visualise Your Optimisation Results

# In[ ]:


analytics_result_analytics = results_dict['analytics'] 
intraday = pd.DataFrame(analytics_result_analytics['portfolioAnalyticsIntraday'])
intraday_to_plot = intraday.assign(time = lambda x: pd.to_datetime(x['time'])).set_index('time')

# #### Four examples of visualizing your intraday analysis throughout trade date
# - Gross Remaining
# - Buy/Sell/Net
# - Cost Contribution
# - Risk Contribution 

# In[ ]:


intraday_to_plot.head(5).append(intraday_to_plot.tail(5))

# In[ ]:


plotGrossRemaining(intraday_to_plot)
plotBuySellNet(intraday_to_plot)
plotCost(intraday_to_plot)
plotVar(intraday_to_plot)

# ###### Sources: Goldman Sachs, Bloomberg, Reuters, Axioma

# ##### The creativity around various analytics are endless, here are couple of examples, derived from the various analytics dataframes we use for our APEX clients:
# 
# <img src= "images/apex_analytics_examples.png">

# ###### Sources: Goldman Sachs, Bloomberg, Reuters, Axioma

# ##### save all results to excel for further exploration:

# In[ ]:


xls_path = persistXls(xls_report=results_dict['analytics'],
                                  path='',
                                  filename='apex_optimization_detailed_analytics',
                                  indentifier_marqueeid_map=portfolio_input[
                                      [identifier_type, 'marqueeid']])
print('saving all analytics frames to {0}...'.format(xls_path))

# <img src= "images/apex_excel_example.png">

# ### 2. APEX Optimization - Trade Scenario Analysis: run my trade list in the APEX optimizer across multiple risk aversions\urgency parameters to assess ideal parameters set.

# #### Define a function for running multiple optimizations, keeping all constrains intact and change urgency only:

# In[ ]:


def optimisationMulti(portfolio_input, apex_optimization_config, urgency_list = ['VERY_LOW', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']):
    results_dict_multi = {}
    apex_optimization_config_temp = copy.deepcopy(apex_optimization_config)
    
    for u in urgency_list:
        apex_optimization_config_temp['parameters']['urgency'] = u
        apex_optimization_config_temp['parameters']['imbalance'] = .3
        apex_optimization_config_temp['parameters']['participationRate'] = .5
        
        print('INFO Running urgency={0} optimization....'.format(u))
        results_dict_multi[u] = sendApexRequestAndGetAnalytics(portfolio_input, apex_optimization_config_temp)
        
        print('INFO: High Level Cost estimation and % expected Completion:\n{0}'\
              .format(pd.DataFrame(results_dict_multi[u]['analytics']['portfolioAnalyticsDaily'])))
    
    return results_dict_multi

# ##### Run Optimization Across Urgencies

# In[ ]:


urgency_list = ['VERY_LOW', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']
results_dict_multi = optimisationMulti(portfolio_input = portfolio_input,\
                                       apex_optimization_config = apex_optimization_config,\
                                       urgency_list=urgency_list)

# #### Compare Results from Different Urgencies on Day 1:

# In[ ]:


ordering = ['grey', 'sky_blue', 'black', 'cyan', 'light_blue', 'dark_green']
urgency_list = ['VERY_LOW', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']
ptAnalyticsDaily_list = []
for u in urgency_list:
    ptAnalyticsDaily_list.append(pd.DataFrame(results_dict_multi[u]['analytics']['portfolioAnalyticsDaily']).iloc[[0]].assign(urgency=u) )
pd.concat(ptAnalyticsDaily_list).set_index('urgency')

# #### Visualise Optimization Results
# - Plotting 'Trade_cum_sum, Total Cost, Total Risk' against time for the chosen urgencies
# - Trade_cum_sum: Cumulative sum of the intraday trades

# In[ ]:


metrics_list = ['tradePercentageCumulativeSum', 'totalRiskBps', 'totalCost', 'advAveragePercentage']
title = ['Intraday Trade', 'Risk', 'Cost', 'Participation Rate']
ylabel = ['Trade Cum Sum %', 'Risk(bps) ', 'Cost(bps)', 'Prate(%)']

plotMultiStrategyPortfolioLevelAnalytics(results_dict_multi, metrics_list, title, ylabel)

# ###### Sources: Goldman Sachs, Bloomberg, Reuters, Axioma

# #### Plot the optimal Efficient Frontier - the expected Market Impact vs. Residual Risk Trade-off:

# In[ ]:


initial_gross = pd.DataFrame(results_dict_multi['VERY_LOW']['analytics']['portfolioAnalyticsIntraday'])['gross'].iloc[0]
risk_cost_tradeoff = pd.concat( [\
    pd.DataFrame(results_dict_multi[urgency]['analytics']['portfolioAnalyticsDaily'])\
    [['estimatedCostBps', 'meanExpectedCostVersusBenchmark']]\
.assign(totalRiskBps = lambda x: x['estimatedCostBps'] - x['meanExpectedCostVersusBenchmark'])\
.iloc[0].rename(urgency).to_frame()
for urgency in ['VERY_LOW', 'LOW', 'MEDIUM']], axis=1).T

cmap = cm.get_cmap('Set1')
ax = risk_cost_tradeoff.plot.scatter(x='totalRiskBps', y='meanExpectedCostVersusBenchmark',\
                                     title='The Example Basket Efficient Frontier',\
                                     colormap=cmap, c=range(len(risk_cost_tradeoff)), s=100)

for k, v in risk_cost_tradeoff[['totalRiskBps', 'meanExpectedCostVersusBenchmark']].iterrows():
    ax.annotate(k, v,
                xytext=(10,-5), textcoords='offset points',
                family='sans-serif', fontsize=10, color='darkslategrey')

ax.plot(risk_cost_tradeoff['totalRiskBps'].values, risk_cost_tradeoff['meanExpectedCostVersusBenchmark'].values,
       color='grey', alpha=.5)

# ###### Sources: Goldman Sachs, Bloomberg, Reuters, Axioma

# # And That's IT! Find below an holistic view of our APEX platform in visual from:
# <img src= "images/apex_box.png">

# ##### Disclaimers:
# ###### Indicative Terms/Pricing Levels: This material may contain indicative terms only, including but not limited to pricing levels. There is no representation that any transaction can or could have been effected at such terms or prices. Proposed terms and conditions are for discussion purposes only. Finalized terms and conditions are subject to further discussion and negotiation.
# ###### www.goldmansachs.com/disclaimer/sales-and-trading-invest-rec-disclosures.html If you are not accessing this material via Marquee ContentStream, a list of the author's investment recommendations disseminated during the preceding 12 months and the proportion of the author's recommendations that are 'buy', 'hold', 'sell' or other over the previous 12 months is available by logging into Marquee ContentStream using the link below. Alternatively, if you do not have access to Marquee ContentStream, please contact your usual GS representative who will be able to provide this information to you.
# ###### Please refer to https://marquee.gs.com/studio/ for price information of corporate equity securities.
# ###### Notice to Australian Investors: When this document is disseminated in Australia by Goldman Sachs & Co. LLC ("GSCO"), Goldman Sachs International ("GSI"), Goldman Sachs Bank Europe SE ("GSBE"), Goldman Sachs (Asia) L.L.C. ("GSALLC"), or Goldman Sachs (Singapore) Pte ("GSSP") (collectively the "GS entities"), this document, and any access to it, is intended only for a person that has first satisfied the GS entities that: 
# ###### • the person is a Sophisticated or Professional Investor for the purposes of section 708 of the Corporations Act of Australia; and 
# ###### • the person is a wholesale client for the purpose of section 761G of the Corporations Act of Australia. 
# ###### To the extent that the GS entities are providing a financial service in Australia, the GS entities are each exempt from the requirement to hold an Australian financial services licence for the financial services they provide in Australia. Each of the GS entities are regulated by a foreign regulator under foreign laws which differ from Australian laws, specifically: 
# ###### • GSCO is regulated by the US Securities and Exchange Commission under US laws;
# ###### • GSI is authorised by the Prudential Regulation Authority and regulated by the Financial Conduct Authority and the Prudential Regulation Authority, under UK laws;
# ###### • GSBE is subject to direct prudential supervision by the European Central Bank and in other respects is supervised by the German Federal Financial Supervisory Authority (Bundesanstalt für Finanzdienstleistungsaufischt, BaFin) and Deutsche Bundesbank;
# ###### • GSALLC is regulated by the Hong Kong Securities and Futures Commission under Hong Kong laws; and
# ###### • GSSP is regulated by the Monetary Authority of Singapore under Singapore laws.
# ###### Notice to Brazilian Investors
# ###### Marquee is not meant for the general public in Brazil. The services or products provided by or through Marquee, at any time, may not be offered or sold to the general public in Brazil. You have received a password granting access to Marquee exclusively due to your existing relationship with a GS business located in Brazil. The selection and engagement with any of the offered services or products through Marquee, at any time, will be carried out directly by you. Before acting to implement any chosen service or products, provided by or through Marquee you should consider, at your sole discretion, whether it is suitable for your particular circumstances and, if necessary, seek professional advice. Any steps necessary in order to implement the chosen service or product, including but not limited to remittance of funds, shall be carried out at your discretion. Accordingly, such services and products have not been and will not be publicly issued, placed, distributed, offered or negotiated in the Brazilian capital markets and, as a result, they have not been and will not be registered with the Brazilian Securities and Exchange Commission (Comissão de Valores Mobiliários), nor have they been submitted to the foregoing agency for approval. Documents relating to such services or products, as well as the information contained therein, may not be supplied to the general public in Brazil, as the offering of such services or products is not a public offering in Brazil, nor used in connection with any offer for subscription or sale of securities to the general public in Brazil.
# ###### The offer of any securities mentioned in this message may not be made to the general public in Brazil. Accordingly, any such securities have not been nor will they be registered with the Brazilian Securities and Exchange Commission (Comissão de Valores Mobiliários) nor has any offer been submitted to the foregoing agency for approval. Documents relating to the offer, as well as the information contained therein, may not be supplied to the public in Brazil, as the offer is not a public offering of securities in Brazil. These terms will apply on every access to Marquee.
# ###### Ouvidoria Goldman Sachs Brasil: 0800 727 5764 e/ou ouvidoriagoldmansachs@gs.com
# ###### Horário de funcionamento: segunda-feira à sexta-feira (exceto feriados), das 9hs às 18hs.
# ###### Ombudsman Goldman Sachs Brazil: 0800 727 5764 and / or ouvidoriagoldmansachs@gs.com
# ###### Available Weekdays (except holidays), from 9 am to 6 pm.
#  
# ###### Note to Investors in Israel: GS is not licensed to provide investment advice or investment management services under Israeli law.
# ###### Notice to Investors in Japan
# ###### Marquee is made available in Japan by Goldman Sachs Japan Co., Ltd.
# 
# ###### 本書は情報の提供を目的としております。また、売却・購入が違法となるような法域での有価証券その他の売却若しくは購入を勧めるものでもありません。ゴールドマン・サックスは本書内の取引又はストラクチャーの勧誘を行うものではございません。これらの取引又はストラクチャーは、社内及び法規制等の承認等次第で実際にはご提供できない場合がございます。
# 
# ###### <適格機関投資家限定　転売制限>
# ###### ゴールドマン・サックス証券株式会社が適格機関投資家のみを相手方として取得申込みの勧誘（取得勧誘）又は売付けの申込み若しくは買付けの申込みの勧誘(売付け勧誘等)を行う本有価証券には、適格機関投資家に譲渡する場合以外の譲渡が禁止される旨の制限が付されています。本有価証券は金融商品取引法第４条に基づく財務局に対する届出が行われておりません。なお、本告知はお客様によるご同意のもとに、電磁的に交付させていただいております。
# ###### ＜適格機関投資家用資料＞ 
# ###### 本資料は、適格機関投資家のお客さまのみを対象に作成されたものです。本資料における金融商品は適格機関投資家のお客さまのみがお取引可能であり、適格機関投資家以外のお客さまからのご注文等はお受けできませんので、ご注意ください。 商号等/ゴールドマン・サックス証券株式会社 金融商品取引業者　関東財務局長（金商）第６９号 
# ###### 加入協会/　日本証券業協会、一般社団法人金融先物取引業協会、一般社団法人第二種金融商品取引業協会 
# ###### 本書又はその添付資料に信用格付が記載されている場合、日本格付研究所（JCR）及び格付投資情報センター（R&I）による格付は、登録信用格付業者による格付（登録格付）です。その他の格付は登録格付である旨の記載がない場合は、無登録格付です。無登録格付を投資判断に利用する前に、「無登録格付に関する説明書」（http://www.goldmansachs.com/disclaimer/ratings.html）を十分にお読みください。 
# ###### If any credit ratings are contained in this material or any attachments, those that have been issued by Japan Credit Rating Agency, Ltd. (JCR) or Rating and Investment Information, Inc. (R&I) are credit ratings that have been issued by a credit rating agency registered in Japan (registered credit ratings). Other credit ratings are unregistered unless denoted as being registered. Before using unregistered credit ratings to make investment decisions, please carefully read "Explanation Regarding Unregistered Credit Ratings" (http://www.goldmansachs.com/disclaimer/ratings.html).
# ###### Notice to Mexican Investors: Information contained herein is not meant for the general public in Mexico. The services or products provided by or through Goldman Sachs Mexico, Casa de Bolsa, S.A. de C.V. (GS Mexico) may not be offered or sold to the general public in Mexico. You have received information herein exclusively due to your existing relationship with a GS Mexico or any other Goldman Sachs business. The selection and engagement with any of the offered services or products through GS Mexico will be carried out directly by you at your own risk. Before acting to implement any chosen service or product provided by or through GS Mexico you should consider, at your sole discretion, whether it is suitable for your particular circumstances and, if necessary, seek professional advice. Information contained herein related to GS Mexico services or products, as well as any other information, shall not be considered as a product coming from research, nor it contains any recommendation to invest, not to invest, hold or sell any security and may not be supplied to the general public in Mexico.
# ###### Notice to New Zealand Investors: When this document is disseminated in New Zealand by Goldman Sachs & Co. LLC ("GSCO") , Goldman Sachs International ("GSI"), Goldman Sachs Bank Europe SE ("GSBE"), Goldman Sachs (Asia) L.L.C. ("GSALLC") or Goldman Sachs (Singapore) Pte ("GSSP") (collectively the "GS entities"), this document, and any access to it, is intended only for a person that has first satisfied; the GS entities that the person is someone: 
# ###### (i) who is an investment business within the meaning of clause 37 of Schedule 1 of the Financial Markets Conduct Act 2013 (New Zealand) (the "FMC Act");
# ###### (ii) who meets the investment activity criteria specified in clause 38 of Schedule 1 of the FMC Act;
# ###### (iii) who is large within the meaning of clause 39 of Schedule 1 of the FMC Act; or
# ###### (iv) is a government agency within the meaning of clause 40 of Schedule 1 of the FMC Act. 
# ###### No offer to acquire the interests is being made to you in this document. Any offer will only be made in circumstances where disclosure is not required under the Financial Markets Conducts Act 2013 or the Financial Markets Conduct Regulations 2014.
# ###### Notice to Swiss Investors: This is marketing material for financial instruments or services. The information contained in this material is for general informational purposes only and does not constitute an offer, solicitation, invitation or recommendation to buy or sell any financial instruments or to provide any investment advice or service of any kind.
# ###### THE INFORMATION CONTAINED IN THIS DOCUMENT DOES NOT CONSITUTE, AND IS NOT INTENDED TO CONSTITUTE, A PUBLIC OFFER OF SECURITIES IN THE UNITED ARAB EMIRATES IN ACCORDANCE WITH THE COMMERCIAL COMPANIES LAW (FEDERAL LAW NO. 2 OF 2015), ESCA BOARD OF DIRECTORS' DECISION NO. (9/R.M.) OF 2016, ESCA CHAIRMAN DECISION NO 3/R.M. OF 2017 CONCERNING PROMOTING AND INTRODUCING REGULATIONS OR OTHERWISE UNDER THE LAWS OF THE UNITED ARAB EMIRATES. ACCORDINGLY, THE INTERESTS IN THE SECURITIES MAY NOT BE OFFERED TO THE PUBLIC IN THE UAE (INCLUDING THE DUBAI INTERNATIONAL FINANCIAL CENTRE AND THE ABU DHABI GLOBAL MARKET). THIS DOCUMENT HAS NOT BEEN APPROVED BY, OR FILED WITH THE CENTRAL BANK OF THE UNITED ARAB EMIRATES, THE SECURITIES AND COMMODITIES AUTHORITY, THE DUBAI FINANCIAL SERVICES AUTHORITY, THE FINANCIAL SERVICES REGULATORY AUTHORITY OR ANY OTHER RELEVANT LICENSING AUTHORITIES IN THE UNITED ARAB EMIRATES. IF YOU DO NOT UNDERSTAND THE CONTENTS OF THIS DOCUMENT, YOU SHOULD CONSULT WITH A FINANCIAL ADVISOR. THIS DOCUMENT IS PROVIDED TO THE RECIPIENT ONLY AND SHOULD NOT BE PROVIDED TO OR RELIED ON BY ANY OTHER PERSON.
