#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.markets.indices_utils import *
from gs_quant.session import Environment, GsSession

# In[ ]:


client = 'CLIENT ID'
secret = 'CLIENT SECRET'

GsSession.use(Environment.PROD, client_id=client, client_secret=secret, scopes=('read_product_data',))

# You may choose any combination of the following styles:
# 
# #### Custom Basket Styles
# * **Ad Hoc Desk Work:** CustomBasketStyles.*AD_HOC_DESK_WORK*
# * **Client Constructed/Wrapper:** CustomBasketStyles.*CLIENT_CONSTRUCTED_WRAPPER*
# * **Consumer:** CustomBasketStyles.*CONSUMER*
# * **Energy:** CustomBasketStyles.*ENERGY*
# * **Enhanced Index Solutions:** CustomBasketStyles.*ENHANCED_INDEX_SOLUTIONS*
# * **ESG:** CustomBasketStyles.*ESG*
# * **Factors:** CustomBasketStyles.*FACTORS*
# * **Financials:** CustomBasketStyles.*FINANCIALS*
# * **Flagship:** CustomBasketStyles.*FLAGSHIP*
# * **Geographic:** CustomBasketStyles.*GEOGRAPHIC*
# * **Growth:** CustomBasketStyles.*GROWTH*
# * **Health Care:** CustomBasketStyles.*HEALTHCARE*
# * **Hedging:** CustomBasketStyles.*HEDGING*
# * **Industrials:** CustomBasketStyles.*INDUSTRIALS*
# * **Materials:** CustomBasketStyles.*MATERIALS*
# * **Momentum:** CustomBasketStyles.*MOMENTUM*
# * **PIPG:** CustomBasketStyles.*PIPG*
# * **Sectors/Industries:** CustomBasketStyles.*SECTORS_INDUSTRIES*
# * **Size:** CustomBasketStyles.*SIZE*
# * **Structured One Delta:** CustomBasketStyles.*STRUCTURED_ONE_DELTA*
# * **Thematic:** CustomBasketStyles.*THEMATIC*
# * **TMT:** CustomBasketStyles.*TMT*
# * **Utilities:** CustomBasketStyles.*UTILITIES*
# * **Value:** CustomBasketStyles.*VALUE*
# * **Volatility:** CustomBasketStyles.*VOLATILITY*
# 
# #### Research Basket Styles
# * **Asia ex-Japan:** ResearchBasketStyles.*ASIA_EX_JAPAN*
# * **Equity Thematic:** ResearchBasketStyles.*EQUITY_THEMATIC*
# * **Europe:** ResearchBasketStyles.*EUROPE*
# * **Fund Ownership:** ResearchBasketStyles.*FUND_OWNERSHIP*
# * **Fundamentals:** ResearchBasketStyles.*FUNDAMENTALS*
# * **FX/Oil:** ResearchBasketStyles.*FX_OIL*
# * **Geographical Exposure:** ResearchBasketStyles.*GEOGRAPHICAL_EXPOSURE*
# * **Hedge Fund:** ResearchBasketStyles.*HEDGE_FUND*
# * **Investment Profile (IP) Factors:** ResearchBasketStyles.*IP_FACTORS*
# * **Japan:** ResearchBasketStyles.*JAPAN*
# * **Macro:** ResearchBasketStyles.*MACRO*
# * **Macro Slice/Styles:** ResearchBasketStyles.*MACRO_SLICE_STYLES*
# * **Mutual Fund:** ResearchBasketStyles.*MUTUAL_FUND*
# * **Positioning:** ResearchBasketStyles.*POSITIONING*
# * **Portfolio Strategy:** ResearchBasketStyles.*PORTFOLIO_STRATEGY*
# * **Risk & Liquidity:** ResearchBasketStyles.*RISK_AND_LIQUIDITY*
# * **Sector:** ResearchBasketStyles.*SECTOR*
# * **Shareholder Return:** ResearchBasketStyles.*SHAREHOLDER_RETURN*
# * **Style, Factor and Fundamental:** ResearchBasketStyles.*STYLE_FACTOR_AND_FUNDAMENTAL*
# * **Style/Themes:** ResearchBasketStyles.*STYLES_THEMES*
# * **Tactical Research:** ResearchBasketStyles.*TACTICAL_RESEARCH*
# * **Thematic:** ResearchBasketStyles.*THEMATIC*
# * **US:** ResearchBasketStyles.*US*
# * **Wavefront Components:** ResearchBasketStyles.*WAVEFRONT_COMPONENTS*
# * **Wavefront Pairs:** ResearchBasketStyles.*WAVEFRONT_PAIRS*
# * **Wavefronts:** ResearchBasketStyles.*WAVEFRONTS*
# 
# These options will work with any of the following functions:

# In[ ]:


get_flagship_baskets(styles=[CustomBasketStyles.FACTORS, ResearchBasketStyles.WAVEFRONTS])

get_flagships_with_assets(identifiers=['AAPL UW'], region=[Region.AMERICAS], styles=[CustomBasketStyles.FACTORS, ResearchBasketStyles.WAVEFRONTS])

get_flagships_performance(region=[Region.EUROPE, Region.GLOBAL], styles=[CustomBasketStyles.FACTORS, ResearchBasketStyles.WAVEFRONTS])

get_flagships_constituents(region=[Region.EM], styles=[CustomBasketStyles.FACTORS, ResearchBasketStyles.WAVEFRONTS])
