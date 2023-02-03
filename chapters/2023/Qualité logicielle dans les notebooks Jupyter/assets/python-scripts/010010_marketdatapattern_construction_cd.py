#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# The MarketDataPattern class allows the user to select a collection of observable market data
# MarketDataPattern arguments include: mkt_type, mkt_asset, mkt_class, mkt_point and mkt_quoting_style
from gs_quant.risk import MarketDataPattern

# In[ ]:


# A MarketDataPattern defined with 'CD' mkt_type contains market data related to credit instruments
# (such as CDS, Index Spreads, Recovery Rates, etc.)

cd = MarketDataPattern(mkt_type='CD')

# In[ ]:


# The collection of 'CD' market data can be further constrained by setting the 'mkt_class'

# CDS Index Products are selected by the 'INDEX' mkt_class
index = MarketDataPattern(mkt_type ='CD', mkt_class='INDEX')
# The mkt_asset isolates the series and mt_point indicates maturity
index_series = MarketDataPattern(mkt_type ='CD', mkt_asset = 'ITRAXX EUROPE`36', mkt_class='INDEX', mkt_point='5Y')

# CDS are selected by the 'CDS' mkt_class and mkt_asset isolates the specific cds curve
cds = MarketDataPattern(mkt_type ='CD', mkt_class='CDS')

