#!/usr/bin/env python
# coding: utf-8

# In[5]:


from gs_quant.common import PayReceive, Currency
from gs_quant.instrument import IRSwaption
from gs_quant.markets.portfolio import Portfolio
from gs_quant.session import Environment, GsSession

# In[8]:


from gs_quant.session import GsSession
from gs_quant.markets import PricingContext,LiveMarket
from gs_quant.datetime import date_range
from gs_quant.markets import PricingContext,LiveMarket
from gs_quant.markets.portfolio import Portfolio
from gs_quant.risk.results import PricingFuture, PortfolioRiskResult
from gs_quant.target.common import MapParameter
from gs_quant_internal.boltweb import valuation
import datetime as dt

from gs_quant_analytics_internal.components.fx.vanilla_contour_pricer import GenericPayoffContourPricer,VanillaPayoffContourPricer
from gs_quant_analytics_internal.components.fx.exotic_contour_pricer import ExoticPayoffContourPricer
from gs_quant_analytics_internal.components.xasset.payoff_contour_pricer import PayoffContourPricerParams
from gs_quant_analytics_internal.components.xasset.payoff_contours import PayoffContours
from gs_quant.session import GsSession,Environment
from gs_quant.common import AssetClass
from gs_quant_analytics_internal.xasset.utils import builder_to_priceable_asset, quick_entry_to_portfolio
GsSession.use()
live_pricing = PricingContext(market=LiveMarket('LDN'),market_data_location='LDN',)
opt = tdapi.FXOptionBuilder(
    over="JPY",
    under='USD',
    expiry='2021-12-29',
    size='5M',
    strike='113.465',
    hedgetype='spot')
with live_pricing:
    spot=opt.calc(valuation('FXSpot'))
    fwd=opt.calc(valuation('FXFwd'))
    pricepct = opt.calc(valuation('PricePct'))
    volatility = opt.calc(valuation('FXVol'))
    winghedge = opt.calc(valuation('FXCalcHedge'))

opt.valuation_overrides = {
        'FXSpot':spot.result(), 
        'FXFwd':fwd.result(),
        'FXVol':volatility.result(),
        'PricePct':pricepct.result()}
opt.hedge = winghedge.result()
with live_pricing:
    opt.resolve()
    new_portfolio = Portfolio(opt)
    new_portfolio.save_as_quote()
print(new_portfolio.id) 

# In[2]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[3]:


swaption1 = IRSwaption(PayReceive.Pay, '5y', Currency.EUR, expiration_date='3m', name='EUR-3m5y')
swaption2 = IRSwaption(PayReceive.Pay, '7y', Currency.EUR, expiration_date='6m', name='EUR-6m7y')

# In[4]:


portfolio = Portfolio((swaption1, swaption2))
