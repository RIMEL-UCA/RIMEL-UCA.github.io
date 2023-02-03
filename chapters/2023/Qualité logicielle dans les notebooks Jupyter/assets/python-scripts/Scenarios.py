#!/usr/bin/env python
# coding: utf-8

# # Scenario Context
# Scenario contexts enable the user to price and calculate risk under varying market states and pricing environments.

# In[ ]:


from gs_quant.instrument import IRSwaption
from gs_quant.risk import MarketDataPattern, MarketDataShock, MarketDataShockType, MarketDataShockBasedScenario, \
    RollFwd, CurveScenario, IndexCurveShift
from gs_quant.session import Environment, GsSession

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


# Create and price a swaption
swaption = IRSwaption('Pay', '5y', 'USD', expiration_date='3m')
base_price = swaption.price()
base_price

# ### MarketDataShockBasedScenario
# Allows the user to create a bespoke market data shock

# In[ ]:


MarketDataShockBasedScenario?

# In[ ]:


# Scenario Creation: Shock all points on the Vol Curve by 1bp
ir_vol_scenario = MarketDataShockBasedScenario(
    shocks={
        MarketDataPattern('IR Vol'): MarketDataShock(MarketDataShockType.Absolute, 1e-4)
    }
)

# In[ ]:


# Price swaption under scenario
with ir_vol_scenario:
    scenario_price = swaption.price()

scenario_price

# In[ ]:


# Swaption Price Comparison: Base vs. Shocked Vol Curves
diff = scenario_price - base_price
diff

# In[ ]:


# Comparing Parallel Bump Scenario w/ Vega
from gs_quant.risk import IRVegaParallel
vega = swaption.calc(IRVegaParallel)
vega

# ### Curve Scenario 
# A predefined scenario used to modify the shape of the curve with bespoke transformations - by applying both parallel and slope shifts.  
# * market_data_pattern - Market pattern for matching curve assets <i>(required parameter)</i>
# * parallel_shift - A constant (X bps) which shifts all points by the same amount
# * curve_shift - A double which represents the net rate change (X bps) between tenorStart and tenorEnd
# * pivot_point – The tenor in years (float) at which there is zero rate change, which is between tenor_start and tenor_end inclusive, informing the type of curve shift. If not specified, pivot_point is the midpoint of tenor_start and tenor_end
# * tenor_start – The tenor, in years, (float) which is the starting point of the curve shift <i>(required parameter is curve_shift is specified)</i>
# * tenor_end – The tenor, in years, (float) which is the end point of the curve shift <i>(required parameter is curve_shift is specified)</i>

# In[ ]:


CurveScenario?

# In[ ]:


# Scenario Creation: modify the Vol Curve by a 5bp parallel shift, 1bp slope shift pivoted at 5y point (up to 50y)
curve_scenario = CurveScenario(market_data_pattern=MarketDataPattern('IR', 'USD'), parallel_shift=5,
                               curve_shift=1, pivot_point=5, tenor_end=50, tenor_start=0)

with curve_scenario:
    swaption_scenario_price = swaption.price()

# Look at the difference between scenario and base prices
print('Base price:     {:,.2f}'.format(base_price))
print('Scenario price: {:,.2f}'.format(swaption_scenario_price))

# ### RollFwd Scenario
# A predefined scenario used to evolve market data and trades over a period of time
# * date - Absolute or Relative Date to shift markets to
# * realise_fwd - Roll along the forward curve or roll in spot space
# * holiday_calendar - Calendar to use if relative date is specified in the date parameter

# In[ ]:


RollFwd?

# In[ ]:


# RollFwd Scenario - Roll forward 1 month
base_price = swaption.price()
with RollFwd(date='1m', holiday_calendar='NYC', realise_fwd=False):
    fwd_price = swaption.price()

print('Base price:     {:,.2f}'.format(base_price))
print('Scenario price: {:,.2f}'.format(fwd_price))
print('Diff: {:,.2f}'.format(fwd_price - base_price))

# ### Index Curve Shift Scenario 
# A predefined scenario used to modify the shape of the index curve. This allows the user to easily shock the curve including parallel shift and slope shift. Users can even specify which part of the curve for parallel shift through custom bucket. 
# * market_data_pattern - Market pattern for matching curve assets
# * rate_option - Rate option of the index curve.
# * tenor - Tenor of the index curve.
# * floor - Floor size in bps of the index curve.
# * annualised_parallel_shift – Parallel shift in bps.
# * annualised_slope_shift – Annual slope shift in bps with pivot = 0.
# * cutoff – Cutoff time in years to stop applying the shift.
# * bucket_start - Start date for the custom bucket.
# * bucket_end - End date for the custom bucket.
# * bucket_shift - Bucket shift in bps for the custom bucket.

# In[ ]:


IndexCurveShift?

# In[ ]:


# Scenario Creation: modify the index curve of this swaption by a 1bp parallel shift, 1bp slope shift, pivoted default to 0 year with floor value -1bp (up to 50y)
index_curve_shift = IndexCurveShift(rate_option="USD-LIBOR-BBA", tenor="3m", annualised_parallel_shift=1,
                               annualised_slope_shift=1, floor=-1, cutoff=50)

with index_curve_shift:
    swaption_scenario_price = swaption.price()

# Look at the difference between scenario and base prices
print('Base price:     {:,.2f}'.format(base_price))
print('Scenario price: {:,.2f}'.format(swaption_scenario_price))
