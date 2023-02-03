#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.session import GsSession, Environment
from gs_quant.instrument import IRSwap
from gs_quant.risk import IRFwdRate, RollFwd
from gs_quant.markets.portfolio import Portfolio
from gs_quant.markets import PricingContext
from datetime import datetime
import matplotlib.pylab as plt
import pandas as pd
import numpy as np

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics', ))

# In[ ]:


ccy = 'EUR'
# construct a series of 6m FRAs going out 20y or so
fras = Portfolio([IRSwap('Pay', '6m', ccy, effective_date='{}m'.format(i-6), 
                         fixed_rate_frequency='6m', floating_rate_frequency='6m') for i in range(6, 123, 6)])
fras.resolve()
results = fras.calc(IRFwdRate)

# get the fwd rates for these fras under the base sceneraio (no shift in time)
base = {fras[i].termination_date: res for i, res in enumerate(results)}
base_series = pd.Series(base, name='base', dtype=np.dtype(float))

# In[ ]:


# calculate the fwd rates with a shift forward of 6m.  This shift keeps fwd rates constant.  
# So 5.5y rate today will be 5y rate under the scenario of pricing 6m in the future.
with RollFwd(date='6m', realise_fwd=True, holiday_calendar='LDN'):
    results_fwd = fras.calc(IRFwdRate)
with RollFwd(date='6m', realise_fwd=False, holiday_calendar='LDN'):
    results_spot = fras.calc(IRFwdRate)

# In[ ]:


roll_to_fwds = {fras[i].termination_date: res for i, res in enumerate(results_fwd)}
roll_to_fwds_series = pd.Series(roll_to_fwds, name='roll to fwd', dtype=np.dtype(float))

roll_to_spot = {fras[i].termination_date: res for i, res in enumerate(results_spot)}
roll_to_spot_series = pd.Series(roll_to_spot, name='roll to spot', dtype=np.dtype(float))

# In[ ]:


# show the curves, the base in blue, the roll to fwd in green and the roll to spot in orange.
# note blue and green curves are not exactly on top of each other as we aren't using the curve instruments themselves
# but instead using FRAs to show a smooth curve.
base_series.plot(figsize=(20, 10))
roll_to_spot_series.plot()
roll_to_fwds_series.plot()
plt.legend()
