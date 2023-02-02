#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.markets.portfolio import Portfolio
from gs_quant.instrument import IRSwaption
from gs_quant.risk import IRVega, IRDelta, Price
from gs_quant.markets import HistoricalPricingContext, PricingContext
import datetime as dt
import pandas as pd

pd.options.display.float_format = '{:,.0f}'.format

# In[ ]:


# instantiate session if not running in jupyter hub
from gs_quant.session import GsSession
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics',)) 

# ### Extract instruments 

# In[ ]:


with PricingContext(pricing_date=dt.date(2019, 4, 24)):
    trade1=IRSwaption(pay_or_receive='Pay', notional_currency='EUR', expiration_date='6m', termination_date='3y', strike='ATMF', name='payer')
    trade2=IRSwaption(pay_or_receive='Receive', notional_currency='EUR', expiration_date='6m', termination_date='3y', strike='ATMF+50', name='receiver')
    trade2.resolve()
portfolio = Portfolio((trade1, trade2))

# In[ ]:


# We can reference the instruments in the portfolio in different ways 
print(portfolio[1]) # by index
print(portfolio[trade2])  # by instrument object - - note instrument object changes once you resolve it
print(portfolio['receiver'])  # by instrument name

# ### Extract risk results

# In[ ]:


with PricingContext(pricing_date=dt.date(2019, 4, 24)):
    results = portfolio.calc((IRVega, IRDelta, Price))
type(results)

# In[ ]:


# we can extract all the risks for a single trade and it will return a dictionary of results keyed by the risk type
# note that the risks can come back as a number of formats depending on what the risk is.  So it could come back as a
# scalar float or as a panda series or as a panda dataframe
results[0]

# In[ ]:


# we can extract a single risk for a single trade in a number of ways
print('{:,.0f}'.format(results[Price]['receiver'])) # by risk name and trade name
print('{:,.0f}'.format(results[Price][1])) # by risk name and index of trade in portfolio
print('{:,.0f}'.format(results[Price][trade2])) # by risk name and trade object

# order doesn't matter when extracting results from a PorfolioRiskResult
print('{:,.0f}'.format(results['receiver'][Price]))

# In[ ]:


# we can aggregate risks
a = results[Price]['payer']
b = results[Price]['receiver']
print('{:,.0f}'.format(a + b))
print('{:,.0f}'.format(results[Price].aggregate()))

# In[ ]:


# we could return the results as a dataframe
pd.DataFrame(results[Price])

# In[ ]:


# if the risk is not scalar as in the case of IRDelta we can use a similar pattern but the result will be a dataframe
results[IRDelta]['payer']

# In[ ]:


# if we wanted to see the total IRDelta for this trade we could do
results[IRDelta]['payer'].value.sum()

# In[ ]:


# we can aggregate dataframe results
print('{:,.0f}'.format(results[IRDelta][0].value[1]))
print('{:,.0f}'.format(results[IRDelta][1].value[1]))
results[IRDelta].aggregate()

# In[ ]:


# when we price with an HistoricalPricingContext we introduce a new dimension to our results

with HistoricalPricingContext(dt.date(2019, 4, 24), dt.date(2019, 5, 24)):
    hist_results = portfolio.calc((IRVega, IRDelta, Price))

type(hist_results)

# In[ ]:


# now if we ask for the price of a single trade instead of a float we will receieve a panda series of floats indexed by date
hist_results[Price][trade1]

# In[ ]:


# we can still aggregate in the same way and the series will be aggregated by date
hist_results[Price].aggregate()

# In[ ]:


# to get a single price for a specific date we can do
hist_results[Price].aggregate().at[dt.date(2019, 4, 30)]

# In[ ]:


# if we return a dataframe then the result will be concatinated with an index of the date.
hist_results[IRDelta][0]

# In[ ]:



