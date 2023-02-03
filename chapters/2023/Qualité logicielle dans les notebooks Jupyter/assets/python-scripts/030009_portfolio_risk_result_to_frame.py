#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.session import Environment, GsSession
import gs_quant.risk as risk
from gs_quant.instrument import IRSwap, IRSwaption
from gs_quant.markets.portfolio import Portfolio
from gs_quant.config import DisplayOptions

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


# portfolio construction
swap_1 = IRSwap('Pay', '5y', 'EUR', fixed_rate=-0.005, name='5y')
swap_2 = IRSwap('Pay', '10y', 'EUR', fixed_rate=-0.005, name='10y')
swap_3 = IRSwap('Pay', '5y', 'USD', fixed_rate=-0.005, name='5y')
swap_4 = IRSwap('Pay', '10y', 'USD', fixed_rate=-0.005, name='10y')
swaption_1 = IRSwaption('Pay', '5y', 'USD', expiration_date='1y', name='5y')
eur_port = Portfolio([swap_1, swap_2], name='EUR')
usd_port = Portfolio([swap_3, swap_4], name='USD')
nested_port = Portfolio([eur_port, usd_port, swaption_1])

# In[ ]:


# risk calculations
eur_port_price = eur_port.price()
nested_port_price = nested_port.price()
nested_port_vega = nested_port.calc(risk.IRVegaParallel)

# Pivot Parameters
# 
# -  inherited from pandas.pivot_table()
# -  index and column values are the names of instruments and portfolios

# In[ ]:


# default to_frame()
nested_port_price.to_frame()

# In[ ]:


# to_frame() with no pivot
nested_port_price.to_frame(values=None, columns=None, index=None)

# In[ ]:


# modify to_frame() with custom pivot parameters - similar to pandas.pivot_table()
nested_port_price.to_frame(values='value', columns='portfolio_name_0',index='instrument_name')

# Aggregation
# -  inherited from pandas.pivot_table()

# In[ ]:


swap_5 = IRSwap('Pay', '5y', 'EUR', fixed_rate=-0.005, name='5y')
swap_6 = IRSwap('Pay', '10y', 'EUR', fixed_rate=-0.005, name='5y')
port=Portfolio([swap_5, swap_6])
res=port.price()

# In[ ]:


# when instruments have the same name, the values are summed by default
res.to_frame()

# In[ ]:


# change aggregation of values
res.to_frame(aggfunc='mean')

# Show N/A values

# In[ ]:


# to_frame() by default eliminates N/A values from the dataframe result
nested_port_vega.to_frame()

# In[ ]:


# pass in display_options to show N/A values
nested_port_vega.to_frame(display_options=DisplayOptions(show_na=True))
