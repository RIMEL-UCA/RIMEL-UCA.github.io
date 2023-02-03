#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.instrument import FXBinary

# In[ ]:


from gs_quant.session import GsSession
# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics'))

# In[ ]:


FXBinary?

# In[ ]:


binary = FXBinary(pair='USDJPY', buy_sell='Buy', expiration_date='3m')
binary.resolve()
print(binary.as_dict())

# #### Strike Price 
# 
# If the strike price is not specified, the current spot is used as the default. Similar to FXOption, the strike price can be specified by a double or a string.
# 
# 
# The specific solver keys are: 
# 
#     - 'S'    - current spot rate
#     - 'F'    - forward
#     - 'P'    - Payout
# 
# You can use these keys to strike_price with the following formats: 
# 
#     - For S, F: 's*1.1', 'F+10%' 
#     - For Delta Strikes, specify the option delta: '25D', '20D-.01', etc.
#     - You can also solve for Premium:  P={target}%
# 

# In[ ]:


from gs_quant.risk import FXSpot
#solve for strike price by setting payout ratio to 10%
binary_10x = FXBinary(pair='EURUSD', buy_sell='Buy', option_type='Put', expiration_date='1m', strike_price='p=10%', premium=0)
binary_10x.resolve()
print('strike price:', binary_10x.strike_price)
print('spot level:', binary_10x.calc(FXSpot))

# In[ ]:


binary_itm = FXBinary(pair='AUDUSD', buy_sell='Buy', expiration_date='3m', strike_price='1.1*s')
binary_itm.resolve()
print('strike price:', binary_itm.strike_price)
print('spot level:', binary_itm.calc(FXSpot))

# In[ ]:


binary_otm = FXBinary(pair='AUDUSD', buy_sell='Buy', expiration_date='3m', notional_amount='100k', strike_price='F-2%')
binary_otm.resolve()
print('strike price:', binary_otm.strike_price)
print('spot level:', binary_otm.calc(FXSpot))
