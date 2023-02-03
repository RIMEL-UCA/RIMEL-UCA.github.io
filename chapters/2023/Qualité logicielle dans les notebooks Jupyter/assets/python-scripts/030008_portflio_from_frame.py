#!/usr/bin/env python
# coding: utf-8

# ### Import Portfolios from Excel
# 
# This example will demonstrate how to create a `Portfolio` from an excel file.

# In[ ]:


from gs_quant.session import Environment, GsSession
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# Let's assume you have an excel file with many different trades on which you would like to compute `gs-quant` analytics.
# You can import you file into a pandas dataframe with the below command: 

# In[ ]:


# import pandas as pd
# data = pd.read_excel(r'path_to_my_file.csv', sheet_name='my_sheet', usecols=None)

# For the benefit of the analysis here I simulated a dummy dataframe representing the result of your file import.

# In[ ]:


import pandas as pd
data = pd.DataFrame.from_dict({'name': {0: 'my favourite swap',
  1: 'my favourite swaption',
  2: None,
  3: None,
  4: None,
  5: None},
 'trade_type': {0: 'Swap',
  1: 'Swaption',
  2: 'Swaption',
  3: 'Swaption',
  4: 'Swaption',
  5: 'Swaption'},
 'rate': {0: 0.01, 1: None, 2: None, 3: None, 4: None, 5: None},
 'strike': {0: None, 1: '2%', 2: '0.02', 3: '0.02', 4: '0.02', 5: '0.02'},
 'ccy': {0: 'EUR', 1: 'GBP', 2: 'GBP', 3: 'GBP', 4: 'GBP', 5: 'GBP'},
 'freq': {0: '3m/6m',
  1: '3m/6m',
  2: '3m/6m',
  3: '3m/6m',
  4: '3m/6m',
  5: '3m/6m'},
 'index': {0: 'EURIBOR-TELERATE',
  1: 'LIBOR-BBA',
  2: 'LIBOR-BBA',
  3: 'LIBOR-BBA',
  4: 'LIBOR-BBA',
  5: 'LIBOR-BBA'},
 'expiration_date': {0: '30/06/2021',
  1: '30/06/2021',
  2: '3d',
  3: '30/06/2021',
  4: '30/06/2021',
  5: '3m'},
 'asset_class': {0: 'rates',
  1: 'rates',
  2: 'rates',
  3: 'rates',
  4: 'rates',
  5: 'rates'},
 })

# In order to support various excel files and formats, the `from_frame` function takes a mapping argument which enables you 
# to specify which columns of your excel file correspond to the associated `gs_quant` instrument attribute. 
# You may also specify date formats to expand the list of the ones supported by default.

# In[ ]:


from gs_quant.markets.portfolio import Portfolio

mapper = {
    'type' : 'trade_type',
    'fixed_rate': 'rate',
    'pay_ccy': 'ccy',
    'fixed_rate_frequency': lambda row: row['freq'][:row['freq'].index("/")],
    'floating_rate_frequency': lambda row: row['freq'][row['freq'].index("/")+1:],
    'floating_rate_option': lambda row: row['ccy']+'-'+row['index'],
}

portfolio = Portfolio.from_frame(data, mappings=mapper)
portfolio.to_frame().reset_index(drop=True)

# If you excel file is in csv format, you may also use the `from_csv` command which executes the two above steps 
# all together and converts your csv to a `Portfolio` directly.

# In[ ]:


# portfolio = Portfolio.from_csv(r'path_to_my_file.csv', mappings=mapper)

# You can now leverage all the risk and pricing functionality of `gs-quant` on your Excel built portfolio!
