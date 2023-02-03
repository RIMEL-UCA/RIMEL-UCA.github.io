#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.session import GsSession, Environment
from gs_quant.markets.portfolio import Portfolio
from gs_quant.instrument import IRSwap
import numpy as np
import pandas as pd
from gs_quant.instrument import Instrument

# In[ ]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('run_analytics',))

# In[ ]:


IRSwap?

# The Portfolio.from_csv mappings is a dictionary mapping a gs-quant instrument field to your csv column name (columns must not have duplicates). Mappings can also be a callable lambda function.

# In[ ]:


mappers = {'type': lambda row: IRSwap.type_.value, 'asset_class': lambda row: IRSwap.asset_class.value,
           'effective_date': 'EffectiveDate',
           'pay_or_receive': lambda row: 'Pay' if float(row['Notional'].replace(',', '')) < 0 else 'Receive',
           'termination_date': 'EndDate',
           'fixed_rate': 'Coupon/Spread',
           'notional_amount': 'Notional',
           'notional_currency': 'CCY1',
           'roll_convention': lambda row: 'IMM' if row['Roll Conv'] == 'IMM' else None,
           'fixed_rate_frequency': lambda row: '3m' if row['Frequency'] == 'QUARTERLY' else '6m'
           }


# In[ ]:


p = Portfolio.from_csv('my_excel_portfolio.csv', mappers)
p.resolve()

# In[ ]:


p[0].as_dict()
