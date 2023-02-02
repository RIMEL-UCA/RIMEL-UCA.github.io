#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

from datetime import date
from gs_quant.markets.position_set import PositionSet
from gs_quant.session import Environment, GsSession

# In[ ]:


client = 'CLIENT ID'
secret = 'CLIENT SECRET'

GsSession.use(Environment.PROD, client_id=client, client_secret=secret, scopes=('read_product_data',))

# Your excel file must be formatted in one of three ways (column names included). The third example will assign each position equal weight.
# <table>
# <tr><th>1. Position Weight</th><th>2. Position Quantity</th><th>3. Position Identifiers</th></tr>
# <tr><td>
#     <table>
#         <th>identifier</th><th>weight</th>
#         <tr>
#             <td>AAPL UW</td>
#             <td>0.4</td>
#         </tr>
#         <tr>
#             <td>MSFT UW</td>
#             <td>0.6</td>
#         </tr>
#         <tr></tr>
#     </table>
# </td><td>
#     <table>
#         <th>identifier</th><th>quantity</th>
#         <tr>
#             <td>AAPL UW</td>
#             <td>100</td>
#         </tr>
#         <tr>
#             <td>MSFT UW</td>
#             <td>100</td>
#         </tr>
#         <tr></tr>
#     </table>
# </td><td>
#     <table>
#         <th>identifier</th>
#         <tr>
#             <td>AAPL UW</td>
#         </tr>
#         <tr>
#             <td>MSFT UW</td>
#         </tr>
#         <tr></tr>
#     </table>
# </td></tr> </table>

# In[ ]:


positions_df = pd.read_excel('positions_data.xlsx') 

# In[ ]:


position_set = PositionSet.from_frame(positions_df)
