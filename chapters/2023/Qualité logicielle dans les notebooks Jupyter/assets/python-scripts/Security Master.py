#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# A Security Master provides financial product reference data and is a foundational building block of any quantitative financial analytics engine. The core component of Goldman Sachs' Security Master is symbology, the data necessary to map from one identifier to another. Clients can retrieve this data via the GS Quant SDK.

# ## Getting Started
# Set up a GsSession. You must input the client_id and client_secret of an application with access to Security Master. For more information on the product and on how to get access, please contact [GS Data Services](mailto:data-services@gs.com).

# In[ ]:


from gs_quant.session import Environment, GsSession
GsSession.use(client_id='client_id', client_secret='client_secret')

# Set source to Security Master (or else it will default to Asset Service).

# In[ ]:


from gs_quant.markets.securities import SecurityMaster, SecurityMasterSource
SecurityMaster.set_source(SecurityMasterSource.SECURITY_MASTER)

# ## Querying
# Retrieve a security by identifier. The get_asset method returns identifiers at a point in time specified by the `as_of` parameter (which should be set to 2021-01-01 or later, given the history available in Security Master).

# In[ ]:


import datetime
from gs_quant.markets.securities import SecurityIdentifier, SecurityMaster

asset = SecurityMaster.get_asset('GS UN', SecurityIdentifier.BBID, as_of=datetime.date(2021, 1, 5))
print(asset, '\n')
asset.get_identifiers()

# You can use SecurityIdentifier members to pull out identifier values.

# In[ ]:


asset.get_identifiers()[SecurityIdentifier.BBID.value]

# ## Bulk Querying
# 
# #### History
# You can retrieve identifier history (from 2021-01-01 onwards) for a list of assets. The following example also shows a ticker change.

# In[ ]:


import datetime
from gs_quant.markets.securities import SecurityIdentifier, SecurityMaster

ids = ['CTLP UW', 'TSLA UW', 'FB UW']
as_of = datetime.datetime(2021, 6, 5)  # as-of-date used to map input IDs to assets
start = datetime.datetime(2000, 1, 5)  # get identifier entries with update time after this time
end = as_of  # get identifier entries with update time before this time
identifiers = SecurityMaster.get_identifiers(ids, SecurityIdentifier.BBID, as_of=as_of, start=start, end=end)
[identifier for identifier in identifiers['CTLP UW'] if identifier['type'] == SecurityIdentifier.TICKER.value]

# #### Point in Time
# 
# You can retrieve point-in-time identifiers without specifying individual assets.

# In[ ]:


import datetime
from gs_quant.markets.securities import AssetType, SecurityMaster
from gs_quant.target.common import AssetClass

as_of = datetime.datetime(2021, 5, 1)
identifiers = SecurityMaster.get_all_identifiers(AssetClass.Equity, [AssetType.STOCK], as_of=as_of)
identifiers[next(iter(identifiers))]  # show identifiers for one of the retrieved securities

# You can also use a generator to retrieve each page of results when needed - just call `next()`.

# In[ ]:


import datetime
from gs_quant.markets.securities import SecurityMaster

generator = SecurityMaster.get_all_identifiers_gen(as_of=datetime.datetime(2021, 5, 1))
page_1 = next(generator)
page_2 = next(generator)
page_3 = next(generator)
# and so on... (keep in mind that a generator will raise StopIteration when it is done)

# ## Mapping Between Identifiers
# 
# To map from one ID to other IDs for the same security, use the `map_identifiers` function.
# For example, if we have a security "GS UN" and want to find its CUSIP over a certain date range: 

# In[ ]:


import datetime
from gs_quant.markets.securities import SecurityMaster

start = datetime.date(2021, 10, 11)
end = datetime.date(2021, 10, 15)
SecurityMaster.map_identifiers(["GS UN"], [SecurityIdentifier.CUSIP], start, end)

