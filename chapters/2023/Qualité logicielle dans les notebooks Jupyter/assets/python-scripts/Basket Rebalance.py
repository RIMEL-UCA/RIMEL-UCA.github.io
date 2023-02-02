#!/usr/bin/env python
# coding: utf-8

# # Rebalancing a Custom Basket
# 
# Here we will go through how to modify your basket's current composition. You may concurrently modify basket details, pricing options, and publishing preferences.

# ## Step 1: Authenticate & Initialize your session
# 
# First you will import the necessary modules and add your client id and client secret.

# In[ ]:


from gs_quant.markets.position_set import PositionSet
from gs_quant.markets.baskets import Basket
from gs_quant.session import Environment, GsSession

client = 'CLIENT ID'
secret = 'CLIENT SECRET'

GsSession.use(Environment.PROD, client_id=client, client_secret=secret, scopes=('read_product_data read_user_profile',))

# ## Step 2: Fetch your basket and set your changes
# 
# Next you will fetch the basket you'd like to update by passing in any of its identifiers such as BloombergId, Cusip, Ticker, etc. If this is a basket you or someone from your organization created, please make sure your application has admin entitlements or you will not be able to submit the rebalance request. You will then be ready to change any of the properties listed below.
# 
# | Parameter Name     | Required?  | Description |
# |:-------------------|:-----------|:------------|
# |position_set        |**Required**|Information of constituents associated with the basket. You may provide the weight or quantity for each position. If neither is provided we will distribute the total weight evenly among each position. Please note that any fractional shares will be rounded up to whole numbers.|
# |divisor             |Optional    |Divisor to be applied to the overall position set. Ideally, you should not to change this as it will cause a price deviation on the basket.|
# |initial_price       |Optional    |Price you'd like to reset the basket to. Ideally, you should not to do this as it will cause a price deviation on the basket.|
# |publish_to_bloomberg|Optional    |If you'd like us to publish your basket to Bloomberg|
# |publish_to_reuters  |Optional    |If you'd like us to publish your basket to Reuters  |
# |publish_to_factset  |Optional    |If you'd like us to publish your basket to Factset  |
# |include_price_history|Optional    |Republish price history based on current composition when publishing to Bloomberg|
# |reweight            |Optional    |If you'd like us to reweight positions if input weights don't add up to 1 upon submission|
# |weighting_strategy  |Optional    |Strategy used to price the position set (will be inferred if not indicated). One of Equal, Market Capitalization, Quantity, Weight|
# |allow_ca_restricted_assets|Optional|Allow your basket to have constituents that will not be corporate action adjusted in the future (You will recieve a message indicating if this action is needed when attempting to rebalance your basket)|
# |allow_limited_access_assets|Optional|Allow basket to have constituents that GS has limited access to (You will recieve a message indicating if this action is needed when attempting to rebalance your basket)|

# In[ ]:


basket = Basket.get('GSMBXXXX')

basket.publish_to_bloomberg = True

positions_df = pd.read_excel('path/to/excel.xlsx') # example composition upload from a local excel file
positions_df.columns = ['identifier', 'weight'] # replace weight column with 'quantity' if using number of shares
position_set = PositionSet.from_frame(positions_df)

position_set.get_positions() # returns a dataframe with each position's identifier, name, Marquee unique identifier, and weight/quantity

my_basket.position_set = position_set

# ## Step 3: Submit your changes
# 
# Once you've ensured that your basket composition has been adjusted to your satisfaction, you're ready to officially submit these changes to Marquee! Once you call update on the basket, this request will be sent for approval. You can check on the approval status by calling get_rebalance_approval_status(). The rebalance will begin processing once the request is approved, where you can then poll its status to make sure that it has processed successfully. This will check the report status every 30 seconds for 10 minutes by default, but you can override this option if you prefer as shown below.

# In[ ]:


basket.update() # submits the rebalance request to Marquee

basket.get_rebalance_approval_status() # check approval status of most recent rebalance submission

basket.poll_status(timeout=120, step=20) # optional: constantly checks rebalance status after request is approved until report succeeds, fails, or the poll times out (this example checks every 20 seconds for 2 minutes)

# ### Not happy with the new composition you submitted?
# 
# If your most recent rebalance request is not yet approved, you may either update your composition and submit a new rebalance request using the steps listed above, or you can simply cancel the request.

# In[ ]:


basket.cancel_rebalance()
