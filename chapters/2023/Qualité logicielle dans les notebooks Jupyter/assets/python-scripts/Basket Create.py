#!/usr/bin/env python
# coding: utf-8

# # Creating a Custom Basket
# 
# Welcome to the basket creation tutorial! Marquee allows you to create your own tradable basket ticker and manage it through the platform. When you create a basket it automatically gets published to Marquee, and you may also publish it to Bloomberg, Reuters, and Factset. This basket will tick live.
# 
# Creating a basket requires enhanced levels of permissioning. If you are not yet permissioned to create baskets please reach out to your sales coverage or to the [Marquee sales team](mailto:gs-marquee-sales@gs.com).

# ## Step 1: Authenticate & Initialize your session
# 
# First you will import the necessary modules and add your client id and client secret.

# In[ ]:


import pandas as pd

from datetime import date
from gs_quant.markets.baskets import Basket
from gs_quant.markets.indices_utils import ReturnType
from gs_quant.markets.position_set import Position, PositionSet
from gs_quant.session import Environment, GsSession

client = 'CLIENT ID'
secret = 'CLIENT SECRET'

GsSession.use(Environment.PROD, client_id=client, client_secret=secret, scopes=('read_product_data read_user_profile modify_product_data',))

# ## Step 2: Define your basket metadata, publishing options, pricing options, & return type
# 
# In this step you are going to define all the specifications needed to create your basket. First, instantiate an empty basket object and then you may begin defining it's settings. The below list contains all the parameters you may set.
# 
# | Parameter Name     | Required?  | Default Value | Description |
# |:-------------------|:-----------|:--------------|:------------|
# |name                |**Required**|--             |Display name of the basket|
# |ticker              |**Required**|--             |Associated 8-character basket identifier (must be prefixed with "GS" in order to publish to Bloomberg). If you would like to request a custom prefix instead of using the default GSMB prefix please reach out to the [baskets team](mailto:gs-marquee-baskets@gs.com)|
# |currency            |**Required**|--             |Denomination you want your basket to tick in. This can not be changed once your basket has been created|
# |return_type         |**Required**|--             |Determines the index calculation methodology with respect to dividend reinvestment. One of Price Return, Gross Return, Total Return|
# |position_set        |**Required**|--             |Information of constituents associated with the basket. You may provide the weight or quantity for each position. If neither is provided we will distribute the total weight evenly among each position. Please also note that any fractional shares will be rounded up to whole numbers.|
# |description         |Optional    |--             |Free text description of basket. Description provided will be indexed in the search service for free text relevance match.|
# |divisor             |Optional    |--             |Divisor to be applied to the overall position set. You need not set this unless you want to change the divisor to a specific number, which will in turn change the basket price (current notional/divisor). This might impact price continuity.|
# |initial_price       |Optional    |100            |Initial price the basket should start ticking at|
# |target_notional     |Optional    |10,000,000     |Target notional for the position set|
# |publish_to_bloomberg|Optional    |True           |If you'd like us to publish your basket to Bloomberg|
# |publish_to_reuters  |Optional    |False          |If you'd like us to publish your basket to Reuters  |
# |publish_to_factset  |Optional    |False          |If you'd like us to publish your basket to Factset  |
# |default_backcast    |Optional    |True           |If you'd like us to backcast up to 5 years of pricing history and compositions, assuming constituents remained constant. Set to false if you'd like to upload your own custom history. If any IPOs are present in this composition, we will stop the backcast period accordingly.|
# |reweight            |Optional    |False          |If you'd like us to reweight positions if input weights don't add up to 1 upon submission|
# |weighting_strategy  |Optional    |--             |Strategy used to price the position set (will be inferred if not indicated). One of Equal, Market Capitalization, Quantity, Weight|
# |allow_ca_restricted_assets|Optional|False        |Allow your basket to have constituents that will not be corporate action adjusted in the future (You will recieve a message indicating if this action is needed when attempting to create your basket)|
# |allow_limited_access_assets|Optional|False       |Allow basket to have constituents that GS has limited access to (You will recieve a message indicating if this action is needed when attempting to create your basket)|

# In[ ]:


my_basket = Basket()

my_basket.name = 'My New Custom Basket'
my_basket.ticker = 'GSMBXXXX'
my_basket.currency = 'USD'
my_basket.publish_to_reuters = True

my_basket.return_type = ReturnType.PRICE_RETURN

# ### Quick Tip!
# At any point, you may call the get_details() method on your basket, which will print the current state of the basket object. We recommend doing this throughout the creation process to ensure there are not any discrepancies between your preferences and the current basket settings.

# In[ ]:


my_basket.get_details() # prints out each parameters on the basket

# ## Step 3: Define your basket's composition
# 
# Now you will decide what your basket composition is. If you'd like to include several positions, you may define the composition using your preferred input method (e.g., uploading an excel file) but it must then be converted to a dictionary or pandas dataframe.
# 
# Your dataframe must have a column entitled 'identifier', which holds any commonly accepted identifier such as BloombergId, Cusip, Ticker, etc. for each position. You may also have a column entitled 'quantity' to store the number of shares for each position, or a column named 'weight' to represent the weight of each. If the second column is missing, we will later assign equal weight to each position when you submit your basket for creation.
# 
# After uploading your composition and converting it to a dataframe, make sure to rename your columns to match our specifications if they aren't in the correct format already, and then you may use it to create a valid Position Set. You should then call get_positions() to make sure that your positions have all been mapped correctly, and can then store this composition on the basket.

# In[ ]:


positions_df = pd.read_excel('path/to/excel.xlsx') # example of uploading composition from excel document
positions_df.columns = ['identifier', 'weight'] # replace weight column with 'quantity' if using number of shares
position_set = PositionSet.from_frame(positions_df)

position_set.get_positions() # returns a dataframe with each position's identifier, name, Marquee unique identifier, and weight/quantity

my_basket.position_set = position_set

# ### Quick Tip!
# Wanting to quickly add one or two positions to a position set without having to modify your dataframe? You can add to a position set by inputting an identifier and an optional weight/quantity to a Position object and modify the position set directly, like below. Refer to the [position_set examples](../examples/03_basket_creation/position_set/0004_add_position_to_existing_position_set.ipynb) section for more tips like this!

# In[ ]:


positions_to_add = [Position('AAPL UW', weight=0.1), Position('MSFT UW', weight=0.1)]
position_set.positions += positions_to_add

my_basket.position_set = position_set

# ## Step 4: Create your basket
# 
# Once you've ensured that your basket has been set up to your satisfaction, you're ready to officially create and publish to Marquee! Once you call create on your new basket, you may poll its status to make sure that it has processed successfully. This will check the report status every 30 seconds for 10 minutes by default, but you can override this option if you prefer as shown below. If you'd like to view your basket on the Marquee site, you can retrieve the link to your page by calling get_url().

# In[ ]:


my_basket.get_details() # we highly recommend verifying the basket state looks correct before calling create!

my_basket.create()

my_basket.poll_status(timeout=120, step=20) # optional: constantly checks create status until report succeeds, fails, or the poll times out (this example checks every 20 seconds for 2 minutes)

my_basket.get_url() # will return a url to your Marquee basket page ex. https://marquee.gs.com/s/products/MA9B9TEMQ2RW16K9/summary

# ## Step 5: Update your basket's entitlements
# 
# The application you use to create your basket will initially be the only one permissioned to view, edit, and submit rebalance requests. If you'd like to entitle other users or groups with view or admin access, you may update your basket's permissions at any time.
# 
# In order to add or remove permissions for a specific user, you will need either their Marquee user id or email. You may also permission groups using their group id. See the snippet below, or refer to the [baskets permissions examples](../examples/07_basket_permissions/0001_permission_application_to_basket.ipynb) for more options.

# In[ ]:


from gs_quant.entities.entitlements import User

user = User.get(user_id='application_id')
basket.entitlements.view.users += [user] # update the desired entitlements block ('edit', 'admin', etc) 'users' property

basket.update()

# ### You're all set, Congrats! What's next?
# 
# * [How do I upload my basket's historical composition?](../examples/03_basket_creation/0001_upload_basket_position_history.ipynb)
# 
# * [How do I retrieve composition data for my basket?](../examples/01_basket_composition_data/0000_get_latest_basket_composition.ipynb)
# 
# * [How do I retrieve pricing data for my basket?](../examples/02_basket_pricing_data/0000_get_latest_basket_close_price.ipynb)
# 
# * [How do I change my basket's current composition?](./Basket%20Rebalance.ipynb)
#   
# * [How do I make other changes to my basket (name, description, etc.)?](./Basket%20Edit.ipynb)
# 
# * [What else can I do with my basket?](https://developer.gs.com/docs/gsquant/api/classes/gs_quant.markets.baskets.Basket.html#gs_quant.markets.baskets.Basket)
# 
# Other questions? Reach out to the [baskets team](mailto:gs-marquee-baskets-support@gs.com) anytime!
