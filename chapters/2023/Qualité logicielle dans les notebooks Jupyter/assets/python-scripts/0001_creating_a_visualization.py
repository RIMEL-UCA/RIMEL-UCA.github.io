#!/usr/bin/env python
# coding: utf-8

# ### Introduction to Data Visualizations
# 
# The Data Visualizations class `PlotlyViz` allows you to create and visualize different chart objects in Marquee.
# This class has been created to provide flexibility to our users when creating and visualizing charts. All while 
# maintaining Marquee style and utilizing the proper Marquee workflow to persist and entitle these objects.
# 
# At the moment we support 5 different chart types:
# 
#     * Bar Charts
#     * Scatter Charts
#     * Bubble Charts
#     * Pie Charts
#     * Radar Charts
#  
# Let's take a look at a few example and concepts to better understand how to work and create these visualizations.
# 
# First, let's import all necessary libraries:

# In[ ]:


from gs_quant.analytics.datagrid import DataRow, DataColumn, DataGrid
from gs_quant.analytics.dataviz import PlotlyViz
from gs_quant.analytics.processors import EntityProcessor, VolatilityProcessor
from gs_quant.analytics.processors.utility_processors import   LastProcessor, \
    DivisionProcessor
from gs_quant.api.gs.users import GsUsersApi
from gs_quant.data import DataCoordinate, DataMeasure, DataFrequency, DataDimension
from gs_quant.datetime.relative_date import RelativeDate
from gs_quant.entities.entitlements import Entitlements, EntitlementBlock, User
from gs_quant.markets.securities import Asset, AssetIdentifier

# Next, let's create our Datagrid, all the visualization example we will be using the following [Datagrid](https://developer.gs.com/docs/gsquant/data/data-analytics/datagrid/overview/) 
# as data source:

# In[3]:


# Fetching assets to work with
SPX = Asset.get("MA4B66MW5E27U8P32SB", AssetIdentifier.MARQUEE_ID)
SX5E = Asset.get("MA4B66MW5E27U8P32SY", AssetIdentifier.MARQUEE_ID)
UKX = Asset.get("MA4B66MW5E27U8NN95P", AssetIdentifier.MARQUEE_ID)
HSCEI = Asset.get("MA4B66MW5E27U8P3294", AssetIdentifier.MARQUEE_ID)
NKY = Asset.get("MA4B66MW5E27U8P32LH", AssetIdentifier.MARQUEE_ID)
KS200 = Asset.get("MA4B66MW5E27U8P32DM", AssetIdentifier.MARQUEE_ID)
IBOV = Asset.get("MA4B66MW5E27U8NN8N6", AssetIdentifier.MARQUEE_ID)

# Defining Datagrid rows
rows = [
    DataRow(SPX),
    DataRow(SX5E),
    DataRow(UKX),
    DataRow(HSCEI),
    DataRow(NKY),
    DataRow(KS200),
    DataRow(IBOV)
]

# Defining column operations
spx_spot = DataCoordinate(
    measure=DataMeasure.CLOSE_PRICE,
    frequency=DataFrequency.DAILY
)

implied_vol = DataCoordinate(
    measure=DataMeasure.IMPLIED_VOLATILITY,
    frequency=DataFrequency.DAILY,
    dimensions={
        DataDimension.TENOR: "3m",
        DataDimension.STRIKE_REFERENCE: "delta",
        DataDimension.RELATIVE_STRIKE: 0.5
    }
)

implied_spot = LastProcessor(implied_vol)
realized_vol = LastProcessor(
    DivisionProcessor(VolatilityProcessor(spx_spot, w=63, start=RelativeDate("-1y")), dividend=100)
)

col_0 = DataColumn(name="Name", processor=EntityProcessor(field="short_name"))
col_1 = DataColumn(name="Implied Volatility", processor=implied_spot)
col_2 = DataColumn(name="Realized Vol", processor=realized_vol)

columns = [
    col_0,
    col_1,
    col_2
]

# Entitlements
# Entitle everyone internal to view but only yourself to edit and change entitlements
user_id = GsUsersApi.get_my_guid()
self_entitlement_block = EntitlementBlock(users=[User.get(user_id=user_id)])
entitlements = Entitlements(view=self_entitlement_block, edit=self_entitlement_block, admin=self_entitlement_block)

# Datagrid creation
datagrid = DataGrid(
    name="Datagrid For Visualization",
    rows=rows,
    columns=columns,
    entitlements=entitlements
)

datagrid_id = datagrid.save()
datagrid.initialize()
datagrid.poll()
datagrid.to_frame()

# Once we have our Datagrid we can use it as our data source to create the visualizations. Let's start with a Bar Chart
# example:
# 
# ### Bar Charts
# 
# To create a bar chart, simply pass in your datagrid and entitlements to the `PlotlyViz` class to initialize the class.

# In[ ]:


visualization = PlotlyViz(datagrid=datagrid, entitlements=entitlements)

# Then, call the bar function, passing the parameters that you require for your visualization. In this case we're simply
# passing x, y and title. We're passing the column names from our Datagrid to our `x` and `y` parameters as well as a 
# visualization title:

# In[ ]:


visualization.bar(x='Name', y='Realized Vol', title="Realized Vol Bar Chart Example")

# *Note: All parameters used in our visualization functions are the same parameters you'd use in [Plotly Express](https://plotly.com/python/plotly-express/).
# 
# Next, we can persist our chart and visualize it in Marquee like so:

# In[ ]:


viz_id = visualization.save()
# The open() method does not work in Jupyter, but it works fine from your own IDE.
visualization.open() 
print(f'Your data visualization should be viewable at: https://marquee.gs.com/s/markets/visualizations/{viz_id}') 


# ### Scatter Charts
# 
# Similarly, we can create scatter charts like so:
# 
# In our previous Bar chart example, we passed the Datagrid variable directly to `PlotlyViz`. Now, since that Datagrid
# is already created, let's simply use it's ID as reference:

# In[ ]:


visualization = PlotlyViz(datagrid_id=datagrid_id, entitlements=entitlements)

# Next, use the scatter function and pass the required variables. After that, you should be able to save and open it to
# visualize in Marquee:

# In[ ]:


visualization.scatter(x='Implied Volatility', y='Realized Vol', color="Name", title="Scatter Chart Example")
viz_id = visualization.save()
# The open() method does not work in Jupyter, but it works fine from your own IDE.
visualization.open()
print(f'Your data visualization should be viewable at: https://marquee.gs.com/s/markets/visualizations/{viz_id}') 

# ### Bubble Charts
# 
# Bubble charts are actually scatter charts but with a third variable that specifies the dot size. Here's an example:

# In[ ]:


visualization = PlotlyViz(datagrid_id=datagrid_id, entitlements=entitlements)

# Now, call the scatter function with the required variables. After that, you should be able to save and open it to
# visualize in Marquee:

# In[ ]:


# Using Realized Vol as the column to use for dot size
visualization.scatter(x='Implied Volatility', y='Realized Vol', size='Realized Vol', color="Name", title="Bubble Chart Example")
viz_id = visualization.save()
# The open() method does not work in Jupyter, but it works fine from your own IDE.
visualization.open()
print(f'Your data visualization should be viewable at: https://marquee.gs.com/s/markets/visualizations/{viz_id}') 

# ### Radar Charts
# 
# A useful chart in different scenarios is the Radar Chart. This chart can be created like so:

# In[ ]:


visualization = PlotlyViz(datagrid_id=datagrid_id, entitlements=entitlements)

# The function to call this time is called `line_polar`. Call this function with the required variables. After that, you should be able to save and open it to
# visualize in Marquee:

# In[ ]:


visualization.line_polar(r='Implied Volatility', theta="Name", line_close=True, title="Radar Chart")
viz_id = visualization.save()
# The open() method does not work in Jupyter, but it works fine from your own IDE.
visualization.open()
print(f'Your data visualization should be viewable at: https://marquee.gs.com/s/markets/visualizations/{viz_id}') 

# Again, all the parameters used in these visualizations are the same that [Plotly Express](https://plotly.com/python/plotly-express/) uses,
# You can also see some useful examples on their site for the visualizations we support: [Bar](https://plotly.com/python/bar-charts/), 
# [Scatter](https://plotly.com/python/line-and-scatter/), [Bubble](https://plotly.com/python/line-and-scatter/#bubble-scatter-plots), 
# [Pie](https://plotly.com/python/pie-charts/), [Radar](https://plotly.com/python/radar-chart/)
