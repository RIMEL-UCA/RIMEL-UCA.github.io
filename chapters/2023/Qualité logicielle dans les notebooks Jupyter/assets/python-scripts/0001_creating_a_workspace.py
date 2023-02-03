#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gs_quant.api.gs.users import GsUsersApi
from gs_quant.api.gs.workspaces import GsWorkspacesMarketsApi as ws_api
from gs_quant.session import Environment, GsSession
from gs_quant.target.workspaces_markets import *

# In[2]:


GsSession.use(Environment.PROD, client_id=None, client_secret=None)

# ### Initializing your Workspace
# 
# Determine the basics of your Workspace (i.e name, description and entitlements over who can view and edit)

# In[3]:


name = 'Example Workspace'
alias = 'example-workspace' # This needs to be unique across all users as to not have conflicting URLs
description = 'This workspace was created as an example.'
user_id = GsUsersApi.get_my_guid()

# Entitle everyone internal to view but only yourself to edit and change entitlements
entitlements = Entitlements(view=(user_id,), edit=(user_id,), admin=(user_id,))

components = [] # Empty list of components with some to be added below

# ### Add Components
# 
# Workspaces currently support many components such as Promo (text), Plots, DataGrids, and Commentary. See all available components
# [here](https://developer.gs.com/p/docs/services/data/workspaces-markets/#components).
# 
# To create a component, create the parameters object for the component such as PromoComponentParameters, fill out the required and optional fields.
# 
# Then add the component using these parameters. Let's start with a simple Promo Component.

# #### Add a Promo Component 
# 
# If you want to add simple text to a workspace, this can be done by adding a Promo Component, as seen below.

# In[ ]:


promo_params = PromoComponentParameters(height=52, body='Your text here!', transparent=False)
components.append(WorkspaceComponent(id_='promo-1', type_=ComponentType.promo, parameters=promo_params))

# ### Create the Workspace
# 
# Now you are ready to create your workspace. Let us put everything together.

# In[ ]:


layout = 'r(c12($0))'
parameters = WorkspaceParameters(layout=layout, components=components)
workspace = Workspace(parameters=parameters,  name=name, alias=alias, entitlements=entitlements, description=description)
workspace = ws_api.create_workspace(workspace)

# The above snippet will create a workspace that is now viewable at the URL https://marquee.gs.com/s/markets/{alias}. 
# Substitute {alias} with the alias you set. Remember to change the alias since the example one probably already exists.
# 
# The layout string controls the layout of your components. In this case, a simple layout with a single component that spans a single row.
# Learn more about layouts [here](https://developer.gs.com/p/docs/services/data/workspaces-markets/#layouts). 
# 

# ### Updating the Workspace with Additional Components
# 
# Now let us create a Workspace with some plots and a commentary component.

# #### Add a plot 
# 
# Start by creating plots in PlotTool Pro [here](https://marquee.gs.com/s/plottool/new) 
# 
# After you have created your plot, grab the plot id from the browser.
# * For example, the id for [this plot](https://marquee.gs.com/s/plottool/CH5RJJ9APZMRQ7B7) is <b>CHYYNR2YSD8W21GA<b>
#     
# You want all the underlying components that have entitlements to have the same entitlements as the Workspace, so all components are visible on the Workspace for the intended audience.

# In[4]:


# Create the plot component parameters, setting the height to the desired height in pixels. Also, other configurations can be set such as hiding the legend.
plot_params = PlotComponentParameters(height=300, hideLegend=False)
plot_id = 'CHYYNR2YSD8W21GA' # example plot id

# Add the plot component to the list of components
components.append(WorkspaceComponent(id_=plot_id, type_=ComponentType.plot, parameters=plot_params))

# #### Add a commentary stream

# In[7]:


channel_1 = 'EQUITIES MACRO'
channel_2 = 'EQUITIES GS BASKETS'
commentary_params = CommentaryComponentParameters(height=300, commentary_channels=(channel_1, channel_2))
components.append(WorkspaceComponent(id_='streaming-commentary-1', type_=ComponentType.commentary, parameters=commentary_params))

# ### Update the Workspace

# In[8]:


workspace.parameters.layout = 'r(c12($0))r(c6($1)c6($2))'
workspace.parameters.components = components
workspace = ws_api.update_workspace(workspace)
