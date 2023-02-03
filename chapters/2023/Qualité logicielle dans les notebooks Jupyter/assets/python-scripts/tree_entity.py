#!/usr/bin/env python
# coding: utf-8

# In[2]:


from gs_quant.session import Environment, GsSession
from gs_quant.entities.tree_entity import TreeHelper

# In[3]:


# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None, scopes=('read_product_data',))

# ### Tree class usage

# In[ ]:


# Initialise the TreeHelper with the MarqueeID of the root Asset and the dataset to be used to build the tree.
# In this example, we are building an STS index tree, so we use STS index datasets.

tree = TreeHelper('MAM9TYYXXXXXXXXX', tree_underlier_dataset='STS_UNDERLIER_WEIGHTS', underlier_column='underlyingAssetId')

# In[ ]:


# Builds the tree with the underlier dataset passed 
tree.build_tree() 

# In[ ]:


# Populate weights and attributions of the asset with the appropriate dataset and column name. 
# In this example, we use the datasets for STS indices.

tree.populate_weights(dataset='STS_UNDERLIER_WEIGHTS', weight_column='weight')

tree.populate_attribution(dataset='STS_UNDERLIER_ATTRIBUTION', attribution_column='absoluteAttribution')

# In[ ]:


# Get tree as a df with the contents stored at this point
tree.to_frame()

# In[ ]:


# Prints the tree structure formed by the Index for easy visualisation
tree.get_visualisation(visualise_by='asset_name')

# In[ ]:


# This feature needs treelib to be installed.
tree.get_visualisation(visualise_by='bbid')

# In[ ]:


# Get the time at which the tree was built
tree.update_time

# In[ ]:


# This feature needs the notebook package of gs-quant to be installed.
# You can install it using 'pip install gs-quant[notebook]'
tree.get_visualisation(visualise_by='bbid')

# In[ ]:


# Get the time at which the tree was built
tree.update_time

# ### Accessing the child nodes

# In[ ]:


# Accessing the root node of the tree
root_node = tree.root

# In[ ]:


# Accessing the child nodes of the root node.
child_nodes_depth_1 = root_node.direct_underlier_assets_as_nodes

# Each of these child nodes are TreeNode objects themselves, so their children can be accessed using the same technique. This can be used to recursively explore the full tree. 

# In[ ]:


# Accessing the child nodes of the first child node of the root.
child_nodes_depth_2 = child_nodes_depth_1[0].direct_underlier_assets_as_nodes
