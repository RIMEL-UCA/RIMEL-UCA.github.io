#!/usr/bin/env python
# coding: utf-8

# # GS Quant Screens Tutorial

# Marquee data screens persistent views of asset data that can be filtered, modified, created, and deleted. New features in the `gs_quant` library now allow users to view the screens provisioned to them, make new screens based on existing screens, filter screens, delete screens, and modify existing screens using only library functions. Below, we demonstrate how to use each library function and provide examples of common use cases. 

# ### Imports
# 
# First, we import the basic functions we need to begin manipulating screens in `gs_quant`. Additional imports are also used for easier visualization of screen data. To use any of the screen functions in `GsDataScreenApi`, users must have an active `GsSession`.

# In[ ]:


# Required imports 
from gs_quant.session import GsSession, Environment # Authenticate user
from gs_quant.api.gs.data_screen import GsDataScreenApi # Screen functions
from gs_quant.target.data_screen import AnalyticsScreen, FilterRequest, OrderByBody, DataRow # Screen data classes

# Visualization imports 
import pandas as pd 
import pprint
import copy

GsSession.use(Environment.PROD) #Initialize GsSession

# ## Overview

# Below is a brief overview of a typical use case: getting, modifying, filtering, saving, and deleting a screen. For more detail on these actions, continue to the sections below.

# In[ ]:


existing_screen = GsDataScreenApi.get_screens()[0]

column_info = GsDataScreenApi.get_column_info(existing_screen.id)

new_screen_specs = copy.deepcopy(existing_screen)

new_screen_specs.name = 'New Screen'
new_screen_specs.filter_parameters.filters = ({'columnName': 'Name', 'type': 'Substring', 'q': 'Z'},)
new_screen_specs.filter_parameters.limit = 3
new_screen_specs.filter_parameters.include_columns = ('Name', 'Skew')

new_screen = GsDataScreenApi.create_screen(new_screen_specs)

data = GsDataScreenApi.filter_screen(new_screen.id, new_screen.filter_parameters)

display(pd.DataFrame(data))

new_screen_specs = copy.deepcopy(existing_screen)

new_screen_specs.name = 'New Screen'
new_screen_specs.filter_parameters.filters = ({'columnName': 'Name', 'type': 'Substring', 'q': 'Z'},)
new_screen_specs.filter_parameters.limit = 3
new_screen_specs.filter_parameters.include_columns = ('Name', 'Skew')

new_screen = GsDataScreenApi.create_screen(new_screen_specs)

data = GsDataScreenApi.filter_screen(new_screen.id, new_screen.filter_parameters)

display(pd.DataFrame(data))

new_screen.filter_parameters.include_columns = new_screen.filter_parameters.include_columns + ('BBG Ticker',)

new_screen = GsDataScreenApi.update_screen(new_screen.id, new_screen)

data = GsDataScreenApi.filter_screen(new_screen.id, new_screen.filter_parameters)

display(pd.DataFrame(data))

GsDataScreenApi.delete_screen(new_screen.id)

# ## Viewing Screens

# ### Class `AnalyticsScreen`

# Each instance of class `AnalyticsScreen` represents a screen and contains all information about the screen's applied filters, ordering, columns, identifiers, data sources, and other properties. `AnalyticsScreen` objects are commonly returned when gathering data on existing screens or created by the user to generate or update screens. Some notable fields of this type include: 
# 
# 
# - **`name`** *(str, Required)*: The name of this screen. 
# 
# 
# - **`filter_parameters`** *(FilterRequest, Required)*: The filters applied to this screen's data currently.
# 
# 
# - **`base_screener`** *(str, Required)*: An identifier of the data source that this screen uses. All data available in a screen comes directly from the corresponding base screener. The screen also applies all specified filters to this data before displaying it to the user.
# 
# 
# - **`id`** *(str, Optional)*: The unique identifier of this screen. This ID can be used to retrieve screen information from permanent storage. 
# 
# 
# - **`hidden_columns`** *(Tuple[str, ...], Optional)*: Available columns that are currently not shown in this screen. Filtering on a screen allows users to display or hide specific columns. The `hidden_columns` field lists which columns have been excluded from view by the user. 

# ### Function `get_screens()`

# When new screens are created, they are stored persistently. Function `get_screens()` returns a list of all screens available to the user. Each screen or entry in this list is an `AnalyticsScreen` object that can be used to retrieve the screen's data, manipulate the screen, or create a new screen. 

# In[ ]:


# Usage Example: get_screens()

all_screens = GsDataScreenApi.get_screens()
print(all_screens)

screen = all_screens[0]

print(f'\nScreen Name: %s, Base Screener: %s, Screen ID: %s' % (screen.name,  screen.base_screener, screen.id))

# ### Function `get_screen()`

# Use function `get_screen(screen_id)`to retrieve the `AnalyticsScreen` object for a specific screen using its screen ID. 

# In[ ]:


# Usage Example: get_screen()

screen = GsDataScreenApi.get_screen('SCQC82C1C960G1I30P')#REMOVE:Keep permanent screen for this demo? in prod/dev? 

print(screen)
print(f'\nScreen Name: %s, Base Screener: %s, Screen ID: %s' % (screen.name,  screen.base_screener, screen.id_))

# ## Filtering and Viewing Screen Data 

# ### Function `get_column_info()`

# Before filtering a screen's data for specific assets, we may want to view a description of all columns available in this screen and their properties. Use function `get_column_info(screen_id)` to get information about all columns in a screen, including previously hidden columns. Descriptions of certain column types also include other metrics: 
# 
# - For `Enum` type columns, column descriptions include each possible value for the `Enum` as well as the number of times each value appears in the unfiltered data. 
# 
# 
# - For `Number` type columns, column descriptions include the maximum and minimum numeric values present in the unfiltered data.
# 
# 
# - For `String` type columns, column descriptions are currently empty. If an `Enum` type column has more than 20 values, it is automatically converted to a `String` type column. 
# 
# These column properties can help users create more effective filters. The keys of the dictionary returned from `get_column_info()` correspond to the names of each column, and the corresponding values are the description of each column. 

# In[ ]:


# Usage Example: get_column_info() #ASK - is any string coluymn w/ <20 values immediately converted to enum? or no?
#Do some enum cols not show the values? 

column_info = GsDataScreenApi.get_column_info('SCQC82C1C960G1I30P')
pprint.pprint(column_info)

# Example: Number type column description

num_colname = 'Skew'
print(f'\033[4m\nNumeric Column:\033[0m %s\n ' % num_colname)
pprint.pprint(column_info[num_colname])

# Example: Enum type column description

enum_colname = 'Entity ID'
print(f'\033[4m\nEnum Column:\033[0m %s\n ' % enum_colname)
pprint.pprint(column_info[enum_colname])

# Example: String type column description

str_colname = 'Name'
print(f'\033[4m\nString Column:\033[0m %s\n ' % str_colname)
pprint.pprint(column_info[str_colname])


# ### Creating Filters

# Using information about our column types and values, we can now create filters that we will apply to screens to view their filtered data. Additionally, we can permanently filter screens by assigning filters to the `filter_parameters` field in screen objects.
# 
# A configuration of filters for a single screen is stored in a `FilterRequest` object. Some notable fields of the `FilterRequest` class include: 
# 
# 
# - **`include_columns`** *(Tuple[str, ...], Optional)*: A tuple of column names. The columns included in this field will be visible in the screen's data. All other columns will be hidden. If nothing is supplied in this field, all columns will be visible.
# 
# 
# - **`filters`** *(Tuple[dict, ...], Optional)*: A tuple of dictionaries. Each dictionary defines a filter to be applied to the data. Each filter pertains to data in a single specified column. All assets returned from the screen data must meet the requirements of all filters in `filters`. If nothing is supplied in this field, all screen data will be visible. 
# 
# 
# - **`order_by`** *(OrderByBody, Optional)*: An `OrderByBody` object. An instance of `OrderByBody` specifies how the returned data rows of the screen should be ordered after filtering. 
# 
# 
# - **`limit`** *(float, Optional)*: A value specifying the maximum number of results that should be returned from the screen. If the `limit` field has value `n`, the first `n` rows of data will be retrieved based on the ordering in `order_by`.
# 
# 

# Different filters are available based on the different columns and column types in the screen. `Number` type columns support `Range` filters, `Enum` type columns support `Include` filters (which enums to include), and `String` type columns support `Substring` filters. Below is an example of how to generate each type of filter:

# In[ ]:


# Number Columns: Range Filtering 

num_col = "Skew"

range_filter = {'columnName': num_col, 'type': 'Range', 'greaterThanEqual': 0, 'lessThanEqual': 1}

print('\033[4mRange Filter Example:\033[0m\n')
pprint.pprint(range_filter)


# Enum Columns: Include Filtering 

enum_col = "Entity ID"

include_filter = {'columnName': enum_col, 'type': 'Include', 'values': ['VALUE1', 'VALUE2']} #find a screen w/ actual enums

print('\n\033[4mInclude Filter Example:\033[0m\n')
pprint.pprint(include_filter)

# String Columns: Substring Filtering

str_col = "Name"

substring_filter = {'columnName': str_col, 'type': 'Substring', 'q': 'D'}

print('\n\033[4mSubstring Filter Example:\033[0m\n')
pprint.pprint(substring_filter)


# When creating a `FilterRequest` object, include an `OrderByBody` object to enforce an ordering of the retrieved data rows. Notable fields in these instances include: 
# 
# 
# - **`column_name`** *(str, Optional)*: The name of the column to order by.
# 
# 
# - **`type`** *(str, Optional)*: The type of ordering to enforce. Must be either `Ascending` or `Descending`.

# In[ ]:


# Example: Ascending Order by Skew 

order_by = OrderByBody(column_name='Skew', type='Ascending')
print('Column Name: %s, Row Ordering: %s' % (order_by.column_name, order_by.type ))

# ### Function `filter_screen()`

# Use function `filter_screen(screen_id, filter_request)` to view filtered data from a screen. The `filters` parameter of this function is a filter configuration that temporarily overrides the existing filter configuration associated with the given screen. All existing filters applied to this screen are temporarily removed, and the filters supplied in this function will be applied instead.

# In[ ]:


# Usage Example: filter_screen()

# Create a FilterRequest 

filters = (substring_filter, range_filter)

include_columns = ('Name', 'Skew', 'BBG Ticker', 'Vol Premia')

limit = 3

filter_request= FilterRequest(filters=filters, include_columns=include_columns, order_by=order_by, limit=limit)

# Get Filtered Data 

filtered_data = GsDataScreenApi.filter_screen('SCQC82C1C960G1I30P', filter_request)

pprint.pprint(filtered_data)

# Visualize Filtered Data as Dataframe

df = pd.DataFrame(filtered_data)

display(df)

# To view filtered data using the screen's existing configuration, pass the value stored in the `filter_parameters` field of the screen's object into `filter_screen()`.

# In[ ]:


# Example: View Current Configuration Data

screen = GsDataScreenApi.get_screen('SCQC82C1C960G1I30P')

filtered_data = GsDataScreenApi.filter_screen(screen.id, screen.filter_parameters)

pprint.pprint(filtered_data[0:3])

# ## Creating and Deleting Screens

# ### Function `delete_screen()`

# Use function `delete_screen()` to permanently delete an existing screen using its ID. Once a screen is deleted, its information and data cannot be retrieved again. However, it is possible that users may still have access to stale data or `AnalyticsScreen` objects remaining from the old screen.

# ### Function `create_screen()`

# Use `create_screen()` to create new permanent screens using `AnalyticsScreen` objects. To specify a new screen's data source, applied filters, and other attributes, edit the information in the `AnalyticsScreen` instance provided to create the screen. All screens with the same `base_screener` field value will reference the same data source.

# ### Examples: Screen Creation and Deletion

# In[ ]:


# Usage Example: create_screen() and delete_screen()

# Create new screen

filter_request = FilterRequest(filters=({'columnName': 'Name', 'type': 'Substring', 'q': 'A'},), limit=3)

screen_specs = AnalyticsScreen(name='New Screen', filter_parameters=filter_request, base_screener='BS88M7XNRA1D1FL3OM')

print(f'Name: %s, Screen ID: %s' % (screen_specs.name, screen_specs.id))

new_screen = GsDataScreenApi.create_screen(screen_specs)

print(f'Name: %s, Screen ID: %s' % (new_screen.name, new_screen.id))

# View screen data 

data = GsDataScreenApi.filter_screen(new_screen.id, new_screen.filter_parameters)

display(pd.DataFrame(data))

# Delete screen

GsDataScreenApi.delete_screen(new_screen.id)


# Additionally, create copies of existing screens by passing their screen objects directly into `create_screen()`. These existing screen objects can also be modified and passed to `create_screen()` to generate new screens that only differ marginally from an existing screen. 

# In[ ]:


# Example: Create and Delete a Copy of an Existing Screen

# Retrieve an Existing Screen 

existing_screen = GsDataScreenApi.get_screens()[0]

print(f'\033[4mExisting Screen ID and Filters:\033[0m %s\n' % existing_screen.id)
pprint.pprint(existing_screen.filter_parameters.filters)

# Make a Copy

new_screen = GsDataScreenApi.create_screen(existing_screen)

print(f'\n\033[4mCopy Screen ID and Filters:\033[0m %s\n' % new_screen.id)
pprint.pprint(new_screen.filter_parameters.filters)


# Delete Copy 

GsDataScreenApi.delete_screen(new_screen.id)



# ## Updating Existing Screens

# ### Function `update_screen()`

# Use function `update_screen()` to permanently update an existing screen using the ID of the screen and an `AnalyticsScreen` object specifying how the screen should be changed. Note that all previous specifications for the existing screen will be replaced by the information in the `AnalyticsScreen` instance passed into `update_screen()`, including screen names, filters, and data sources. Only the screen ID will remain the same after updating. 
# 
# Additionally, `update_screen()` requires that the screen ID passed in must match the ID field of the screen object passed in. An error is thrown if these two values do not match.

# In[ ]:


#Usage Example: update_screen()

# Create a New Screen 

filter_request = FilterRequest()

screen_spec = AnalyticsScreen(name='Screen to Update', base_screener='BS88M7XNRA1D1FL3OM', filter_parameters=filter_request)

screen = GsDataScreenApi.create_screen(screen_spec)

print(f'Name: %s, ID: %s\n' % (screen.name, screen.id))
pprint.pprint(screen.filter_parameters.filters)

# Update the New Screen

updated_filter_request = FilterRequest(filters=({'columnName': 'Name', 'type': 'Substring', 'q': 'A'},))

screen.filter_parameters = updated_filter_request 

screen.name = 'Updated Screen'

screen = GsDataScreenApi.update_screen(screen.id, screen)

print(f'\nName: %s, ID: %s\n' % (screen.name, screen.id))

pprint.pprint(screen.filter_parameters.filters)

# Delete the New Screen

GsDataScreenApi.delete_screen(screen.id)
