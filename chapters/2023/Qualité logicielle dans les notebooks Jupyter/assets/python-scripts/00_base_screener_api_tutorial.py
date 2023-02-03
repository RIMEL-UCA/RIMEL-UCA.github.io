#!/usr/bin/env python
# coding: utf-8

# # GS Quant Base Screener Tutorial

# Base screeners are tools used to explore and compare information about specific assets. These screeners act as permanent data tables containing rows of information about different assets and indices. Each base screener is recomputed once a day, and `gs_quant` screens act as persistent filters for screeners. Each base screener has two main components: the screener schema, which determines what data is permitted to be in the screener, and the screener data, which can be published to the screener after the schema is fixed. 
# 
# New features in the `gs_quant` library now allow users with the requisite permissions to create and manipulate screeners using only library functions. Below, we demonstrate how to use each library function and provide examples of common use cases. 

# ### Imports
# 
# First, we import the basic functions we need to begin manipulating base screeners in `gs_quant`. Additional imports are also used for easier visualization of screener data. 
# 
# 
# To use any of the screener functions in `GsBaseScreenerApi`, users must have an active `GsSession`. Users must also be members of the `PlotScreenerAdmins` group in `gs_quant` in order to use functions such as `create_screener()`, `edit_screener()`, `clear_screener()`, `delete_screener()`, and `publish_to_screener()`. 

# In[ ]:


# Required imports 
from gs_quant.session import GsSession, Environment # Authenticate user
from gs_quant.api.gs.base_screener import GsBaseScreenerApi # Base screener functions
from gs_quant.target.base_screener import Screener, ScreenerParameters, ScreenerRow, ScreenerColumn # Screener data classes
from gs_quant.target.common import FilterRequest, DataRow

# Visualization imports 
import pandas as pd 
import pprint
from gs_quant.api.gs.data_screen import GsDataScreenApi
from gs_quant.target.data_screen import AnalyticsScreen
from time import sleep

GsSession.use(Environment.PROD) #Initialize GsSession

# ## Overview

# Below is a brief overview of a typical use case: getting a screener, creating a new screener, editing a schema, publishing data to a screener, viewing screener data, clearing a screener, and deleting a screener. For more detail on these actions, continue to the sections below.

# In[ ]:



existing_schema = GsBaseScreenerApi.get_screener('BSW3GT9DS8A86MTSH3')


existing_schema.name = 'Tutorial Overview Screener'

new_screener = GsBaseScreenerApi.create_screener(existing_schema)



new_col = ScreenerColumn(column_name='1m IV', expression='ASSET.implied_volatility(1m, spot, 100)')

new_screener.parameters.columns = new_screener.parameters.columns + (new_col, )



data = {
  'rows': [
    {
        'id': 0, 
        'Name': 'Obalon Therapeutics Inc',
        'BBID': 'DSE',
        '1m IV': 0.15
        
    },
    {
        'id': 1,
        'Name': 'Disney Inc',
        'BBID': 'DIS',
        '1m IV': 0.16
    }
  ]
}

GsBaseScreenerApi.publish_to_screener(new_screener.id, data)

sleep(1)



screen_schema = AnalyticsScreen(name='Overview Screener View', filter_parameters=FilterRequest(), base_screener=new_screener.id)

screen = GsDataScreenApi.create_screen(screen_schema)

display(pd.DataFrame(GsDataScreenApi.filter_screen(screen.id, screen.filter_parameters)))



GsBaseScreenerApi.clear_screener(new_screener.id)

sleep(1)

display(pd.DataFrame(GsDataScreenApi.filter_screen(screen.id, screen.filter_parameters)))


GsDataScreenApi.delete_screen(screen.id)

GsBaseScreenerApi.delete_screener(new_screener.id)

# ## Viewing Existing Base Screeners

# ### Class `Screener`

# In `gs_quant`, a screener is represented by an object of class `Screener`. More explicitly, a `Screener` object describes the schema of an existing base screener. These objects are often returned when retrieving a screener using its ID or creating a new screener with a specific schema. Some notable fields of this type include: 
# 
# 
# - **`name`** *(str, Required)*: The name of the base screener.
# 
# 
# - **`parameters`** *(ScreenerParameters, Required)*: An object of type ScreenerParameters that specifies the exact rows (ex. assets or indices) and columns (attributes) that should be visible in this screener. Only data conforming to the ScreenerParameters object of the screener can be published to the screener. This `ScreenerParameters` object also determines how values in the screener are updated over time. 
# 
# 
# - **`id`** *(str, Optional)*: An ID that uniquely identifies this screener, which is permanent. As long as this screener is not deleted, this ID will reference the same screener. This ID can be used to retrieve screener information and data from permanent storage. 
# 
# 
# - **`cron_schedule`** *(str, Required)*: A cron schedule expression to schedule a cron job using this screener. Currently, this feature is not implemented; however, the functions using `Screener` objects still expect a valid cron schedule expression in this location.

# ### Function `get_screeners()`

# When new screeners are created, they are stored persistently. Function `get_screeners()` returns a list of all base screeners available to the user. Each screener or entry in this tuple is a `Screener` object that can be used to retrieve the screener's data, manipulate the screener, or create a new screener. 

# In[ ]:


# Usage Example: get_screeners()

all_screeners = GsBaseScreenerApi.get_screeners()
print(all_screeners)

screener = all_screeners[0]

print(f'\nBase Screener Name: %s, Base Screener Column 1: %s, Base Screener ID: %s' \
      % (screener.name,  screener.parameters.columns[0].column_name, screener.id))

# ### Function `get_screener()`

# Use function `get_screener(screener_id)`to retrieve the `Screener` object for a specific screener using its ID. This function allows a user to search for a specific screener to view its schema and information. 

# In[ ]:


# Usage Example: get_screener()

screener = GsBaseScreenerApi.get_screener('BSW3GT9DS8A86MTSH3')

print(screener)

print(f'\nBase Screener Name: %s, Base Screener Column 1: %s, Base Screener ID: %s' \
      % (screener.name,  screener.parameters.columns[0].column_name, screener.id))

# ## Creating and Deleting Base Screeners 

# ### Class `ScreenerRow`

# Class `ScreenerRow` defines a row or set of rows (i.e. assets or indices) that can be viewed in this screener. Note that this class does not publish data to a screener. Rather, `ScreenerRow` objects contain information about possible rows that the base screener's schema should accept. Tuples of `ScreenerRow` objects are added to the `rows` field of `ScreenerParameter` instances to define the rowwise schema of a screener; it is up to the user to then publish corresponding data to the screener in order to make this data visible. Notable fields of this type include: 
# 
# 
# - **`entity_id`**, *(str, Required)*: The unique ID of the entity that this `ScreenerRow` adds to the rowwise schema.
# 
# 
# - **`entity_type`**, *(str, Required)*: The type of the entity specified in this `ScreenerRow` object. Currently, this field is not in use and can just be set to the string `'asset'`.
# 
# 
# - **`expand`**, *(bool, Optional)*: In the event that the asset is a positioned entity, such as an index or a mutual fund, a user can opt to enter all assets contained in this entity as their own rows in the schema by setting the `expand` field to be `True`. 
# 
# **Note:** Currently, although users specify individual assets for each row in the schema, these specifications are not strictly enforced. The screener will not reject data that is not one of the permitted assets specified in the schema, but it is best practice to create a schema that matches the intended data closely.

# ### Class `ScreenerColumn`

# Class `ScreenerColumn` defines a single column to be added to a base screener's schema. Similarly to `ScreenerRow`, this class does not publish data to a screener. Rather, `ScreenerColumn` objects contain information about possible columns that the base screener's schema should accept. Tuples of `ScreenerColumn` objects are added to the `columns` field of `ScreenerParameter` instances to define the columnwise schema of a screener; it is up to the user to then publish corresponding data to the screener in order to make this data visible. Notable fields of this type include: 
# 
# 
# - **`column_name`** *(str, Required)*: The name of the column to add to the base screener's schema.
# 
# 
# - **`entity_parameter`** *(str, Optional)*: The entity parameter name corresponding to the values in this column, if applicable. Specifying this field allows the screener to be automatically updated once per day with the correct value of this column for each asset. Possible `entity_parameter` values include `id`, `bbid`, `region`, `assetClass`, `currency`, `description`, `exchange`, `name`, `type`, `ric`, `region`, `assetClassificationsCountryName`, `assetClassificationsGicsSector`, and `assetClassificationGicsIndustry`.
# 
# 
# - **`expression`** *(str, Optional)*: The PlotTool expression to calculate this column, if applicable. For calculated columns, providing a PlotTool expression allows the screener to update once per day with the most recent value of this expression. No specific asset should be provided with this expression, as it will be applied individually to each asset in each row of the screener's data. Use the placeholder `ASSET` for an asset name in the expression where needed (ex. `"ASSET.implied_volatility(1m, delta_call, 50)"`).
# 
# 
# **Note**: Although both the `entity_parameter` and `expression` fields are optional, each individual column must fill out exactly one of these fields. Currently, base screeners do not support entering custom asset values not tied to expressions or entity information. 

# ### Class `ScreenerParameters`

# Class `ScreenerParameters` gathers and defines an entire schema for a base_screener. This schema object is then assigned to the `parameters` field in a `Screener` object when creating a new screen with this specific schema. Notable fields of this type include: 
# 
# 
# - **`name`** *(str, Optional)*: The name of this schema. 
# 
# 
# - **`rows`** *(ScreenerRow tuple, Required)*: A tuple of `ScreenerRow` objects that define each allowed data row in the schema.
# 
# 
# - **`columns`** *(ScreenerColumn tuple, Required)*: A tuple of `ScreenerColumn` objects that define the allowed columns in the schema.

# ### Function `create_screener()`

# To create a new base screener, pass a `Screener` object into the function `create_screener(screener)`. The `Screener` object should have the correct specifications for the new base screener, including a `parameters` field with a `ScreenParameters` object populated with the correct schema. The `id` field of the input `Screener` object can be left blank, as this function will return an identical `Screener` object to the input, except it will have its `id` field populated with the ID for the new base screener. 
# 
# The returning of the identical `Screener` objects confirms that the new base screener has the intended schema, which is also visible by using the `get_screener()` function with the new screener ID.
# 
# **Note:** Only users who are members of the `PlotScreenerAdmins` group can use the `create_screener()` function to make new screeners.

# ### Function `delete_screener()`

# Pass a screener's unique ID into the function `delete_screener(screener_id)` to permanently delete a base screener and all of its associated data. The screener will no longer be accessible to any user after this action. 
# 
# **Note:** Only users who are members of the `PlotScreenerAdmins` group can use the `delete_screener()` function to remove screeners.

# In[ ]:


# Usage Example: create_screener(), delete_screener()

# Create a Schema 

rows = (ScreenerRow(entity_id='MAYDMZQXXVXD9JK7', entity_type='asset'), \
        ScreenerRow(entity_id='MAQJQTVRMKVKX9FZ', entity_type='asset'))

columns = (ScreenerColumn(column_name='Name', entity_parameter='name'), \
          ScreenerColumn(column_name='BBID', entity_parameter='bbid'), \
          ScreenerColumn(column_name='RSI 14 DAY', expression='relative_strength_index(ASSET.spot(), 14d)'),\
          ScreenerColumn(column_name='Skew', expression='ASSET.skew(1m, delta, 25, normalized)'))

params = ScreenerParameters(rows=rows, columns=columns)

cron_schedule = '5 4 * * *'

schema = Screener(name='My New Screener', parameters=params, cron_schedule=cron_schedule)
        
# Create a New Screener 

new_screener = GsBaseScreenerApi.create_screener(schema)

print(f'Screener Name: %s, Screener ID: %s\n' % (new_screener.name, new_screener.id))

# Delete Screener

GsBaseScreenerApi.delete_screener(new_screener.id)


# Additionally, we can use existing `Screener` objects and function `create_screener(screener)` to create new screeners by copying or modifying existing screeners. 

# In[ ]:


# Example: Create a New Screener from an Existing Schema and Delete It

# Retrieve Existing Screener

screener = GsBaseScreenerApi.get_screener('BSW3GT9DS8A86MTSH3')

print(f'Screener ID: %s, Screener Columns: %s\n' \
      % (screener.id, [column.column_name for column in screener.parameters.columns]))

# Modify Schema 

screener.parameters.columns = (screener.parameters.columns[0],)

# Create a New Screener

new_screener = GsBaseScreenerApi.create_screener(screener)

print(f'New Screener ID: %s, New Screener Columns: %s' \
      % (new_screener.id, [column.column_name for column in new_screener.parameters.columns]))

# Delete Screener 

GsBaseScreenerApi.delete_screener(new_screener.id)

# ## Viewing Base Screener Data
# 
# 
# To view data from a base screener, please create a `gs_quant` screen that sources data from this base screener and uses any desired filters. More information on how to create a screen can be found in the `gs_quant` screens tutorial. 
# 
# **Note**: Please wait a short amount of time between alternating operations of editing the base screener and viewing its data to allow time for the base screener to update. Using the python `sleep(seconds)` function in the `time` module between operations is a simple way to ensure consistency when viewing recently edited data from a base screener. Between nearby operations, `sleep(1)` should be sufficient.

# ## Publishing Data to Base Screeners and Clearing Screeners

# ### Function `clear_screener()`

# To clear a base screener of its data, use the function `clear_screener(screener_id)` with the screener ID of the base screener being cleared. The base screener schema will still be permanently stored and accessible with the same ID, but the screener will be empty.

# ### Data Publishing Format

# After a base screener is created using a schema, data following this schema can be added to the screener. 
# 
# 
# `gs_quant`'s publishing function accepts data as a dictionary containing a single list of dictionaries. Each dictionary in the list identifies a single row (or asset) of data. The keys of the dictionary must match the names of columns in the schema, and the value corresponding to a specific key will be the value entered at this column for a particular asset. Including additional keys within data rows will cause an error, and no data will be published to the screener. Including fewer keys within data rows will cause unspecified columns to have an empty value for this asset. This complete list of dictionaries is then inserted as the value of the `'rows'` key in an outer dictionary. 
# 
# 
# In addition to column name-value pairs within row dictionaries, users can also optionally include a unique `'id'` key and value in each row of data. This `'id'` key allows users to overwrite specific rows with new data after earlier data has already been published to the screener. If no `'id'` key is specified when data is first published, each row will automatically be given a unique ID that can be retrieved when the screener's data is viewed. If a data row is re-published with the same `'id'` value, this data row will be overwritten in favor of newer data. If a data row is re-published without specifying an ID, a new data row will be appended to the screener regardless of whether or not this data already exists in the base screener.  
# 
# 
# Below is an example of a correctly formatted data dictionary. 

# In[ ]:


# Example: Data Format

{
    'rows':[
    {
      'id': 0, # Specifies a unique ID 
      'Name': 'Obalon Therapeutics Inc',
      'Entity ID': 'ABDFDSCD',
      'BBID': 'DSE',
      'Impliedretailbuynotional': 4.0,
      'Notional': 450.3,
      'Implied / realized': 14.2
    },
    {
      'Name': 'Disney Inc', # Does not specify a unique ID
      'Entity ID': 'ABDFDSCe',
      'BBID': 'DIS',
      'Impliedretailbuynotional': 4.0,
      'Notional': 45.0,
      'Implied / realized': 14.0
    }
        
    ]
}

# ### Function `publish_to_screener()`

# Use function `publish_to_screener(screener_id, data)` to publish the specified data to the base screener with the corresponding screener ID. This function returns a copy of the data published, along with additional storage information. 
# 
# Any data that includes an `'id'` field will overwrite any existing row with the same `'id'` value or add a new row with this ID if none exists. Any data that is provided without a specified `'id'` value will be appended to any existing screener data. Publishing to a screener mulitple times will not erase any previously published data unless it is overwritten through the use of row IDs. 
# 
# **Note:** Only users who are members of the `PlotScreenerAdmins` group can use the `publish_to_screener()` function to add data to existing screeners.

# In[ ]:


# Usage Example: publish_to_screener(), clear_screener()

# Create Screener

rows = (ScreenerRow(entity_id='MAYDMZQXXVXD9JK7', entity_type='asset'),)

columns = (ScreenerColumn(column_name='Name', entity_parameter='name'),)

params = ScreenerParameters(rows=rows, columns=columns)

cron_schedule = '5 4 * * *' 

schema = Screener(name='My New Screener', parameters=params, cron_schedule=cron_schedule)

screener = GsBaseScreenerApi.create_screener(schema)

# Publish data to screener 

data = {
    'rows':[
    {
      'id': 0, # Specifies a unique ID 
      'Name': 'Obalon Therapeutics Inc',
    }
    ]
}

GsBaseScreenerApi.publish_to_screener(screener.id, data)

sleep(1) # Sleep to Ensure Viewing Consistency 

# Create an Empty Screen to View Screener Data 

screen_schema = AnalyticsScreen(name='My View', filter_parameters=FilterRequest(), base_screener=screener.id)

screen = GsDataScreenApi.create_screen(screen_schema)

view_data = GsDataScreenApi.filter_screen(screen.id, screen.filter_parameters)

print(f'Screener %s with Initial Data:\n' % (screener.id))

display(pd.DataFrame(view_data))

# Re-Publish to a Screener 

new_data = {
    'rows':[
    {
        'id': 0, # Overwrite this Row
        'Name': 'MY NEW NAME',
    }, 
    {
        'Name': 'Disney Inc'
    }
    ]
}

GsBaseScreenerApi.publish_to_screener(screener.id, new_data)

sleep(1)

# View Screener Data 

view_data = GsDataScreenApi.filter_screen(screen.id, screen.filter_parameters)

print(f'\nScreener %s with Re-Published Data:\n' % (screener.id))

display(pd.DataFrame(view_data))

# Clear Screener 

GsBaseScreenerApi.clear_screener(screener.id)

sleep(1)

view_data = GsDataScreenApi.filter_screen(screen.id, screen.filter_parameters)

print(f'\nCleared Screener %s Data:\n' % (screener.id))

display(pd.DataFrame(view_data))

# Delete Screen and Screener

GsDataScreenApi.delete_screen(screen.id)

GsBaseScreenerApi.delete_screener(screener.id) 


# ## Editing Base Screeners

# ### Function `edit_screener()`

# To modify the schema of a base screener, use the function `edit_screener(screener_id, screener)` with the screener ID of the screener to modify and a `Screener` object specifying the schema that this screener should now have. 
# 
# 
# It is best practice to clear the screener of any existing data before making modifications to its schema. Then, users can re-publish their data to take full advantage of the new schema without needing to manually adjust individual rows. 
# 
# 
# `edit_screener(screener_id, screener)` will throw an error if the screener ID provided does not match the ID field in the `Screener` object provided.
# 
# 
# **Note:** Only users who are members of the `PlotScreenerAdmins` group can use the `edit_screener()` function to modify existing screeners.

# In[ ]:


# Usage Example: edit_screener()

# Create Screener 

rows = (ScreenerRow(entity_id='MAYDMZQXXVXD9JK7', entity_type='asset'),)

columns = (ScreenerColumn(column_name='Name', entity_parameter='name'),)

params = ScreenerParameters(rows=rows, columns=columns)

cron_schedule = '5 4 * * *'

schema = Screener(name='My New Screener', parameters=params, cron_schedule=cron_schedule)

screener = GsBaseScreenerApi.create_screener(schema)

print(f'Screener ID: %s, Columns: %s\n' % (screener.id, [col.column_name for col in screener.parameters.columns]))

# Create Screen

screen_schema = AnalyticsScreen(name='My View', filter_parameters=FilterRequest(), base_screener=screener.id)

screen = GsDataScreenApi.create_screen(screen_schema)

# Publish Data and View

data = {
    'rows':[
    {
        'id': 0, 
        'Name': 'Obalon Therapeutics Inc',
    }
    ]
}

GsBaseScreenerApi.publish_to_screener(screener.id, data)

sleep(1)

view_data = GsDataScreenApi.filter_screen(screen.id, screen.filter_parameters)

print(f'Screener %s Data:\n' % (screener.id))

display(pd.DataFrame(view_data))

# Clear Data 

GsBaseScreenerApi.clear_screener(screener.id)

# Edit Screener Schema (Columns)

screener.parameters.columns = screener.parameters.columns + (ScreenerColumn(column_name='BBID',entity_parameter='bbid'),)
                                                             
updated_screener = GsBaseScreenerApi.edit_screener(screener.id, screener)

# View Screener Columns 

print(f'\nScreener ID: %s, Columns: %s' % (updated_screener.id, [col.column_name for col in updated_screener.parameters.columns]))

# Publish Data and View 

data = {
    'rows':[
    {
        'id': 0, 
        'Name': 'Obalon Therapeutics Inc',
        'BBID': 'DSE'
    }
    ]
}

GsBaseScreenerApi.publish_to_screener(screener.id, data)

sleep(1)

view_data = GsDataScreenApi.filter_screen(screen.id, screen.filter_parameters)

print(f'\nScreener %s Data:\n' % (updated_screener.id))

display(pd.DataFrame(view_data))

# Delete Screen and Screener

GsDataScreenApi.delete_screen(screen.id)

GsBaseScreenerApi.delete_screener(updated_screener.id)


# If an additional column is added to the schema without clearing the screener, all previously existing rows will have no value in the new column. The old data in these rows will remain the same. To fill in data for existing rows in the new column, users will need to manually re-publish data for all columns of each existing row using the corresponding row IDs. This method will completely overwrite the data in each specified row to better fit the new schema. 

# In[ ]:


# Example: Manually Editing Rows

# Create Screener 

rows = (ScreenerRow(entity_id='MAYDMZQXXVXD9JK7', entity_type='asset'),)

columns = (ScreenerColumn(column_name='Name', entity_parameter='name'),)

params = ScreenerParameters(rows=rows, columns=columns)

cron_schedule = '5 4 * * *'

schema = Screener(name='My New Screener', parameters=params, cron_schedule=cron_schedule)

screener = GsBaseScreenerApi.create_screener(schema)

print(f'Screener ID: %s, Columns: %s\n' % (screener.id, [col.column_name for col in screener.parameters.columns]))

# Create Screen

screen_schema = AnalyticsScreen(name='My View', filter_parameters=FilterRequest(), base_screener=screener.id)

screen = GsDataScreenApi.create_screen(screen_schema)

# Publish Data and View

data = {
    'rows':[
    {
        'id': 0, 
        'Name': 'Obalon Therapeutics Inc',
    }
    ]
}

GsBaseScreenerApi.publish_to_screener(screener.id, data)

sleep(1)

view_data = GsDataScreenApi.filter_screen(screen.id, screen.filter_parameters)

print(f'Screener %s Data:\n' % (screener.id))

display(pd.DataFrame(view_data))

# Edit Screener Schema 

screener.parameters.columns = screener.parameters.columns + (ScreenerColumn(column_name='BBID',entity_parameter='bbid'),)
                                                             
updated_screener = GsBaseScreenerApi.edit_screener(screener.id, screener)

sleep(1)

# View Screener Columns 

print(f'\nScreener ID: %s, Columns: %s' % (updated_screener.id, [col.column_name for col in updated_screener.parameters.columns]))

# Publish Additional Data

data = {
    'rows':[
    {
        'id': 1,
        'Name': 'Disney Inc',
        'BBID': 'DIS'
    }
    ]
}

GsBaseScreenerApi.publish_to_screener(updated_screener.id, data)

sleep(1)

# View Screener Data

view_data = GsDataScreenApi.filter_screen(screen.id, screen.filter_parameters)

print(f'\nScreener %s Data:\n' % (updated_screener.id))

display(pd.DataFrame(view_data))

# Delete Screen and Screener

GsDataScreenApi.delete_screen(screen.id)

GsBaseScreenerApi.delete_screener(updated_screener.id)

