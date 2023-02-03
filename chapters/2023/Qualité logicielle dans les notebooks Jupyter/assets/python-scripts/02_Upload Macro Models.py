#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gs_quant.session import Environment, GsSession
# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(Environment.PROD, client_id=None, client_secret=None)

# # Macro Models
# 
# The GS Quant `MacroRiskModel` class gives users the power to upload their own risk models to Marquee for seamless integration with the Marquee Portfolio Analytics and Plot Tool Pro suite. After uploading a custom `MacroRiskModel`, users can access their Macro model data programmatically using GS Quant.

# ## Create a Macro Model
# 
# Input fields to create the initial Macro Risk Model object
# 
# | Attribute       |Can be Modified    |Description
# |-----------------|-------------------|-------------
# | id              | No                |Model id|
# | name            | Yes               |Name of model|
# | description     | Yes               |Longer description of model|
# | term            | Yes               |Term or horizon of model. One of: Long, Medium, Short|
# | coverage        | Yes               |Geographical coverage of assets within model universe. One of: Global, Region, Region Excluding Countries, Country|
# | vendor          | Yes               |Who creates the model|
# | version         | Yes               |Version of model|
# | identifier      | No                |Identifier used to upload the model's asset universe. One of: sedol, cusip, bcid, gsid|
# | entitlements    | Yes               |Who can manage, edit, and view the risk model|
# 

# In[ ]:


from gs_quant.models.risk_model import MacroRiskModel, RiskModelCalendar, Term, CoverageType, UniverseIdentifier

model_id = 'MY_MODEL'
model_name = 'My Risk Model'
description = 'My Custom Macro Risk Model'
term = Term.Medium
coverage = CoverageType.Country
universe_identifier = UniverseIdentifier.sedol
vendor = 'Goldman Sachs'

# In[ ]:


# create model with inputs

model = MacroRiskModel(
    id_=model_id,
    name=model_name,
    description=description,
    coverage=coverage,
    term=term,
    universe_identifier=universe_identifier,
    vendor=vendor,
    version=1,
)

model.save()

# ## Upload a Calendar To Your Model
# The calendar associated with the Macro Risk Model contains the dates which the risk model should have posted data on to be considered "complete." The calendar can go further back as well as forward in time than the data that is currently posted for the calendar, but there cannot be any gaps in the data posted to the risk model according to the calendar.

# In[ ]:


calendar = RiskModelCalendar([
    '2021-01-29', '2021-01-28', '2021-01-27', '2021-01-26', '2021-01-25', '2021-01-22', '2021-01-21',
    '2021-01-20', '2021-01-19', '2021-01-18', '2021-01-15', '2021-01-14', '2021-01-13', '2021-01-12',
    '2021-01-11', '2021-01-08', '2021-01-07', '2021-01-06', '2021-01-05', '2021-01-04', '2021-01-01'
])

model.upload_calendar(calendar)

# ## Upload Data To Your Model
# 
# Once the calendar is posted for a model, we can start uploading data to it. We can supply data multiple ways:
# 
# 1. Upload total data one day at a time
# 2. Upload partial data one day at a time
# 
# For a complete day of data, we need three things, defined in `RiskModelData`
# 1. Factor Data
#    - factorId: Can be any string, but needs to map consistently to the same factor across every date
#    - factorName: Can be any string, will be the display name of the factor, should be consistent across every date
#    - factorCategoryId: Id of the category that the factor belongs to
#    - factorCategory: Name of the category that the factor belongs to, will be the display name of the category (Style, Industry, Market, Currency, ect.)
#    - factorReturn: Daily return of the factor in percent units
# 2. Asset Data
#    - universe: Array of assets in the universe
#    - factorExposure: Array of dictionaries that map factorId to the factor exposure of that asset, corresponds to ordering of asset universe
#    - specificRisk: Array of annualized specific risk in percent units, corresponds to ordering of asset universe (null values not allowed)
#    - totalRisk: (optional) Array of total risk in percent units, corresponds to ordering of asset universe (null values not allowed)
#    - historicalBeta: (optional) Array of historical beta, corresponds to ordering of asset universe (null values not allowed)
# 
# ### Upload Full Data

# In[ ]:


data = {
    'date': '2021-01-13',  # Note: You can only upload to dates in your risk model's calendar
    'assetData': {
        'universe': ['B02V2Q0', '6560713', 'B3Q15X5', '0709954'],
        'specificRisk': [12.09, 45.12, 3.09, 1.0],
        'factorExposure': [
            {'1': 0.23, '2': 0.023},
            {'1': 0.023, '2': 2.09, '3': 0.3},
            {'1': 0.063, '2': 2.069, '3': 0.73},
            {'2': 0.067, '3': 0.93}
        ],
        'totalRisk': [12.7, 45.5, 12.7, 10.3]
    },
    'factorData': [
        {
            'factorId': '1',
            'factorName': 'USD',
            'factorCategory': 'Currency',
            'factorCategoryId': 'CUR',
            'factorReturn': 0.5
        },
        {
            'factorId': '2',
            'factorName': 'JPY 1Y Basis Swap',
            'factorCategory': 'GDP',
            'factorCategoryId': 'GDP',
            'factorReturn': 0.3
        },
        {
            'factorId': '3',
            'factorName': 'US HY',
            'factorCategory': 'Credit Spreads',
            'factorCategoryId': 'CDS',
            'factorReturn': 0.2
        }
    ]
}

model.upload_data(data)

# ## Query Data From Model
# 
# Once the data is uploaded, you can query it back using the same class

# In[ ]:


from gs_quant.models.risk_model import Measure, DataAssetsRequest
import datetime as dt

model = MacroRiskModel.get(model_id)
# get multiple measures across a date range for a universe specified
start_date = dt.date(2021, 1, 13)
end_date = dt.date(2021, 1, 13)

universe_for_request = DataAssetsRequest(universe_identifier.value, []) # an empty assets request returns the full universe
data_measures = [Measure.Universe_Factor_Exposure,
                 Measure.Asset_Universe,
                 Measure.Specific_Risk,
                 Measure.Total_Risk,
                 Measure.Factor_Id,
                 Measure.Factor_Name,
                 Measure.Factor_Category,
                 Measure.Factor_Category_Id,
                 Measure.Factor_Return
                ]

macro_factor_data = model.get_data(data_measures, start_date, end_date, universe_for_request, limit_factors=True)


